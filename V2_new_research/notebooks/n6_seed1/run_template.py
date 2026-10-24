"""Private offline N6: independent matched seed1 arms on two GPUs, bounded runtime."""
import hashlib,json,os,shutil,subprocess,sys,time,signal
from pathlib import Path
START=time.monotonic()
WORK=Path('/kaggle/working')
SOURCES=__SOURCES__
WORKER_SOURCE=__WORKER_SOURCE__

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(16*1024*1024),b''):h.update(b)
    return h.hexdigest()

def save(path,data):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(data,indent=2));tmp.replace(path)

def worker(head):
    sys.path.insert(0,str(WORK/'src'))
    os.environ['HF_HUB_OFFLINE']='1'
    import numpy as np
    import torch
    import torch.nn.functional as F
    import timm
    import train_v3 as T
    from v3_model import V3,TIMM
    from run_n1 import trainer_argv
    bundle=Path(os.environ['N6_INPUT_DIR'])
    conf=json.loads((bundle/'config.json').read_text())
    args=conf['parent_args']|dict(cache=json.loads(os.environ['N6_CACHES']),head=head,out=str(WORK/f'outputs/n6_{head}_seed1'))
    init=torch.load(bundle/'imagenet_encoder.pt',map_location='cpu',weights_only=True)
    def make(dev,head='max',backbone='convnext_tiny'):
        assert backbone=='convnext_tiny'
        m=V3(head,backbone=backbone)
        # Match the parent's second model construction and RNG consumption; only load is offline.
        src=timm.create_model(TIMM[backbone],pretrained=False,num_classes=0)
        src.load_state_dict(init,strict=True);m.encoder.load_state_dict(src.state_dict(),strict=True)
        return m.to(dev).to(memory_format=torch.channels_last)
    T.make_model=make
    split=T.read_split();labels=T.read_labels('v3')
    studies=T.build_studies(T.read_index(args['cache']),split,3)
    tr=sorted(u for u in studies if split[u][1]=='candidate_train' and T.targets(labels[u],'soft')[1].any())
    va=sorted(u for u in studies if split[u][1]=='candidate_validation')
    assert len(tr)==3428 and len(va)==885
    assert not {split[u][0] for u in tr}&{split[u][0] for u in va}
    out=Path(args['out']);out.mkdir(parents=True,exist_ok=True)
    save(out/'manifest.json',dict(args=args,config_sha256=sha(bundle/'config.json'),environment=dict(python=sys.version,torch=torch.__version__,cuda=torch.version.cuda,timm=timm.__version__,numpy=np.__version__,gpu=torch.cuda.get_device_name(0)),cohort=dict(train=len(tr),validation=len(va)),init='Exact frozen ImageNet encoder; fresh task head'))
    # A short genuine-cache gradient check is discarded before the seeded fresh run.
    torch.manual_seed(1)
    ds=T.CacheDS(args['cache'],[(u,studies[u]) for u in tr[:2]],4,True,seed=1,mirror_left=True,slots=3)
    batch=[ds[(k,1)] for k in range(2)]
    x=torch.stack([b[1] for b in batch]).cuda();mask=torch.stack([b[2] for b in batch]).cuda()
    model=make(torch.device('cuda'),head)
    target,observed=zip(*(T.targets(labels[u],'soft') for u in tr[:2]))
    target=torch.tensor(np.stack(target),device='cuda');observed=torch.tensor(np.stack(observed),device='cuda')
    opt=T.make_optimizer(model,args['lr'],args['wd']);scaler=torch.amp.GradScaler('cuda')
    finite_steps=0
    for _ in range(10):
        opt.zero_grad(set_to_none=True)
        with torch.autocast('cuda',dtype=torch.float16):logits=model(T.prep(x,224),mask)
        loss=F.binary_cross_entropy_with_logits(logits.float(),target,reduction='none')[observed].mean()
        assert torch.isfinite(loss)
        scaler.scale(loss).backward();scaler.unscale_(opt)
        finite=all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
        if finite:
            assert model.head.weight.grad.abs().sum()>0
            assert any(p.grad is not None and p.grad.abs().sum()>0 for p in model.encoder.parameters())
            if head=='attn':assert model.att_w.weight.grad.abs().sum()>0
            finite_steps+=1
        scaler.step(opt);scaler.update()
        if finite_steps==3:break
    assert finite_steps==3
    save(out/'preflight.json',dict(passed=True,finite_updates=finite_steps,smoke_weights_discarded=True))
    del model,opt,scaler,x,mask;torch.cuda.empty_cache()
    print('PREFLIGHT PASSED; starting fresh seed1',head,flush=True)
    T.main(trainer_argv(args))
    save(out/'run_status.json',dict(state='completed',epochs=12))

def comparisons(bundle):
    sys.path.insert(0,str(WORK/'src'))
    import numpy as np
    from scipy.stats import rankdata
    from compare_v3 import main as compare,read_preds
    from m00_model import TARGETS
    import csv
    a=WORK/'outputs/n6_max_seed1/val_pred_epoch12.csv';b=WORK/'outputs/n6_attn_seed1/val_pred_epoch12.csv'
    for label in ('public','v3'):
        compare([str(a),str(b),'--labels',label,'--out',str(WORK/f'seed1_attention_vs_max_{label}.json')])
        compare([str(bundle/'e8_seed0.csv'),str(bundle/'n1_seed0.csv'),'--labels',label,'--out',str(WORK/f'seed0_attention_vs_max_{label}.json')])
    # Existing base contains seed0 N1. One fixed 20% seed1 attention addition; no weight fitting.
    base=bundle/'base.csv';aa,bb=read_preds(base),read_preds(b);uids=sorted(aa);assert aa.keys()==bb.keys()
    p=.8*np.array([aa[u] for u in uids])+.2*rankdata(np.array([bb[u] for u in uids]),axis=0)/len(uids)
    blend=WORK/'base80_seed1attn20.csv'
    with blend.open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['StudyInstanceUID',*TARGETS]);w.writerows([u,*v] for u,v in zip(uids,p))
    compare([str(base),str(blend),'--out',str(WORK/'ensemble_vs_base_public.json')])

def main():
    WORK.mkdir(exist_ok=True)
    src=WORK/'src';src.mkdir(exist_ok=True)
    for name,text in SOURCES.items():(src/name).write_text(text)
    bundle=next(p.parent for p in Path('/kaggle/input').rglob('config.json') if (p.parent/'imagenet_encoder.pt').exists())
    conf=json.loads((bundle/'config.json').read_text())
    for name,expected in conf['files_sha256'].items():assert sha(bundle/name)==expected,name
    for name,text in SOURCES.items():assert hashlib.sha256(text.encode()).hexdigest()==conf['source_sha256'][name],name
    dests={'split.csv':'outputs/kaggle_split_source/split.csv','public.csv':'outputs/public_label_candidate/llm_labels_full.csv','v3.csv':'outputs/labels_v3/labels_v3.csv'}
    for name,dest in dests.items():
        dest=WORK/dest;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(bundle/name,dest)
    import numpy as np
    caches={}
    for path in Path('/kaggle/input').rglob('summary.json'):
        s=json.loads(path.read_text())
        if s.get('nshard')!=5 or s.get('pilot') is not False:continue
        k=s['shard'];folder=path.parent
        assert sha(folder/'index.csv')==conf['cache_index_sha256'][str(k)]
        a=np.load(folder/'slices.npy',mmap_mode='r')
        assert a.shape[1:]==(256,256) and a.dtype==np.uint8 and a.nbytes==s['slices_bytes']
        assert k not in caches
        caches[k]=str(folder)
    assert sorted(caches)==list(range(5)),caches
    import torch
    assert torch.cuda.device_count()>=2,'N6 parallel paired run requires two GPUs; no single-GPU fallback consumes the budget'
    save(WORK/'n6_manifest.json',dict(config=conf,gpus=[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],source='Bundled frozen project sources',budget_seconds=conf['max_wall_seconds'],comparison='N1 attention versus E8 max, seed1 matched on Kaggle hardware; cross-hardware differences recorded'))
    (WORK/'n6_worker.py').write_text(WORKER_SOURCE)
    jobs=[]
    for gpu,head in enumerate(('max','attn')):
        env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',HF_HUB_OFFLINE='1',N6_INPUT_DIR=str(bundle),N6_CACHES=json.dumps([caches[k] for k in range(5)]))
        log=(WORK/f'{head}.log').open('w');err=(WORK/f'{head}.stderr.log').open('w')
        p=subprocess.Popen([sys.executable,'-u',str(WORK/'n6_worker.py'),head],cwd=WORK,env=env,stdout=log,stderr=err,start_new_session=True)
        jobs.append((head,p,log,err))
    save(WORK/'n6_status.json',dict(state='running',arms={h:p.pid for h,p,_,_ in jobs}))
    deadline=START+conf['max_wall_seconds']
    while any(p.poll() is None for _,p,_,_ in jobs) and time.monotonic()<deadline:
        progress={}
        for head,p,_,_ in jobs:
            path=WORK/f'outputs/n6_{head}_seed1/log.json'
            try:progress[head]=len(json.loads(path.read_text())['epochs'])
            except (OSError,ValueError):progress[head]=0
            lp=WORK/f'{head}.log'
            lines=lp.read_text().splitlines() if lp.exists() else []
            live=next((line for line in reversed(lines) if line.startswith('ep ')),None)
            if live:progress[head+'_live']=live
        print('N6 PROGRESS',json.dumps(progress),flush=True)
        if any(p.poll() not in (None,0) for _,p,_,_ in jobs):break
        time.sleep(30)
    failed={h:p.returncode for h,p,_,_ in jobs if p.poll() not in (None,0)}
    partial=False
    for head,p,log,err in jobs:
        if p.poll() is None:
            partial=True;os.killpg(p.pid,signal.SIGTERM)
            try:p.wait(timeout=15)
            except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
        log.close();err.close()
    if failed:
        save(WORK/'n6_status.json',dict(state='failed',returncodes=failed))
        raise RuntimeError('N6 arm failed; see arm stderr')
    elif partial:
        save(WORK/'n6_status.json',dict(state='partial_budget_limited',note='Last complete epoch checkpoints retained; no final12 claim or comparisons'))
        print('N6 stopped within budget; partial checkpoints retained',flush=True)
    elif any(p.returncode!=0 for _,p,_,_ in jobs):
        save(WORK/'n6_status.json',dict(state='failed',returncodes={h:p.returncode for h,p,_,_ in jobs}))
        raise RuntimeError('N6 arm failed; see arm stderr')
    else:
        comparisons(bundle);save(WORK/'n6_status.json',dict(state='completed',epochs=12,arms=2,elapsed_seconds=time.monotonic()-START))
        print('N6 COMPLETED WITH FIXED COMPARISONS',flush=True)

if __name__=='__main__':
    if len(sys.argv)>1 and sys.argv[1]=='--worker':worker(sys.argv[2])
    else:main()
