"""N5: two fresh, matched 16-epoch N1 arms; only encoder/head LR differs."""
from pathlib import Path
import json
import os
import subprocess
import sys
import time
os.environ['HF_HUB_OFFLINE']='1'
os.environ['HF_HUB_DISABLE_TELEMETRY']='1'
import numpy as np
import torch
import torch.nn.functional as F
import train_v3 as T
from run_n1 import trainer_argv,sha
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/n5_lr'
SOURCES=('run_n5.py','train_v3.py','v3_model.py','m00_data.py','m00_model.py','p00_model.py','canon.py','run_n1.py','compare_v3.py')

def save(name,data):
    OUT.mkdir(parents=True,exist_ok=True)
    p=OUT/name;tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(data,indent=2));tmp.replace(p)

def configs():
    parent=json.loads((ROOT/'outputs/n1_attn_random/log.json').read_text())['args']
    assert parent['head']=='attn' and parent['slots']==3 and parent['res']==224
    base=parent|dict(epochs=16)
    return [base|dict(out='outputs/n5_uniform16'),
            base|dict(out='outputs/n5_differential16',encoder_lr=3e-5,head_lr=1e-3)]

def smoke(args):
    split,labels=T.read_split(),T.read_labels('v3')
    studies=T.build_studies(T.read_index(args['cache']),split,3)
    train=sorted(u for u in studies if split[u][1]=='candidate_train' and T.targets(labels[u],'soft')[1].any())
    val=sorted(u for u in studies if split[u][1]=='candidate_validation')
    assert len(train)==3428 and len(val)==885
    assert not {split[u][0] for u in train}&{split[u][0] for u in val}
    selected=train[:2]
    ds=T.CacheDS(args['cache'],[(u,studies[u]) for u in selected],4,True,seed=0,mirror_left=True,slots=3)
    batch=[ds[(k,1)] for k in range(2)]
    x=torch.stack([v[1] for v in batch]).cuda();mask=torch.stack([v[2] for v in batch]).cuda()
    torch.manual_seed(0);m=T.make_model(torch.device('cuda'),'attn','convnext_tiny')
    opt=T.make_optimizer(m,args['lr'],args['wd'],3e-5,1e-3)
    targets,observed=zip(*(T.targets(labels[u],'soft') for u in selected))
    target=torch.tensor(np.stack(targets),device='cuda');obs=torch.tensor(np.stack(observed),device='cuda')
    scaler=torch.amp.GradScaler('cuda');good=0;losses=[]
    for _ in range(10):
        opt.zero_grad(set_to_none=True)
        with torch.autocast('cuda',dtype=torch.float16):p=m(T.prep(x,224),mask)
        loss=F.binary_cross_entropy_with_logits(p.float(),target,reduction='none')[obs].mean()
        assert torch.isfinite(loss)
        scaler.scale(loss).backward();scaler.unscale_(opt)
        finite=all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None)
        if finite:
            assert m.att_w.weight.grad.abs().sum()>0
            assert any(p.grad is not None and p.grad.abs().sum()>0 for p in m.encoder.parameters())
            good+=1;losses.append(float(loss.detach()))
        scaler.step(opt);scaler.update()
        if good==3:break
    assert good==3
    save('preflight.json',dict(passed=True,losses=losses,train_studies=len(train),val_studies=len(val),note='Smoke weights discarded; both arms initialize fresh from cached ImageNet'))

def main():
    os.chdir(ROOT);OUT.mkdir(parents=True,exist_ok=True)
    arms=configs()
    assert not any((ROOT/a['out']/'last.pt').exists() for a in arms),'Existing N5 checkpoint; refusing overwrite'
    identity=dict(configs=arms,source_sha256={n:sha(ROOT/'src'/n) for n in SOURCES},
                  input_sha256={str(p.relative_to(ROOT)):sha(p) for p in [T.SPLIT,*T.LABELS.values(),ROOT/'outputs/n1_attn_random/log.json',*[ROOT/d/'index.csv' for d in arms[0]['cache']]]})
    snapshot=OUT/'source';snapshot.mkdir(exist_ok=True)
    for n in SOURCES:(snapshot/n).write_bytes((ROOT/'src'/n).read_bytes())
    save('run_manifest.json',dict(identity=identity,pid=os.getpid(),started=time.time(),
         git_head=subprocess.run(['git','rev-parse','HEAD'],capture_output=True,text=True).stdout.strip(),
         hypothesis='Differential vs uniform learning rates, matched 16 epochs; no EMA, schedule or augmentation changes',
         promotion='Paired final16 macro gain >=0.003 with positive CI, or >=0.002 fixed blend gain with positive CI'))
    save('run_status.json',dict(state='preflight',pid=os.getpid()))
    try:
        smoke(arms[1]);torch.cuda.empty_cache()
        for args in arms:
            name=Path(args['out']).name
            save('run_status.json',dict(state='training',arm=name,pid=os.getpid()))
            with (ROOT/f'outputs/{name}.log').open('w') as out,(ROOT/f'outputs/{name}.stderr.log').open('w') as err:
                p=subprocess.run([sys.executable,'-u','src/train_v3.py',*trainer_argv(args)],cwd=ROOT,stdout=out,stderr=err)
            if p.returncode:raise RuntimeError(f'{name} failed: exit {p.returncode}')
        from compare_v3 import main as compare
        uniform=ROOT/arms[0]['out']/'val_pred_epoch16.csv';candidate=ROOT/arms[1]['out']/'val_pred_epoch16.csv'
        for label in ('public','v3'):
            compare([str(uniform),str(candidate),'--labels',label,'--out',str(OUT/f'differential_vs_uniform_{label}.json')])
        compare([str(ROOT/'outputs/n1_attn_random/val_pred_epoch12.csv'),str(uniform),'--out',str(OUT/'uniform16_vs_n1_12_secondary.json')])
        # One fixed addition to the accepted N3a blend, without weight search.
        import csv
        from compare_v3 import read_preds
        from scipy.stats import rankdata
        base=ROOT/'outputs/n3a_attn6/base80_n3a20.csv'
        a,b=read_preds(base),read_preds(candidate);assert a.keys()==b.keys();uids=sorted(a)
        probs=.8*np.array([a[u] for u in uids])+.2*rankdata(np.array([b[u] for u in uids]),axis=0)/len(uids)
        blend=OUT/'base80_n5_20.csv'
        with blend.open('w',newline='') as f:
            writer=csv.writer(f);writer.writerow(['StudyInstanceUID',*T.TARGETS]);writer.writerows([u,*p] for u,p in zip(uids,probs))
        compare([str(base),str(blend),'--out',str(OUT/'ensemble_vs_base_public.json')])
        save('run_status.json',dict(state='completed',pid=os.getpid()))
    except BaseException as exc:
        save('run_status.json',dict(state='failed',error=str(exc),pid=os.getpid()));raise

if __name__=='__main__':main()
