"""Resume the original N5 arms without rerunning completed epochs; --check-only uses CPU."""
import argparse,json,os,subprocess,sys,time
import numpy as np
import torch
from run_n5 import ROOT,OUT,T,configs,sha,trainer_argv,save

def comparisons():
    arms=configs()
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

def main():
    ap=argparse.ArgumentParser(__doc__)
    ap.add_argument('--check-only',action='store_true')
    options=ap.parse_args();os.chdir(ROOT)
    manifest=json.loads((OUT/'run_manifest.json').read_text())
    identity=manifest['identity'];arms=configs()
    assert arms==identity['configs'],'N5 configurations changed'
    for name,expected in identity['source_sha256'].items():
        assert sha(ROOT/'src'/name)==expected,f'Original N5 source changed: {name}'
    for name,expected in identity['input_sha256'].items():
        assert sha(ROOT/name)==expected,f'N5 input changed: {name}'
    completed={}
    for args in arms:
        name=args['out'].split('/')[-1];path=ROOT/args['out']/'last.pt'
        ck=torch.load(path,map_location='cpu',weights_only=False)
        assert {k:v for k,v in ck['args'].items() if v is not None}=={k:v for k,v in args.items() if v is not None}
        assert len(ck['log_epochs'])==ck['epoch']
        assert ck['sched']['last_epoch']==ck['epoch']*1714
        completed[name]=ck['epoch']
        assert ck['opt']['state'] and ck['scaler']
        print(f"Verified {name}: epoch {ck['epoch']}/16; optimizer, cosine scheduler and AMP state present",flush=True)
        del ck
    if options.check_only:return
    save('resume_manifest.json',dict(time=time.time(),completed=completed,original_identity=identity))
    try:
        for args in arms:
            name=args['out'].split('/')[-1]
            if completed[name]>=16:continue
            save('run_status.json',dict(state='training',arm=name,pid=os.getpid(),resumed_after=completed[name]))
            with (ROOT/f'outputs/{name}.log').open('a') as out,(ROOT/f'outputs/{name}.stderr.log').open('a') as err:
                result=subprocess.run([sys.executable,'-u','src/train_v3.py',*trainer_argv(args)],cwd=ROOT,stdout=out,stderr=err)
            if result.returncode:raise RuntimeError(f'{name} resume failed: {result.returncode}')
        comparisons();save('run_status.json',dict(state='completed',pid=os.getpid()))
    except BaseException as exc:
        save('run_status.json',dict(state='failed',error=str(exc)));raise

if __name__=='__main__':main()
