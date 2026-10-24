"""Build the private, offline N6 input bundle from the accepted N1/E8 comparison."""
import csv,hashlib,json,os,shutil,sys
from pathlib import Path
os.environ['HF_HUB_OFFLINE']='1'
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
import torch
import train_v3 as T
HERE=Path(__file__).resolve().parent
OUT=ROOT/'outputs/n6_kaggle_inputs'
SOURCES={n:(ROOT/'src'/n).read_text() for n in ('train_v3.py','m00_data.py','m00_model.py','p00_model.py','v3_model.py','canon.py','run_n1.py','compare_v3.py')}
args=json.loads((ROOT/'outputs/n1_attn_random/log.json').read_text())['args']
assert args['head']=='attn' and args['slots']==3 and args['res']==224 and args['epochs']==12
args.update(seed=1,workers=1,prefetch=1,log_every=100)
shutil.copy2(T.SPLIT,OUT/'split.csv')
split=T.read_split()
allowed={u for u,(_,part) in split.items() if part in T.PARTS}
for kind,path in T.LABELS.items():
    with path.open(newline='',encoding='utf-8-sig') as f:
        rows=list(csv.DictReader(f))
    with (OUT/f'{kind}.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,['StudyInstanceUID',*T.TARGETS],extrasaction='ignore');w.writeheader()
        w.writerows(r for r in rows if r['StudyInstanceUID'] in allowed)
for name,path in [('e8_seed0.csv','outputs/e8_cnxt/val_pred_epoch12.csv'),('n1_seed0.csv','outputs/n1_attn_random/val_pred_epoch12.csv'),('base.csv','outputs/n3a_attn6/base80_n3a20.csv')]:
    shutil.copy2(ROOT/path,OUT/name)
# Preserve exactly the cached ImageNet encoder used in the accepted parent.
model=T.make_model(torch.device('cpu'),'max','convnext_tiny')
torch.save(model.encoder.state_dict(),OUT/'imagenet_encoder.pt')
config=dict(parent_args=args,source_sha256={n:hashlib.sha256(s.encode()).hexdigest() for n,s in SOURCES.items()},
            cache_index_sha256={str(k):json.loads((ROOT/f'outputs/cache_v3/s{k}/summary.json').read_text())['sha256']['index.csv'] for k in range(5)},
            files_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in OUT.iterdir() if p.is_file() and p.name not in ('config.json','dataset-metadata.json')},
            hypothesis='N1 attention vs E8 max pooling, matched seed1, final epoch12; no hyperparameter search',
            max_wall_seconds=10080,init='Exact cached ConvNeXt-Tiny ImageNet encoder; no knee-pretrained checkpoints')
(OUT/'config.json').write_text(json.dumps(config,indent=2))
(OUT/'dataset-metadata.json').write_text(json.dumps(dict(title='RSNA knee N6 seed1 private inputs',id='aymenslimani/rsna-knee-n6-seed1-inputs',licenses=[dict(name='other')]),indent=2))
(HERE/'sources.json').write_text(json.dumps(SOURCES))
print('Prepared private inputs:',sum(p.stat().st_size for p in OUT.iterdir())/2**20,'MiB')
