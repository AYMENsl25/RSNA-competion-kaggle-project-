import os,sys,json,hashlib
from pathlib import Path
os.environ['HF_HUB_OFFLINE']='1'
sys.path.insert(0,'src')
import torch,timm
import train_v3 as T
from v3_model import V3,TIMM
torch.set_num_threads(2)
bundle=Path('outputs/n6_kaggle_inputs');conf=json.loads((bundle/'config.json').read_text())
for name,expected in conf['files_sha256'].items():assert hashlib.sha256((bundle/name).read_bytes()).hexdigest()==expected
init=torch.load(bundle/'imagenet_encoder.pt',map_location='cpu',weights_only=True)
for head in ('max','attn'):
    torch.manual_seed(1);original=T.make_model(torch.device('cpu'),head,'convnext_tiny');state=torch.get_rng_state()
    torch.manual_seed(1);offline=V3(head,backbone='convnext_tiny');src=timm.create_model(TIMM['convnext_tiny'],pretrained=False,num_classes=0);src.load_state_dict(init,strict=True);offline.encoder.load_state_dict(src.state_dict(),strict=True)
    assert all(torch.equal(v,offline.state_dict()[k]) for k,v in original.state_dict().items()),head
    assert torch.equal(state,torch.get_rng_state()),head
split=T.read_split()
for kind in ('public','v3'):
    import csv
    orig=T.read_labels(kind)
    rows=list(csv.DictReader((bundle/f'{kind}.csv').open()))
    assert all(split[r['StudyInstanceUID']][1] in T.PARTS for r in rows)
    assert all(all(float(r[t])==orig[r['StudyInstanceUID']][i] for i,t in enumerate(T.TARGETS)) for r in rows)
compile(Path('notebooks/n6_seed1/run.py').read_text(),'n6','exec')
print('N6 PASS: input SHA256s; exact parent initialization and RNG state for both heads; unchanged allowed-label values; no excluded label rows; notebook compiles')
