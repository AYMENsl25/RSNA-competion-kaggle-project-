"""Generate three standalone, competition-only fresh baseline notebooks."""
from pathlib import Path
import textwrap
import nbformat as nbf

OUT = Path(__file__).resolve().parents[1] / 'notebooks' / 'fresh'
COMMON = '''
from pathlib import Path
import json, time, hashlib
import numpy as np
import pandas as pd
ROOTS = ['/kaggle/input/datasets/aymenslimani', '/kaggle/input']
def locate(name):
    hits = sorted(set(p.resolve() for root in ROOTS if Path(root).exists() for p in Path(root).rglob(name)))
    if not hits: raise FileNotFoundError(f'Attach the saved previous notebook output containing {name}')
    if len(hits)>1 and len({hashlib.sha256(p.read_bytes()).hexdigest() for p in hits})>1:
        raise ValueError(f'Conflicting inputs; select one attachment: {hits}')
    return hits[0]
TARGETS = ['ACL','MCL','Medial Meniscus','Lateral Meniscus','Medial OA','Lateral OA','PF OA','Effusion','Synovitis',"Baker's",'Contusion','Fracture']
'''

def save(name, intro, cells):
    nb = nbf.v4.new_notebook(cells=[nbf.v4.new_markdown_cell(intro), nbf.v4.new_code_cell(textwrap.dedent(COMMON).strip())])
    for kind, source in cells:
        source = textwrap.dedent(source).strip()
        nb.cells.append(nbf.v4.new_markdown_cell(source) if kind == 'md' else nbf.v4.new_code_cell(source))
    nb.metadata.kernelspec = dict(name='python3', display_name='Python 3', language='python')
    for i, cell in enumerate(nb.cells):
        if cell.cell_type == 'code': compile(cell.source, f'{name}:{i}', 'exec')
    nbf.validate(nb)
    OUT.mkdir(parents=True, exist_ok=True)
    nbf.write(nb, OUT / name)
    print(name, 'PASS schema and Python syntax')

save('01_Preparation_Labels_and_Audit.ipynb', '''# Fresh preparation, report labels and split audit
CPU. Attach ONLY the original RSNA competition. No previous pseudo labels, folds, slot plans or checkpoints.
This fast audit checks series directories, not every DICOM file. The report rules are a transparent preliminary baseline, not validated clinical labels. Review label_audit.csv before training. Save & Run All; attach this output to notebooks 02 and 03.''', [
('code', '''
import re
from sklearn.model_selection import GroupKFold
COMP = next((Path(p) for p in ['/kaggle/input/competitions/rsna-knee-abnormality-detection','/kaggle/input/rsna-knee-abnormality-detection'] if (Path(p)/'train.csv').exists()), None)
if COMP is None: raise FileNotFoundError('Add the RSNA competition input')
OUT = Path('/kaggle/working/rsna_fresh_preparation'); OUT.mkdir(exist_ok=True)
N_FOLDS = 5
UID = 'StudyInstanceUID'
train = pd.read_csv(COMP/'train.csv', dtype=str)
series = pd.read_csv(COMP/'train_series.csv', dtype=str)
print('Report columns:', train.columns.tolist())
print('Series columns:', series.columns.tolist())
REPORT_COLUMN = next((c for c in ['Report','report','ReportText','report_text'] if c in train), None)
if REPORT_COLUMN is None: raise ValueError('Set REPORT_COLUMN to the report-text column printed above')
if UID not in train or UID not in series or 'SeriesInstanceUID' not in series: raise ValueError('Missing study/series identifier columns')
if train[UID].isna().any() or train[UID].duplicated().any(): raise ValueError('Missing/duplicate report study identifiers')
if series[[UID,'SeriesInstanceUID']].isna().any().any(): raise ValueError('Missing series identifiers')
image_root = COMP/'train_series'
if not image_root.exists(): raise FileNotFoundError(image_root)
series['series_path'] = [str(image_root/u/s) for u,s in zip(series[UID],series.SeriesInstanceUID)]
t0=time.time()
series['folder_exists'] = [Path(p).is_dir() for p in series.series_path]
series.to_csv(OUT/'series_audit.csv',index=False)
print('Directory audit seconds:', round(time.time()-t0,1), 'missing:', int((~series.folder_exists).sum()))
print('Studies without series:', len(set(train[UID])-set(series.loc[series.folder_exists,UID])))
print('Empty reports:', int(train[REPORT_COLUMN].fillna('').str.strip().eq('').sum()))
'''),
('md', '''Labels use explicit abnormality mentions and conservative local negation/uncertainty handling. Unmentioned targets receive zero weight, rather than being assumed negative. Negation scope is only a heuristic; contradictory mentions are masked and exported for review. Rules and report column are saved for reproducibility.'''),
('code', '''
RULES = {
 'ACL': r'(?:acl|anterior cruciate ligament).{0,65}(?:tear|ruptur|disrupt|sprain)|(?:tear|ruptur).{0,40}(?:acl|anterior cruciate)',
 'MCL': r'(?:mcl|medial collateral ligament).{0,65}(?:tear|sprain|injur|disrupt)',
 'Medial Meniscus': r'medial menisc.{0,65}(?:tear|extrus|macerat)|tear.{0,40}medial menisc',
 'Lateral Meniscus': r'lateral menisc.{0,65}(?:tear|extrus|macerat)|tear.{0,40}lateral menisc',
 'Medial OA': r'medial (?:compartment|tibiofemoral).{0,80}(?:chondr|cartilage loss|osteoarth|degenerat)',
 'Lateral OA': r'lateral (?:compartment|tibiofemoral).{0,80}(?:chondr|cartilage loss|osteoarth|degenerat)',
 'PF OA': r'(?:patellofemoral|patellar|trochlear).{0,80}(?:chondr|cartilage loss|osteoarth|degenerat)',
 'Effusion': r'(?:joint )?effusion', 'Synovitis': r'synovitis|synovial (?:thicken|hypertroph)',
 "Baker's": r'baker.?s? cyst|popliteal cyst', 'Contusion': r'contusion|bone bruise',
 'Fracture': r'fracture',
}
NEG = r'\\b(?:no|without|absent|negative for|no evidence of)\\b'
UNCERTAIN = r'\\b(?:possible|possibly|may|might|questionable|suspect|cannot exclude)\\b'
def label_report(text):
    sentences = re.split(r'[.;\\n]+', str(text).lower())
    ys, ws, states = [], [], []
    for target in TARGETS:
        evidence=[]
        for sentence in sentences:
            for match in re.finditer(RULES[target], sentence):
                context=sentence[max(0,match.start()-55):match.end()]
                evidence.append('uncertain' if re.search(UNCERTAIN,context) else 'negative' if re.search(NEG,context) else 'positive')
        kinds=set(evidence)
        state='unmentioned' if not kinds else 'conflict' if len(kinds)>1 else next(iter(kinds))
        ys.append(float(state=='positive')); ws.append(float(state in ['positive','negative'])); states.append(state)
    return ys,ws,states
rows=[]; audit=[]
for _,r in train.iterrows():
    y,w,states=label_report(r[REPORT_COLUMN] if pd.notna(r[REPORT_COLUMN]) else '')
    rows.append([r[UID]]+y+w)
    audit.append([r[UID],r[REPORT_COLUMN]]+states)
manifest=pd.DataFrame(rows,columns=[UID]+['y_'+str(i) for i in range(12)]+['w_'+str(i) for i in range(12)])
# Prefer patient identifiers if available. Otherwise group identical normalized reports.
# This fallback cannot prove that different reports belong to different patients.
patient_col=next((c for c in ['PatientID','patient_id','PatientId'] if c in train),None)
normalized=train[REPORT_COLUMN].fillna('').str.lower().str.replace(r'\\s+',' ',regex=True).str.strip()
manifest['group']=[hashlib.sha256(s.encode()).hexdigest() if s else 'empty_'+u for s,u in zip(normalized,train[UID])]
if patient_col:
    if train[patient_col].isna().any(): raise ValueError('Patient identifiers missing; resolve before freezing splits')
    manifest['group']=train[patient_col].values
manifest=manifest[manifest[UID].isin(series.loc[series.folder_exists,UID])].sort_values(UID).reset_index(drop=True)
manifest['fold']=-1
for fold,(_,val) in enumerate(GroupKFold(N_FOLDS).split(manifest,groups=manifest.group)):
    manifest.loc[val,'fold']=fold
assert manifest.groupby('group').fold.nunique().max()==1
manifest.to_csv(OUT/'fresh_manifest.csv',index=False)
pd.DataFrame(audit,columns=[UID,'report']+TARGETS).to_csv(OUT/'label_audit.csv',index=False)
coverage=pd.DataFrame({'target':TARGETS,'positive':manifest[['y_'+str(i) for i in range(12)]].sum().values,'supervised':manifest[['w_'+str(i) for i in range(12)]].sum().values})
display(coverage); display(manifest.groupby('fold').size())
meta={'targets':TARGETS,'rules':RULES,'report_column':REPORT_COLUMN,'group_method':patient_col or 'normalized_report_hash_fallback','manifest_sha256':hashlib.sha256((OUT/'fresh_manifest.csv').read_bytes()).hexdigest(),'n_folds':N_FOLDS,'scope':'heuristic report supervision; audit required'}
(OUT/'fresh_preparation.json').write_text(json.dumps(meta,indent=2))
print('Saved:',OUT, 'Review label_audit.csv; weak coverage requires improving rules before larger training.')
''')])

save('02_MRI_Preprocessing_Pilot.ipynb', '''# Fresh MRI cache pilot
CPU. Attach competition and saved notebook 01 output. Default: at most 120 studies per split, one series per study, 8 slices, 160 pixels. This is a speed baseline, not a full six-series production pipeline. Only selected slices are decoded. Save output for notebook 03.''', [
('code', '''
import importlib.util, subprocess, sys
if importlib.util.find_spec('pydicom') is None:
    subprocess.check_call([sys.executable,'-m','pip','install','pydicom']) # Internet required if missing
import pydicom
from PIL import Image
COMP=next((Path(p) for p in ['/kaggle/input/competitions/rsna-knee-abnormality-detection','/kaggle/input/rsna-knee-abnormality-detection'] if (Path(p)/'train.csv').exists()),None)
if COMP is None: raise FileNotFoundError('Attach competition')
manifest_path=locate('fresh_manifest.csv'); prep=manifest_path.parent
manifest=pd.read_csv(manifest_path,dtype={'StudyInstanceUID':str,'group':str})
series=pd.read_csv(COMP/'train_series.csv',dtype=str)
FOLD=0; MAX_PER_SPLIT=120; N_SLICES=8; SIZE=160
OUT=Path('/kaggle/working/rsna_fresh_cache'); OUT.mkdir(exist_ok=True)
chosen=pd.concat([manifest[manifest.fold!=FOLD].sample(n=min(MAX_PER_SPLIT,len(manifest[manifest.fold!=FOLD])),random_state=42),manifest[manifest.fold==FOLD].sample(n=min(MAX_PER_SPLIT,len(manifest[manifest.fold==FOLD])),random_state=42)])
# Prefer sagittal series when metadata is available; deterministic fallback by series UID.
series['priority']=0
if 'SeriesDescription' in series: series['priority']=~series.SeriesDescription.fillna('').str.contains('sag',case=False)
series=series.sort_values(['priority','SeriesInstanceUID']).drop_duplicates('StudyInstanceUID').set_index('StudyInstanceUID')
records=[]; errors=[]; start=time.time()
for _,row in chosen.iterrows():
    uid=row.StudyInstanceUID
    try:
        sid=series.loc[uid,'SeriesInstanceUID']; folder=COMP/'train_series'/uid/sid
        files=sorted(p for p in folder.iterdir() if p.is_file())
        headers=[]
        for p in files:
            try:
                d=pydicom.dcmread(p,stop_before_pixels=True)
                pos=getattr(d,'ImagePositionPatient',None); ori=getattr(d,'ImageOrientationPatient',None)
                z=float(np.dot(pos,np.cross(ori[:3],ori[3:]))) if pos is not None and ori is not None else float(getattr(d,'InstanceNumber',0))
                headers.append((z,str(p)))
            except Exception: pass
        headers.sort()
        if not headers: raise ValueError('No readable DICOM headers')
        anchors=np.unique(np.linspace(0,len(headers)-1,N_SLICES).round().astype(int)); images=[]
        for anchor in anchors:
            d=pydicom.dcmread(headers[anchor][1]); a=d.pixel_array.astype(np.float32)
            if a.ndim!=2: raise ValueError('Expected single-frame 2D MRI')
            a=a*float(getattr(d,'RescaleSlope',1))+float(getattr(d,'RescaleIntercept',0))
            lo,hi=np.percentile(a,[1,99]); a=np.clip((a-lo)/max(hi-lo,1e-6),0,1)
            if getattr(d,'PhotometricInterpretation','')=='MONOCHROME1': a=1-a
            images.append(np.asarray(Image.fromarray((a*255).astype('uint8')).resize((SIZE,SIZE))))
        arr=np.stack(images); np.savez_compressed(OUT/(uid+'.npz'),images=arr)
        records.append({'StudyInstanceUID':uid,'SeriesInstanceUID':sid,'file':uid+'.npz','n_slices':len(arr)})
    except Exception as exc: errors.append({'StudyInstanceUID':uid,'error':str(exc)})
    if (len(records)+len(errors))%20==0: print('Processed',len(records)+len(errors),'seconds',round(time.time()-start))
index=pd.DataFrame(records)
if index.empty: raise RuntimeError('No cache created; inspect DICOM support and input paths')
index.to_csv(OUT/'fresh_cache_index.csv',index=False)
chosen[chosen.StudyInstanceUID.isin(index.StudyInstanceUID)].to_csv(OUT/'fresh_cached_manifest.csv',index=False)
pd.DataFrame(errors,columns=['StudyInstanceUID','error']).to_csv(OUT/'decode_errors.csv',index=False)
(OUT/'fresh_cache.json').write_text(json.dumps({'fold':FOLD,'size':SIZE,'n_slices':N_SLICES,'max_per_split':MAX_PER_SPLIT,'manifest_sha256':hashlib.sha256(manifest_path.read_bytes()).hexdigest(),'preparation':json.loads((prep/'fresh_preparation.json').read_text()),'recipe':'one series, geometric slice ordering, percentile full FOV','seconds':time.time()-start},indent=2))
print('Cached',len(index),'failed',len(errors),'seconds',round(time.time()-start), 'Saved',OUT)
'''),
('code', '''
import matplotlib.pyplot as plt
a=np.load(OUT/index.iloc[0]['file'])['images']
fig,axes=plt.subplots(1,len(a),figsize=(16,3))
for ax,img in zip(np.atleast_1d(axes),a): ax.imshow(img,cmap='gray'); ax.axis('off')
plt.show()
print('Inspect montage and decode_errors.csv before training; this pilot omits multi-series information.')
''')])

save('03_Small_Baseline_Training.ipynb', '''# Fresh small MRI baseline
GPU recommended. Attach saved notebook 02 output only. A compact CNN starts from scratch, averages slice predictions, and uses masked report-label BCE. No weight downloads. This is a pipeline smoke baseline; validation measures agreement with heuristic labels, not a competition score.''', [
('code', '''
import torch
from torch import nn
from torch.utils.data import Dataset,DataLoader
from sklearn.metrics import roc_auc_score
torch.manual_seed(42); np.random.seed(42)
INDEX=locate('fresh_cache_index.csv'); CACHE=INDEX.parent
meta=json.loads((CACHE/'fresh_cache.json').read_text()); FOLD=meta['fold']
df=pd.read_csv(CACHE/'fresh_cached_manifest.csv',dtype={'StudyInstanceUID':str,'group':str})
EPOCHS=3; BATCH_SIZE=8; NUM_WORKERS=0; LR=1e-3
OUT=Path('/kaggle/working/rsna_fresh_baseline'); OUT.mkdir(exist_ok=True)
yc=['y_'+str(i) for i in range(12)]; wc=['w_'+str(i) for i in range(12)]
tr=df[df.fold!=FOLD].reset_index(drop=True); va=df[df.fold==FOLD].reset_index(drop=True)
if tr.empty or va.empty: raise ValueError('Cache must contain train and validation studies')
assert not set(tr.group)&set(va.group)
class MRI(Dataset):
    def __init__(self,frame): self.frame=frame
    def __len__(self): return len(self.frame)
    def __getitem__(self,i):
        r=self.frame.iloc[i]; a=np.load(CACHE/(r.StudyInstanceUID+'.npz'))['images']
        ids=np.linspace(0,len(a)-1,meta['n_slices']).round().astype(int)
        return torch.tensor(a[ids,None]/255.,dtype=torch.float32),torch.tensor(r[yc].to_numpy(dtype=float),dtype=torch.float32),torch.tensor(r[wc].to_numpy(dtype=float),dtype=torch.float32)
train_loader=DataLoader(MRI(tr),batch_size=BATCH_SIZE,shuffle=True,num_workers=NUM_WORKERS)
val_loader=DataLoader(MRI(va),batch_size=BATCH_SIZE,num_workers=NUM_WORKERS)
class Baseline(nn.Module):
    def __init__(self):
        super().__init__(); layers=[]; channels=1
        for width in [16,32,64]:
            layers += [nn.Conv2d(channels,width,3,stride=2,padding=1),nn.ReLU(),nn.Conv2d(width,width,3,padding=1),nn.ReLU()]; channels=width
        self.encoder=nn.Sequential(*layers,nn.AdaptiveAvgPool2d(1)); self.head=nn.Linear(64,12)
    def forward(self,x):
        b,s,c,h,w=x.shape
        return self.head(self.encoder(x.reshape(b*s,c,h,w)).flatten(1).reshape(b,s,-1).mean(1))
device='cuda' if torch.cuda.is_available() else 'cpu'; model=Baseline().to(device); opt=torch.optim.AdamW(model.parameters(),lr=LR)
active=torch.tensor(tr[wc].sum().values>0,device=device)
if not active.any(): raise ValueError('No supervised training labels; improve report rules')
print('Device',device,'train',len(tr),'validation',len(va),'active targets',int(active.sum()))
def loss_fn(logits,y,w):
    w=w*active
    return (nn.functional.binary_cross_entropy_with_logits(logits,y,reduction='none')*w).sum(),w.sum()
history=[]; best=float('inf'); start=time.time()
for epoch in range(EPOCHS):
    model.train(); ts=tw=0.
    for x,y,w in train_loader:
        x,y,w=x.to(device),y.to(device),w.to(device); opt.zero_grad(); num,den=loss_fn(model(x),y,w)
        if den==0: continue
        (num/den).backward(); opt.step(); ts+=num.item(); tw+=den.item()
    model.eval(); vs=vw=0.; predictions=[]
    with torch.no_grad():
        for x,y,w in val_loader:
            logits=model(x.to(device)); num,den=loss_fn(logits,y.to(device),w.to(device)); vs+=num.item(); vw+=den.item(); predictions.append(logits.sigmoid().cpu().numpy())
    if vw==0: raise ValueError('No validation supervision for trained targets')
    value=vs/vw; history.append({'epoch':epoch+1,'train_bce':ts/max(tw,1),'val_bce':value,'seconds':time.time()-start})
    print(history[-1])
    if value<best:
        best=value; torch.save({'state_dict':model.state_dict(),'cache_metadata':meta,'targets':TARGETS,'epoch':epoch+1,'val_bce':best},OUT/'best.pt')
        pred=np.concatenate(predictions); result=va[['StudyInstanceUID','fold']].copy()
        for i,target in enumerate(TARGETS): result[target]=pred[:,i]
        result.to_csv(OUT/'validation_predictions.csv',index=False)
pd.DataFrame(history).to_csv(OUT/'history.csv',index=False)
metrics=[]; saved=pd.read_csv(OUT/'validation_predictions.csv')
for i,target in enumerate(TARGETS):
    mask=(va[wc[i]].values>0)&bool(active[i].item()); y=va[yc[i]].values[mask]; p=saved[target].values[mask]
    metrics.append({'target':target,'n_supervised':len(y),'auc':float(roc_auc_score(y,p)) if len(np.unique(y))==2 else None})
pd.DataFrame(metrics).to_csv(OUT/'target_metrics.csv',index=False); display(pd.DataFrame(metrics))
print('Saved',OUT,'Best masked BCE',best,'Missing AUC means no supervised positive/negative pair, or no training supervision.')
''')])
