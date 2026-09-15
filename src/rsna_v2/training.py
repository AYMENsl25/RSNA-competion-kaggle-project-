"""Group-safe MRI study training, resumable checkpoints, and shared inference."""
from pathlib import Path
import json
import random
import time
import shutil
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from .common import (UID, TARGETS, PSEUDO_COLS, WEIGHT_COLS, SLOTS,
                     fingerprint, hash_file, write_json, environment)


def load_checkpoint(path):
    # Only load your own locally prepared experiment checkpoints.
    return torch.load(path, map_location='cpu', weights_only=False)


def atomic_save(data,path):
    path=Path(path); temp=path.with_suffix('.tmp'); torch.save(data,temp); temp.replace(path)


def macro_loss(logits,y,w,mean_weight):
    raw=F.binary_cross_entropy_with_logits(logits.float(),y.float(),reduction='none')
    return ((raw*w).mean(0)/mean_weight).mean()


class StudyDataset(Dataset):
    def __init__(self,cache_dir,index,uids,config,training=False,labels=None):
        self.root=Path(cache_dir); self.uids=list(map(str,uids)); self.config=config
        self.training=training; self.labels=labels
        self.rows={str(uid):df.sort_values('slot') for uid,df in index.groupby(UID)}
        if set(self.uids)-set(self.rows): raise ValueError('Studies missing from cache')
        self.k=int(config['train_anchors'] if training else config['eval_anchors'])
        self.size=int(config.get('image_size',224))

    def __len__(self): return len(self.uids)

    def __getitem__(self,i):
        uid=self.uids[i]; images=np.zeros((6,self.k,3,self.size,self.size),np.float32)
        mask=np.zeros((6,self.k),bool)
        for _,r in self.rows[uid].iterrows():
            if r.slot not in SLOTS: raise ValueError(f'Unsupported slot {r.slot}')
            s=SLOTS.index(r.slot); path=(self.root/r['path']).resolve()
            if not path.is_relative_to(self.root.resolve()): raise ValueError('Cache path escapes root')
            with np.load(path,allow_pickle=False) as f:
                x=f['images']; good=np.flatnonzero(f['valid'])
                if x.dtype!=np.uint8 or x.ndim!=3 or x.shape[1:]!=(self.size,self.size):
                    raise ValueError('Cached images do not match model size or uint8 format')
                if len(good)==0: continue
                if self.training and len(good)>self.k:
                    chosen=np.sort(np.random.choice(len(good),self.k,replace=False))
                else: chosen=np.linspace(0,len(good)-1,min(self.k,len(good))).round().astype(int)
                for j,position in enumerate(chosen):
                    if self.config['input_mode']=='repeat': ids=[good[position]]*3
                    else: ids=good[np.clip([position-1,position,position+1],0,len(good)-1)]
                    images[s,j]=x[ids]/255.; mask[s,j]=True
        if not mask.any(): raise ValueError(f'All MRI inputs invalid for study {uid}')
        tensor=torch.from_numpy(images)
        tensor=(tensor-tensor.new_tensor([.485,.456,.406])[None,None,:,None,None])/tensor.new_tensor([.229,.224,.225])[None,None,:,None,None]
        result={'uid':uid,'images':tensor,'mask':torch.from_numpy(mask)}
        if self.labels is not None:
            row=self.labels.loc[uid]
            result.update(y=torch.tensor(row[PSEUDO_COLS].to_numpy(np.float32)),
                          w=torch.tensor(row[WEIGHT_COLS].to_numpy(np.float32)))
        return result


def create_encoder(config,initialize):
    family=config['model_family']
    if family=='resnet18':
        from torchvision.models import resnet18,ResNet18_Weights
        backbone=resnet18(weights=ResNet18_Weights.DEFAULT if initialize and config.get('allow_download') and not config.get('pretrained_path') else None)
        dim=512
    elif family=='dinov2':
        import timm
        backbone=timm.create_model('vit_small_patch14_dinov2.lvd142m',pretrained=bool(initialize and config.get('allow_download') and not config.get('pretrained_path')),img_size=config['image_size'])
        dim=backbone.num_features
    else: raise ValueError('Unknown model_family')
    if initialize:
        if config.get('pretrained_path'):
            path=Path(config['pretrained_path'])
            if path.suffix=='.safetensors':
                from safetensors.torch import load_file
                state=load_file(str(path))
            else: state=torch.load(path,map_location='cpu',weights_only=True)
            backbone.load_state_dict(state,strict=True)
        elif not config.get('allow_download'): raise ValueError('Attach matching pretrained weights or enable allow_download explicitly')
    if family=='resnet18': backbone.fc=nn.Identity()
    else: backbone.reset_classifier(0)
    return backbone,dim


class StudyModel(nn.Module):
    def __init__(self,config,initialize=True):
        super().__init__(); self.config=config
        self.encoder,dim=create_encoder(config,initialize)
        self.frozen=bool(config['frozen_encoder'])
        if self.frozen:
            for p in self.encoder.parameters(): p.requires_grad=False
        if config['pooling']=='meanmax':
            self.head=nn.Sequential(nn.Dropout(.2),nn.Linear(dim*2,12))
        else:
            self.score=nn.Linear(dim,12); self.value=nn.Linear(dim,12)
            self.slot_embedding=nn.Embedding(6,dim)

    def train(self,mode=True):
        super().train(mode)
        if self.frozen: self.encoder.eval()
        return self

    def forward(self,images,mask):
        b,s,k,c,h,w=images.shape; flat=images.reshape(-1,c,h,w); active=mask.reshape(-1)
        # Encode real slices only, so padded zeros never pollute BatchNorm.
        features=self.encoder(flat[active]); dim=features.shape[-1]
        all_features=features.new_zeros((len(flat),dim)); all_features[active]=features
        x=all_features.reshape(b,s,k,dim); m=mask.unsqueeze(-1)
        if self.config['pooling']=='meanmax':
            mean=(x*m).sum(2)/m.sum(2).clamp_min(1)
            maximum=x.masked_fill(~m,-torch.inf).max(2).values
            maximum=torch.where(torch.isfinite(maximum),maximum,torch.zeros_like(maximum))
            series=self.head(torch.cat([mean,maximum],-1)); available=mask.any(2)
            planes=[]; present=[]
            for a,z in [(0,2),(2,4),(4,6)]:
                pm=available[:,a:z].unsqueeze(-1)
                planes.append((series[:,a:z]*pm).sum(1)/pm.sum(1).clamp_min(1))
                present.append(available[:,a:z].any(1))
            planes=torch.stack(planes,1); present=torch.stack(present,1).unsqueeze(-1)
            return (planes*present).sum(1)/present.sum(1).clamp_min(1)
        x=x+self.slot_embedding(torch.arange(6,device=x.device))[None,:,None,:]
        scores=self.score(x).reshape(b,s*k,12).masked_fill(~mask.reshape(b,s*k,1),-torch.inf)
        weights=scores.softmax(1)
        return (self.value(x).reshape(b,s*k,12)*weights).sum(1)


def loader(cache_dir,index,uids,cfg,training=False,labels=None):
    ds=StudyDataset(cache_dir,index,uids,cfg,training,labels)
    return DataLoader(ds,batch_size=cfg['batch_size'],shuffle=training,num_workers=cfg['num_workers'],
                      pin_memory=torch.cuda.is_available(),persistent_workers=cfg['num_workers']>0)


@torch.inference_mode()
def predict_studies(model,cache_dir,cache_index,uids,config,device):
    model.eval(); output=[]
    for batch in loader(cache_dir,cache_index,uids,config):
        probabilities=model(batch['images'].to(device),batch['mask'].to(device)).sigmoid().cpu().numpy()
        output.extend({UID:uid,**dict(zip(TARGETS,map(float,p)))} for uid,p in zip(batch['uid'],probabilities))
    return pd.DataFrame(output)


def load_trained_model(checkpoint,device):
    ckpt=load_checkpoint(checkpoint)
    if ckpt.get('format')!='rsna_v2_1': raise ValueError('Expected v2 checkpoint, not legacy archive')
    model=StudyModel(ckpt['config'],initialize=False)
    model.load_state_dict(ckpt['model'],strict=True); model.to(device).eval()
    return model,ckpt['config']


def run_training(config):
    cfg={'model_family':'resnet18','pooling':'meanmax','input_mode':'repeat','train_anchors':8,
         'eval_anchors':16,'fold':0,'n_folds':5,'mode':'selection','epochs':12,'patience':3,
         'batch_size':2,'num_workers':2,'learning_rate':1e-4,'backbone_lr':1e-5,
         'frozen_encoder':False,'seed':2026,'allow_download':False,'require_cuda':True,
         'resume_path':None,**config}
    if cfg['pooling'] not in ['meanmax','attention'] or cfg['input_mode'] not in ['repeat','triplet']:
        raise ValueError('Unsupported pooling or input mode')
    if cfg['mode'] not in ['selection','refit'] or min(cfg['epochs'],cfg['batch_size'],cfg['train_anchors'],cfg['eval_anchors'])<1:
        raise ValueError('Invalid training mode or budget')
    if cfg['require_cuda'] and not torch.cuda.is_available(): raise RuntimeError('Enable Kaggle GPU accelerator')
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    random.seed(cfg['seed']); np.random.seed(cfg['seed']); torch.manual_seed(cfg['seed'])
    cache=Path(cfg['cache_dir']); sup=Path(cfg['supervision_dir']); out=Path(cfg['out_dir'])
    cm=json.loads((cache/'cache_meta.json').read_text()); sm=json.loads((sup/'supervision_meta.json').read_text())
    if not cm['complete'] or cm['smoke']: raise ValueError('Production training requires complete non-pilot cache')
    if hash_file(cache/'cache_index.csv')!=cm['index_sha256'] or hash_file(sup/'train_manifest.csv')!=sm['train_manifest_sha256']:
        raise ValueError('Input manifest hash mismatch')
    index=pd.read_csv(cache/'cache_index.csv',dtype={UID:str,'cv_group_id':str,'SeriesInstanceUID':str})
    labels=pd.read_csv(sup/'train_manifest.csv',dtype={UID:str,'cv_group_id':str}).set_index(UID)
    if not labels.training_role.eq('PSEUDO_TRAIN').all() or labels.index.duplicated().any(): raise ValueError('Invalid pseudo cohort')
    cache_roles=index[[UID,'cv_group_id','training_role']].drop_duplicates().set_index(UID)
    if cache_roles.index.duplicated().any() or set(labels.index)-set(cache_roles.index): raise ValueError('Cache cohort inconsistency')
    if not cache_roles.loc[labels.index,['cv_group_id','training_role']].equals(labels[['cv_group_id','training_role']]):
        raise ValueError('Cache and supervision groups/roles differ')
    if not index.preprocessing_id.eq(cm['preprocessing_id']).all(): raise ValueError('Mixed cache identities')
    if set(labels.fold)!=set(range(cfg['n_folds'])) or (labels.groupby('cv_group_id').fold.nunique()!=1).any():
        raise ValueError('Invalid grouped folds')
    train_ids=labels.index[labels.fold!=cfg['fold']].tolist() if cfg['mode']=='selection' else labels.index.tolist()
    val_ids=labels.index[labels.fold==cfg['fold']].tolist() if cfg['mode']=='selection' else []
    if not train_ids or (cfg['mode']=='selection' and not val_ids): raise ValueError('Empty train or validation split')
    cfg.update(image_size=cm['config']['size'],preprocessing_id=cm['preprocessing_id'],
               preprocessing_config=cm['config'],supervision_id=sm['supervision_id'])
    init_hash=hash_file(cfg['pretrained_path']) if cfg.get('pretrained_path') else 'download'
    identity=fingerprint({'config':{k:v for k,v in cfg.items() if k not in ['out_dir','resume_path','cache_dir','supervision_dir','pretrained_path']},
                          'pretrained_sha256':init_hash,'train_manifest':sm['train_manifest_sha256']})
    out.mkdir(parents=True,exist_ok=True)
    meta_path=out/'run_meta.json'
    if meta_path.exists() and json.loads(meta_path.read_text())['run_id']!=identity: raise ValueError('Output experiment mismatch')
    resume=load_checkpoint(cfg['resume_path']) if cfg['resume_path'] else None
    if resume and resume['run_id']!=identity: raise ValueError('Resume checkpoint experiment mismatch')
    model=StudyModel(cfg,initialize=resume is None).to(device)
    head=[p for n,p in model.named_parameters() if p.requires_grad and not n.startswith('encoder.')]
    encoder=[p for p in model.encoder.parameters() if p.requires_grad]
    groups=[{'params':head,'lr':cfg['learning_rate']}]
    if encoder: groups.append({'params':encoder,'lr':cfg['backbone_lr']})
    optimizer=torch.optim.AdamW(groups,weight_decay=1e-4)
    scaler=torch.amp.GradScaler('cuda',enabled=device.type=='cuda')
    history=[]; best=float('inf'); bad=0; start=0; best_epoch=0
    if resume:
        model.load_state_dict(resume['model']); optimizer.load_state_dict(resume['optimizer']); scaler.load_state_dict(resume['scaler'])
        history=resume['history']; best=resume['best']; bad=resume['bad']; start=resume['epoch']; best_epoch=resume['best_epoch']
        random.setstate(resume['rng_python']); np.random.set_state(resume['rng_numpy']); torch.set_rng_state(resume['rng_torch'])
        if device.type=='cuda' and resume.get('rng_cuda'): torch.cuda.set_rng_state_all(resume['rng_cuda'])
        if cfg['mode']=='selection':
            source=Path(cfg['resume_path']).parent
            for name in ['best.pt','validation_predictions.csv']:
                original=source/name
                if not original.is_file(): raise ValueError('Attach the complete previous run output (best.pt, last.pt, predictions) for resume')
                if original.resolve() != (out/name).resolve(): shutil.copyfile(original,out/name)
            if load_checkpoint(out/'best.pt')['run_id']!=identity: raise ValueError('Previous best checkpoint mismatch')
    state=resume
    mean=torch.tensor(labels.loc[train_ids,WEIGHT_COLS].to_numpy(float).mean(0),device=device,dtype=torch.float32)
    if (mean<=0).any(): raise ValueError('Zero active training target weight')
    train_loader=loader(cache,index,train_ids,cfg,True,labels)
    meta={'run_id':identity,'config':cfg,'complete':False,'mode':cfg['mode'],
          'supervision_id':sm['supervision_id'],'preprocessing_id':cm['preprocessing_id'],'environment':environment()}
    write_json(meta_path,meta)
    for epoch in range(start,cfg['epochs']):
        model.train(); total=0.; n=0; begin=time.time()
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            images=batch['images'].to(device); mask=batch['mask'].to(device); y=batch['y'].to(device); w=batch['w'].to(device)
            with torch.autocast(device_type=device.type,enabled=device.type=='cuda'):
                loss=macro_loss(model(images,mask),y,w,mean)
            if not torch.isfinite(loss): raise RuntimeError('Nonfinite training loss')
            scaler.scale(loss).backward(); scaler.unscale_(optimizer); nn.utils.clip_grad_norm_(model.parameters(),1.)
            scaler.step(optimizer); scaler.update(); total+=float(loss.detach())*len(y); n+=len(y)
        value=None
        if val_ids:
            pred=predict_studies(model,cache,index,val_ids,cfg,device).set_index(UID).loc[val_ids]
            from .evaluation import weighted_soft_bce
            value=float(weighted_soft_bce(labels.loc[val_ids,PSEUDO_COLS],pred[TARGETS],labels.loc[val_ids,WEIGHT_COLS]).mean())
        improved=value is not None and value<best
        if improved: best=value; bad=0; best_epoch=epoch+1
        elif value is not None: bad+=1
        history.append({'epoch':epoch+1,'train_bce':total/n,'validation_bce':value,'seconds':time.time()-begin})
        state={'format':'rsna_v2_1','run_id':identity,'config':cfg,'model':model.state_dict(),
               'optimizer':optimizer.state_dict(),'scaler':scaler.state_dict(),'epoch':epoch+1,
               'history':history,'best':best,'best_epoch':best_epoch,'bad':bad,
               'rng_python':random.getstate(),'rng_numpy':np.random.get_state(),'rng_torch':torch.get_rng_state(),
               'rng_cuda':torch.cuda.get_rng_state_all() if device.type=='cuda' else None}
        atomic_save(state,out/'last.pt')
        if improved:
            atomic_save(state,out/'best.pt'); pred.to_csv(out/'validation_predictions.csv')
        pd.DataFrame(history).to_csv(out/'history.csv',index=False)
        print(history[-1],flush=True)
        if val_ids and bad>=cfg['patience']: break
    if cfg['mode']=='refit': atomic_save(state,out/'final.pt')
    elif not (out/'best.pt').exists():
        raise RuntimeError('Resume with no new improvement requires the previous best.pt and predictions attached in output directory')
    meta.update(complete=True,best_epoch=best_epoch,right_censored=bool(val_ids and best_epoch==cfg['epochs']),
                checkpoint_sha256=hash_file(out/('best.pt' if val_ids else 'final.pt')))
    if val_ids: meta['validation_predictions_sha256']=hash_file(out/'validation_predictions.csv')
    write_json(meta_path,meta)
    return out
