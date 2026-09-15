import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn
from rsna_v2.common import UID,TARGETS
from rsna_v2 import training
from rsna_v2.submission import validate_submission,select_test_slots


def test_global_macro_loss_has_target_equal_weight():
    logits=torch.tensor([[1.,-2.],[3.,1.]])
    y=torch.tensor([[1.,0.],[0.,1.]]); w=torch.tensor([[1.,.01],[1.,.01]])
    actual=training.macro_loss(logits,y,w,w.mean(0))
    expected=torch.nn.functional.binary_cross_entropy_with_logits(logits,y)
    torch.testing.assert_close(actual,expected)


def test_masked_pooling_ignores_padding(monkeypatch):
    class Encoder(nn.Module):
        def forward(self,x): return x.mean((2,3))
    monkeypatch.setattr(training,'create_encoder',lambda cfg,init:(Encoder(),3))
    cfg={'frozen_encoder':True,'pooling':'meanmax'}
    model=training.StudyModel(cfg).eval()
    images=torch.ones(1,6,2,3,8,8); mask=torch.zeros(1,6,2,dtype=torch.bool); mask[:,0,0]=True
    first=model(images,mask)
    images[:,1:]=999
    torch.testing.assert_close(first,model(images,mask))


def test_cache_read_and_invalid_study(tmp_path):
    np.savez(tmp_path/'x.npz',images=np.ones((3,8,8),np.uint8),valid=np.ones(3,bool),positions=np.arange(3))
    frame=pd.DataFrame([{UID:'a','slot':'SAG_FS','path':'x.npz'}])
    cfg={'train_anchors':2,'eval_anchors':3,'image_size':8,'input_mode':'repeat'}
    ds=training.StudyDataset(tmp_path,frame,['a'],cfg)
    assert ds[0]['mask'].sum()==3
    np.savez(tmp_path/'x.npz',images=np.ones((3,8,8),np.uint8),valid=np.zeros(3,bool),positions=np.arange(3))
    with pytest.raises(ValueError,match='All MRI'): ds[0]


def test_submission_order_and_invalid():
    sample=pd.DataFrame([{UID:'b',**{t:0 for t in TARGETS}},{UID:'a',**{t:0 for t in TARGETS}}])
    result=sample.iloc[::-1].copy()
    assert validate_submission(sample,result)[UID].tolist()==['b','a']
    result.iloc[0,1]=np.nan
    with pytest.raises(ValueError,match='Nonfinite'): validate_submission(sample,result)


def test_slot_assignment():
    frame=pd.DataFrame([{UID:'a','SeriesInstanceUID':'b','Anatomical_Plane':'Sagittal','Fat_Suppression':1}])
    assert select_test_slots(frame).slot.tolist()==['SAG_FS']


def test_training_selection_refit_and_restore(tmp_path,monkeypatch):
    import json
    from rsna_v2.common import PSEUDO_COLS,WEIGHT_COLS,write_json,hash_file
    class Tiny(nn.Module):
        def __init__(self): super().__init__(); self.layer=nn.Linear(3,3)
        def forward(self,x): return self.layer(x.mean((2,3)))
    monkeypatch.setattr(training,'create_encoder',lambda cfg,init:(Tiny(),3))
    cache=tmp_path/'cache'; sup=tmp_path/'supervision'; cache.mkdir(); sup.mkdir()
    rows=[]; labels=[]
    for i in range(10):
        uid=str(i); filename=f'{i}.npz'
        np.savez(cache/filename,images=np.full((2,8,8),20+i,np.uint8),valid=np.ones(2,bool),positions=np.arange(2))
        rows.append({UID:uid,'SeriesInstanceUID':str(i),'slot':'SAG_FS','path':filename,
                     'cv_group_id':str(i),'training_role':'PSEUDO_TRAIN','preprocessing_id':'p'})
        labels.append({UID:uid,'cv_group_id':str(i),'training_role':'PSEUDO_TRAIN','fold':i%5,
                      **{k:.7 for k in PSEUDO_COLS},**{k:1. for k in WEIGHT_COLS}})
    pd.DataFrame(rows).to_csv(cache/'cache_index.csv',index=False)
    pd.DataFrame(labels).to_csv(sup/'train_manifest.csv',index=False)
    write_json(cache/'cache_meta.json',{'complete':True,'smoke':False,'config':{'size':8},
               'preprocessing_id':'p','index_sha256':hash_file(cache/'cache_index.csv')})
    write_json(sup/'supervision_meta.json',{'supervision_id':'s','train_manifest_sha256':hash_file(sup/'train_manifest.csv')})
    cfg={'cache_dir':str(cache),'supervision_dir':str(sup),'out_dir':str(tmp_path/'selection'),
         'epochs':1,'batch_size':2,'num_workers':0,'require_cuda':False,'allow_download':True,
         'train_anchors':2,'eval_anchors':2}
    out=training.run_training(cfg)
    assert (out/'best.pt').is_file() and len(pd.read_csv(out/'validation_predictions.csv'))==2
    model,restored=training.load_trained_model(out/'best.pt','cpu')
    assert restored['preprocessing_id']=='p'
    refit=training.run_training({**cfg,'mode':'refit','out_dir':str(tmp_path/'refit')})
    assert (refit/'final.pt').is_file()
