"""Offline, bounded study-at-a-time inference with training preprocessing parity."""
from pathlib import Path
import json
import tempfile
import numpy as np
import pandas as pd
import torch
from .common import UID,TARGETS,SLOTS,hash_file,resolve_comp_dir,write_json
from .preprocessing import preprocess_one_series
from .training import load_trained_model,predict_studies


def select_test_slots(series):
    required={UID,'SeriesInstanceUID','Anatomical_Plane','Fat_Suppression'}
    if required-set(series): raise ValueError(f'Test series missing {required-set(series)}')
    result=[]
    for _,r in series.iterrows():
        plane=str(r.Anatomical_Plane).strip().upper()
        aliases={'SAGITTAL':'SAG','CORONAL':'COR','AXIAL':'AX','SAG':'SAG','COR':'COR','AX':'AX'}
        if plane not in aliases: continue
        fs=str(r.Fat_Suppression).strip().lower()
        if fs in ['1','1.0','true','yes']: suffix='FS'
        elif fs in ['0','0.0','false','no']: suffix='NONFS'
        else: continue
        result.append({UID:str(r[UID]),'SeriesInstanceUID':str(r.SeriesInstanceUID),
                       'slot':aliases[plane]+'_'+suffix,
                       'n_files':int(r.get('n_files',0)),
                       'fluid_bin':int(str(r.get('Fluid_Sensitive',0)).lower() in ['1','1.0','true','yes'])})
    # Multiple candidates use deterministic UID tie-break; expose this explicitly in audit.
    frame=pd.DataFrame(result,columns=[UID,'SeriesInstanceUID','slot','n_files','fluid_bin'])
    return frame.sort_values([UID,'slot','n_files','fluid_bin','SeriesInstanceUID'],
                             ascending=[True,True,False,False,True]).drop_duplicates([UID,'slot'])


def validate_submission(sample,result):
    if set(sample.columns)!={UID,*TARGETS}: raise ValueError('Unexpected sample_submission target schema')
    if result[UID].duplicated().any() or set(result[UID])!=set(sample[UID]): raise ValueError('Submission UID mismatch')
    result=result.set_index(UID).loc[sample[UID]].reset_index()[sample.columns]
    p=result[TARGETS].to_numpy(float)
    if not np.isfinite(p).all() or (p<0).any() or (p>1).any(): raise ValueError('Nonfinite/out-of-range submission')
    return result


def run_submission(config):
    if config.get('run_submission') is not True: raise ValueError('Submission inference is disabled')
    comp=resolve_comp_dir(config.get('comp_dir')); out=Path(config['out_dir']); out.mkdir(parents=True,exist_ok=True)
    path=Path(config['cache_meta_path'])
    if hash_file(path)!=config['cache_meta_sha256']: raise ValueError('Preprocessing metadata hash mismatch')
    cache_meta=json.loads(path.read_text()); pcfg=cache_meta['config']
    if not cache_meta['complete'] or cache_meta['smoke']: raise ValueError('Use full production preprocessing metadata')
    if pcfg.get('slot_strategy','legacy')!='legacy': raise ValueError('Semantic slot inference is not yet validated; use legacy slots')
    device=torch.device(config.get('device','cuda'))
    if device.type=='cuda' and not torch.cuda.is_available(): raise RuntimeError('Enable GPU')
    members=[]
    for item in config['checkpoints']:
        if hash_file(item['path'])!=item['sha256']: raise ValueError('Checkpoint hash mismatch')
        model,cfg=load_trained_model(item['path'],device)
        if cfg['preprocessing_id']!=cache_meta['preprocessing_id'] or cfg['preprocessing_config']!=pcfg:
            raise ValueError('Checkpoint and preprocessing cache mismatch')
        if cfg['num_workers']!=0: cfg={**cfg,'num_workers':0}
        members.append((model,cfg))
    if not members: raise ValueError('Specify final checkpoint paths and hashes')
    sample=pd.read_csv(comp/'sample_submission.csv',dtype={UID:str})
    series=pd.read_csv(comp/'test_series.csv',dtype={UID:str,'SeriesInstanceUID':str})
    roots=[p for p in [comp/'test_series',comp/'test_images'] if p.is_dir()]
    if len(roots)!=1: raise ValueError('Expected exactly one test_series or test_images directory')
    test_root=roots[0]
    series['n_files']=[sum(p.is_file() for p in (test_root/r[UID]/r.SeriesInstanceUID).iterdir())
                       if (test_root/r[UID]/r.SeriesInstanceUID).is_dir() else 0 for _,r in series.iterrows()]
    selected=select_test_slots(series); predictions=[]; audit=[]
    for uid in sample[UID]:
        with tempfile.TemporaryDirectory(prefix='rsna_v2_test_',dir=out) as temporary:
            root=Path(temporary); rows=[]
            for _,r in selected[selected[UID]==uid].iterrows():
                series_path=test_root/uid/r.SeriesInstanceUID
                if not series_path.is_dir(): raise FileNotFoundError(series_path)
                images,meta=preprocess_one_series(series_path,pcfg)
                audit.append({UID:uid,'SeriesInstanceUID':r.SeriesInstanceUID,'slot':r.slot,
                              'n_valid':meta['n_valid'],'decode_failures':len(meta['decode_failures'])})
                if meta['n_valid']==0: continue
                filename=r.slot+'.npz'
                np.savez_compressed(root/filename,images=images,valid=np.array(meta['valid'],bool),
                                    positions=np.array(meta['positions'],np.float32))
                rows.append({UID:uid,'SeriesInstanceUID':r.SeriesInstanceUID,'slot':r.slot,'path':filename})
            if not rows: raise RuntimeError(f'No valid test MRI inputs for {uid}')
            frame=pd.DataFrame(rows)
            member_predictions=[predict_studies(model,root,frame,[uid],cfg,device)[TARGETS].to_numpy()[0]
                                for model,cfg in members]
            predictions.append({UID:uid,**dict(zip(TARGETS,np.mean(member_predictions,axis=0)))})
    result=validate_submission(sample,pd.DataFrame(predictions))
    result.to_csv(out/'submission.csv',index=False)
    pd.DataFrame(audit).to_csv(out/'test_decode_audit.csv',index=False)
    write_json(out/'submission_provenance.json',{'checkpoints':config['checkpoints'],
        'cache_meta_sha256':config['cache_meta_sha256'],'n_test_studies':len(result),
        'series_selection':'legacy plane/FS; file count, fluid sensitivity, lexical UID tie-break',
        'submission_sha256':hash_file(out/'submission.csv')})
    return out/'submission.csv'
