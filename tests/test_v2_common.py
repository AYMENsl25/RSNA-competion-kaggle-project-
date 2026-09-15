import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from rsna_v2.common import (TARGETS, PSEUDO_COLS, WEIGHT_COLS, build_supervision,
                            resolve_unique, hash_file)
from rsna_v2.evaluation import weighted_soft_bce, gold_metrics


def fixture_inputs(tmp_path):
    rows = [{'StudyInstanceUID':str(i),'cv_group_id':'g'+str(i),
             'training_role':'PSEUDO_TRAIN' if i<10 else 'GOLD_EVAL_LOCKED'} for i in range(11)]
    pd.DataFrame(rows).to_csv(tmp_path/'rsna_phase4b2a_six_slot_plan_v1.csv',index=False)
    pseudo=pd.DataFrame({'StudyInstanceUID':[str(i) for i in range(10)]})
    for c in PSEUDO_COLS: pseudo[c]=.7
    for c in WEIGHT_COLS: pseudo[c]=.5
    pseudo.to_csv(tmp_path/'rsna_pseudo_labels_v1_1_wide.csv',index=False)
    (tmp_path/'PSEUDO_LABELS_v1_1.json').write_text(json.dumps({'freeze_name':'PSEUDO_LABELS_v1_1'}))
    fold=pd.DataFrame(rows[:10]).drop(columns='training_role'); fold['fold']=np.arange(10)%5
    fold.to_csv(tmp_path/'rsna_phase4b1b_pseudo_fold_assignments_v1_2.csv',index=False)
    return {'input_roots':[str(tmp_path)],'out_dir':str(tmp_path/'out'),'expected_counts':None}


def test_freeze_no_gold_and_reuse_folds(tmp_path):
    cfg=fixture_inputs(tmp_path); out=build_supervision(cfg)
    frame=pd.read_csv(out/'train_manifest.csv')
    assert len(frame)==10 and set(frame.training_role)=={'PSEUDO_TRAIN'}
    assert not set(TARGETS).intersection(frame.columns)
    assert frame.fold.tolist()==[i%5 for i in range(10)]
    meta=json.loads((out/'supervision_meta.json').read_text())
    assert hash_file(out/'train_manifest.csv')==meta['train_manifest_sha256']


def test_reject_gold_group_leakage(tmp_path):
    cfg=fixture_inputs(tmp_path)
    path=tmp_path/'rsna_phase4b2a_six_slot_plan_v1.csv'
    f=pd.read_csv(path); f.loc[10,'cv_group_id']='g0'; f.to_csv(path,index=False)
    with pytest.raises(ValueError,match='Gold/group'): build_supervision(cfg)


def test_reject_inconsistent_roles(tmp_path):
    cfg=fixture_inputs(tmp_path); path=tmp_path/'rsna_phase4b2a_six_slot_plan_v1.csv'
    f=pd.read_csv(path); duplicate=f.iloc[[0]].copy(); duplicate['cv_group_id']='changed'
    pd.concat([f,duplicate]).to_csv(path,index=False)
    with pytest.raises(ValueError,match='inconsistent'): build_supervision(cfg)


def test_resolver_rejects_conflicting_copies(tmp_path):
    (tmp_path/'a').mkdir(); (tmp_path/'b').mkdir()
    (tmp_path/'a'/'x.json').write_text('a'); (tmp_path/'b'/'x.json').write_text('b')
    with pytest.raises(ValueError,match='Conflicting'): resolve_unique('x.json',[tmp_path])
    assert resolve_unique('x.json',[tmp_path],tmp_path/'a'/'x.json').parent.name=='a'


def test_soft_loss_target_normalization():
    y=np.array([[1,0],[0,1]]); p=np.array([[.8,.1],[.2,.9]])
    w=np.array([[1,.01],[1,.01]])
    expected=np.array([-np.log(.8),-np.log(.9)])
    np.testing.assert_allclose(weighted_soft_bce(y,p,w),expected)


def test_gold_undefined_target_not_ignored():
    y=np.zeros((4,12)); p=np.full_like(y,.5)
    _,report=gold_metrics(y,p,bootstrap=2)
    assert report['macro_auc'] is None


def test_reject_missing_fold(tmp_path):
    cfg=fixture_inputs(tmp_path)
    path=tmp_path/'rsna_phase4b1b_pseudo_fold_assignments_v1_2.csv'
    f=pd.read_csv(path); f.loc[0,'cv_group_id']='wrong'; f.to_csv(path,index=False)
    with pytest.raises(ValueError,match='Fold missing'): build_supervision(cfg)
