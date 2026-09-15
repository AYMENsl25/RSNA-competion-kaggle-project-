import numpy as np
import pydicom
from pydicom.dataset import FileDataset,FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian,generate_uid
from rsna_v2.preprocessing import preprocess_one_series


def test_synthetic_dicom_spatial_order_and_cache(tmp_path):
    study=generate_uid(); series=generate_uid(); frame=generate_uid()
    for i in [2,0,1]:
        meta=FileMetaDataset(); meta.TransferSyntaxUID=ExplicitVRLittleEndian
        meta.MediaStorageSOPClassUID=pydicom.uid.MRImageStorage; meta.MediaStorageSOPInstanceUID=generate_uid()
        ds=FileDataset(str(tmp_path/f'{2-i}.dcm'),{},file_meta=meta,preamble=b'\0'*128)
        ds.SOPInstanceUID=meta.MediaStorageSOPInstanceUID; ds.SOPClassUID=meta.MediaStorageSOPClassUID
        ds.StudyInstanceUID=study; ds.SeriesInstanceUID=series; ds.FrameOfReferenceUID=frame
        ds.ImageOrientationPatient=[1,0,0,0,1,0]; ds.ImagePositionPatient=[0,0,i*2]
        ds.InstanceNumber=i+1; ds.PixelSpacing=[1,1]; ds.Rows=8; ds.Columns=8
        ds.SamplesPerPixel=1; ds.PhotometricInterpretation='MONOCHROME2'; ds.BitsAllocated=16
        ds.BitsStored=16; ds.HighBit=15; ds.PixelRepresentation=0
        ds.PixelData=np.arange(64,dtype=np.uint16).reshape(8,8).tobytes()
        ds.save_as(tmp_path/f'{2-i}.dcm',enforce_file_format=True)
    x,meta=preprocess_one_series(tmp_path,{'n_slices':3,'image_size':8,'crop_mm':None,'slice_band':[0,1]})
    assert x.dtype==np.uint8 and x.shape==(3,8,8)
    assert meta['positions']==[0,2,4] and meta['complete']
    # Exercise full-cache writer and its public column contract, not just a single series.
    import pandas as pd
    import json
    from rsna_v2.preprocessing import run_preprocessing
    comp=tmp_path/'competition'; comp.mkdir(); (comp/'train.csv').write_text('StudyInstanceUID\n')
    destination=comp/'train_series'/study/series; destination.mkdir(parents=True)
    for p in list(tmp_path.glob('*.dcm')): p.replace(destination/p.name)
    plan=tmp_path/'plan.csv'
    pd.DataFrame([{'StudyInstanceUID':study,'SeriesInstanceUID':series,'slot':'SAG_FS',
                   'split':'train','cv_group_id':'g1','training_role':'PSEUDO_TRAIN'}]).to_csv(plan,index=False)
    cfg={'comp_dir':str(comp),'slot_plan_path':str(plan),'out_dir':str(tmp_path/'cache'),
         'n_slices':3,'image_size':8,'crop_mm':None,'slice_band':[0,1],'smoke_studies':0}
    out=run_preprocessing(cfg)
    assert {'StudyInstanceUID','SeriesInstanceUID','path'}.issubset(pd.read_csv(out/'cache_index.csv').columns)
    assert json.loads((out/'cache_meta.json').read_text())['complete']
    assert run_preprocessing(cfg)==out  # verifies resumable identity and cache checksum
