"""Offline checks for the N4 download integrity gate; never calls Kaggle."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from run_n4_pipeline import verify_cache,digest

with tempfile.TemporaryDirectory(dir=ROOT/'outputs') as d:
    folder=Path(d)
    np.save(folder/'slices.npy',np.zeros((3,384,384),np.uint8))
    (folder/'index.csv').write_text('study,series\na,b\n')
    source={n:hashlib.sha256((ROOT/'src'/n).read_text().encode()).hexdigest() for n in ('canon.py','m00_data.py')}
    s=dict(shard=0,nshard=8,pilot=False,excluded_studies=0,series_failed=0,
           config=dict(out_px=384,model_input=320,crop_mm=160,slots=6,control='N3a',training_epochs=12),
           source_sha256=source,sha256={n:digest(folder/n) for n in ('slices.npy','index.csv')},slices_bytes=3*384*384)
    (folder/'summary.json').write_text(json.dumps(s))
    assert verify_cache(folder,0)['slices_bytes']==3*384*384
    for field,bad in [('excluded_studies',1),('series_failed',1),('shard',1)]:
        t=s|{field:bad};(folder/'summary.json').write_text(json.dumps(t))
        try:verify_cache(folder,0)
        except AssertionError:pass
        else:raise AssertionError(f'Accepted {field}')
    (folder/'summary.json').write_text(json.dumps(s))
    (folder/'index.csv').write_text('corrupted')
    try:verify_cache(folder,0)
    except AssertionError:pass
    else:raise AssertionError('Accepted corrupted cache')
for k in range(8):
    folder=ROOT/f'notebooks/cache_n4/s{k}'
    source=(folder/'cache.py').read_text()
    compile(source,str(folder/'cache.py'),'exec')
    assert f'SHARD, NSHARD, PILOT = {k}, 8, False' in source
    meta=json.loads((folder/'kernel-metadata.json').read_text())
    assert meta['is_private']=='true' and meta['enable_gpu']=='false'
print('N4 pipeline PASS: clean cache accepted; corruption, excluded studies, decode failures and wrong shard rejected; all eight private CPU scripts compile')
