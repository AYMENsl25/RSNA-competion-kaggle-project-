"""N4 resolution and legacy-default regression checks."""
import importlib.util
import sys
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tests'))
from test_canon import BASE, series, bright_extent
from canon import canon_series
torch.set_num_threads(2)
a=np.full((512,512),50,np.uint16);a[176:336,176:336]=1000
raw=series(a,[1,0,0],[0,1,0],.5)
x,m=canon_series(raw,out_px=384)
assert x.shape==(6,384,384) and x.dtype==np.uint8
h,w=bright_extent(x[0]);assert abs(h-192)<=2 and abs(w-192)<=2,(h,w)
y,n=canon_series(raw)
assert np.array_equal(m['positions'],n['positions'])
for k in ('laterality','plane_geom','flip_rows','flip_cols','transpose'):
    assert m[k]==n[k]
p=ROOT/'outputs/n3a_attn6/source/canon.py'
spec=importlib.util.spec_from_file_location('old_canon',p);old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
z,_=old.canon_series(raw);assert np.array_equal(y,z)
for bad in (0,-1,1.5):
    try: canon_series(raw,out_px=bad)
    except ValueError: pass
    else: raise AssertionError(bad)
compile((ROOT/'notebooks/cache_n4/pilot/cache.py').read_text(), 'n4_pilot', 'exec')
print('N4 PASS: native384 dimensions/physical size; identical geometry; byte-exact old256 default; invalid dimensions rejected; generated pilot compiles')
