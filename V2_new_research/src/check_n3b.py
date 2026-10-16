"""CPU regression checks for N3b; execute with project Python."""
import sys
from pathlib import Path
import importlib.util
import numpy as np
import torch
from torch import nn
sys.path.insert(0, str(Path(__file__).resolve().parent))
from v3_model import V3, fill_windows

def main():
    torch.set_num_threads(2)
    torch.manual_seed(0)
    m = V3('context')
    m.encoder = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(3,512))
    m.eval()
    x = torch.randn(2,6,3,3,8,8)
    mask = torch.rand(2,6,3) > .5
    mask[:,0,0] = True
    pos = torch.rand(2,6,3)
    a = m(x,mask,pos)
    xx, pp = x.clone(), pos.clone()
    xx[~mask], pp[~mask] = float('nan'), float('nan')
    assert torch.allclose(a,m(xx,mask,pp),atol=1e-6)
    assert not torch.allclose(a,m(x,mask,1-pos))
    a.sum().backward()
    for name in ['project.0.weight','slot_embed.weight','position.0.weight','context.layers.0.self_attn.in_proj_weight','encoder.2.weight']:
        g = dict(m.named_parameters())[name].grad
        assert g is not None and torch.isfinite(g).all() and g.abs().sum()>0, name
    for bad in [torch.zeros_like(mask),mask]:
        try:
            m(x,bad,None)
        except ValueError:
            pass
        else:
            raise AssertionError('invalid input accepted')
    frozen = Path(__file__).resolve().parents[1]/'outputs/n3a_attn6/source/v3_model.py'
    spec=importlib.util.spec_from_file_location('frozen_n3a',frozen)
    old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
    v=np.arange(12*4*4,dtype=np.uint8).reshape(12,4,4)
    for rngseed in [None,2,99]:
        x1=np.zeros((4,3,4,4),np.uint8);x2=x1.copy()
        m1=np.zeros(4,bool);m2=m1.copy();p=np.zeros(4,np.float32)
        old.fill_windows(x1,m1,v,4,None if rngseed is None else np.random.default_rng(rngseed))
        fill_windows(x2,m2,v,4,None if rngseed is None else np.random.default_rng(rngseed),p)
        assert np.array_equal(x1,x2) and np.array_equal(m1,m2)
        assert ((p>0)&(p<1)).all()
    previous=old.V3('attn');current=V3('attn')
    current.load_state_dict(previous.state_dict(),strict=True)
    previous.eval();current.eval()
    x=torch.randn(1,6,1,3,32,32);mask=torch.ones(1,6,1,dtype=torch.bool)
    with torch.no_grad():
        assert torch.equal(previous(x,mask),current(x,mask))
    print('PASS: masking, positions, gradients, input rejection, frozen N3a sampling and exact legacy logits')

if __name__=='__main__':
    main()
