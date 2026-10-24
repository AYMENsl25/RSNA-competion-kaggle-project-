import copy,sys
import torch
from torch import nn
sys.path.insert(0,'src')
from train_v3 import make_optimizer
from watch_n4_epoch_gate import decide
class Small(nn.Module):
    def __init__(self):
        super().__init__();self.encoder=nn.Linear(2,2);self.head=nn.Linear(2,1)
    def forward(self,x):return self.head(self.encoder(x))
a=Small();b=copy.deepcopy(a)
o=make_optimizer(a,1e-4,.05);ref=torch.optim.AdamW(b.parameters(),lr=1e-4,weight_decay=.05)
x=torch.ones(2,2)
a(x).sum().backward();b(x).sum().backward();o.step();ref.step()
assert all(torch.equal(p,q) for p,q in zip(a.parameters(),b.parameters()))
o=make_optimizer(a,1e-4,.05,3e-5,1e-3)
ids=[id(p) for g in o.param_groups for p in g['params']]
assert len(ids)==len(set(ids))==len(list(a.parameters()))
s=torch.optim.lr_scheduler.CosineAnnealingLR(o,10)
for _ in range(5):
    assert abs(o.param_groups[1]['lr']/o.param_groups[0]['lr']-100/3)<1e-8
    o.step();s.step()
def epoch(i,p):return dict(epoch=i,val={k:dict(macro=p) for k in ('public','v3')})
control=[epoch(3,.82),epoch(4,.85)]
assert decide([epoch(2,.5)],control)[0]=='wait'
assert decide([epoch(3,.56)],control)[0]=='stop'
assert decide([epoch(3,.81)],control)[0]=='wait'
assert decide([epoch(3,.81),epoch(4,.84)],control)[0]=='continue'
assert decide([epoch(3,.81),epoch(4,.78)],control)[0]=='stop'
print('PASS: uniform optimizer exact parity; differential groups complete/disjoint; cosine ratio; gate waits for epoch3 and handles epoch4')
