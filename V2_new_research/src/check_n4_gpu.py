"""N4 worst-case GPU feasibility only; synthetic pixels, no AUC evaluation."""
import json
import time
import torch
import torch.nn.functional as F
from run_n4 import ROOT, OUT, config
import train_v3 as T

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(2);torch.manual_seed(0)
    args=config();dev=torch.device('cuda')
    model=T.make_model(dev,args['head'],args['backbone'])
    model.encoder.set_grad_checkpointing(True)
    model.train()
    x=torch.randint(0,256,(2,6,4,3,384,384),dtype=torch.uint8,device=dev)
    mask=torch.ones((2,6,4),dtype=torch.bool,device=dev)
    target=torch.rand(2,12,device=dev)
    opt=torch.optim.AdamW(model.parameters(),lr=args['lr'],weight_decay=args['wd'])
    scaler=torch.amp.GradScaler('cuda')
    torch.cuda.reset_peak_memory_stats();times=[];skips=0
    for _ in range(10):
        torch.cuda.synchronize();t=time.perf_counter();opt.zero_grad(set_to_none=True)
        with torch.autocast('cuda',dtype=torch.float16):
            logits=model(T.prep(x,args['res']),mask)
        loss=F.binary_cross_entropy_with_logits(logits.float(),target)
        assert torch.isfinite(loss)
        scaler.scale(loss).backward();scaler.unscale_(opt)
        finite=all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
        if finite:
            assert model.att_w.weight.grad.abs().sum()>0
            assert any(p.grad is not None and p.grad.abs().sum()>0 for p in model.encoder.parameters())
        scaler.step(opt);scaler.update();torch.cuda.synchronize()
        if finite:times.append(time.perf_counter()-t)
        else:skips+=1
        if len(times)==3:break
    assert len(times)==3
    train_peak=torch.cuda.max_memory_allocated()/2**20
    model.eval();torch.cuda.reset_peak_memory_stats()
    xv=torch.randint(0,256,(1,6,16,3,384,384),dtype=torch.uint8,device=dev)
    mv=torch.ones((1,6,16),dtype=torch.bool,device=dev)
    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.float16):
        p=model(T.prep(xv,320),mv)
    assert p.shape==(1,12) and torch.isfinite(p).all()
    result=dict(passed=True,synthetic_pixels=True,grad_checkpointing=True,training_valid_windows=48,
                validation_valid_windows=96,train_peak_mib=train_peak,validation_peak_mib=torch.cuda.max_memory_allocated()/2**20,
                finite_step_seconds=times,amp_startup_skips=skips,args=args,
                note='GPU feasibility only; native-cache cohort/pixel checks remain required before training. Smoke weights discarded.')
    (OUT/'gpu_feasibility.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result),flush=True)

if __name__=='__main__':main()
