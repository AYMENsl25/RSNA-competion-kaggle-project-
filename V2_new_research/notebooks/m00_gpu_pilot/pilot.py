"""Synthetic-only M00 GPU engineering pilot. No real data, labels, checkpoints or submission.
Timings are for random tensors on one Kaggle GPU; they are NOT training/inference time estimates."""
import json, sys, time, traceback
from pathlib import Path

import torch
import torchvision
# Embedded M00 source SHA256: e8410ff5ebcfcf2036c5da18538fc194fe6db24eb88f274981bfba9de3ed531e
"""M00: random-init ResNet18 shared encoder, shared max over valid windows (all roles jointly),
one linear head -> 12 logits. No pretrained weights, no network access."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18

TARGETS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA",
           "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]


class M00(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = resnet18(weights=None)
        self.encoder.fc = nn.Identity()
        self.head = nn.Linear(512, len(TARGETS))

    def forward(self, x, mask):
        """x [B,R,W,3,H,W2] float, mask [B,R,W] bool -> logits [B,12]. Only valid windows are
        encoded, so padded windows cannot influence features, BatchNorm stats or the max."""
        if x.ndim != 6 or x.shape[3] != 3 or mask.dtype != torch.bool or mask.shape != x.shape[:3]:
            raise ValueError(f"bad shapes x={tuple(x.shape)} mask={tuple(mask.shape)} {mask.dtype}")
        has = mask.flatten(1).any(1)
        if not has.all():
            raise ValueError(f"studies with no valid window (batch idx {(~has).nonzero().flatten().tolist()})")
        sel = x[mask]
        if not torch.isfinite(sel).all():
            raise ValueError("nonfinite pixels in a valid window")
        feats = self.encoder(sel)                                   # [n,512]
        b = mask.flatten(1).nonzero()[:, 0]                         # study index per valid window
        pooled = feats.new_full((x.shape[0], feats.shape[1]), float("-inf"))
        pooled = pooled.scatter_reduce(0, b[:, None].expand_as(feats), feats, "amax", include_self=True)
        return self.head(pooled)


@torch.no_grad()
def predict_proba(model, x, mask):
    """Finite [B,12] probabilities in [0,1] or a clear error."""
    model.eval()
    p = torch.sigmoid(model(x, mask))
    if not torch.isfinite(p).all():
        raise FloatingPointError("nonfinite probabilities")
    return p


def masked_bce(logits, targets, observed):
    """Mean BCE over observed cells only. observed bool [B,12]; unobserved targets ignored."""
    if logits.shape != targets.shape or logits.shape != observed.shape or observed.dtype != torch.bool:
        raise ValueError("logits, targets and boolean observed mask must have identical shapes")
    if not observed.any():
        raise ValueError("no observed targets in batch")
    selected_logits, selected_targets = logits[observed], targets[observed].float()
    if not torch.isfinite(selected_logits).all() or not torch.isfinite(selected_targets).all():
        raise ValueError("nonfinite observed logits or targets")
    return F.binary_cross_entropy_with_logits(selected_logits, selected_targets)


OUT = Path("/kaggle/working") if Path("/kaggle/working").is_dir() else Path(".")
SHAPE = (1, 3, 4, 3, 224, 224)   # [B, roles, windows, C, H, W]
WARMUP, MAX_STEPS, MODE_BUDGET_S, TOTAL_BUDGET_S = 5, 200, 120.0, 480.0
SEED = 0

summary = {"engineering_only": True, "synthetic_only": True, "x_shape": list(SHAPE),
           "torch": torch.__version__, "torchvision": torchvision.__version__,
           "modes": {}, "failures": [], "completed": False}


def save():
    (OUT / "pilot_summary.json").write_text(json.dumps(summary, indent=2))


def run_mode(name, amp, dev):
    torch.manual_seed(SEED)
    model = M00().to(dev).train()
    summary["param_count"] = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    g = torch.Generator().manual_seed(SEED)
    x = torch.randn(SHAPE, generator=g).to(dev)
    mask = torch.ones(SHAPE[:3], dtype=torch.bool)
    mask[0, 2] = False        # role 2 entirely missing
    mask[0, 1, 3] = False     # role 1 has one padded window
    mask = mask.to(dev)
    targets = torch.randint(0, 2, (SHAPE[0], 12), generator=g).to(dev)
    observed = torch.ones(SHAPE[0], 12, dtype=torch.bool)
    observed[0, ::4] = False  # some unobserved cells
    observed = observed.to(dev)

    def step():
        opt.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
            logits = model(x, mask)
        loss = masked_bce(logits.float(), targets, observed)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        gfinite = all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
        scaler.step(opt)   # GradScaler skips the step itself on inf/nan grads
        scaler.update()
        return logits, loss, gfinite

    for _ in range(WARMUP):
        step()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    times, bad = [], {"logits": 0, "loss": 0, "grad": 0}
    t_mode = time.perf_counter()
    while len(times) < MAX_STEPS and time.perf_counter() - t_mode < MODE_BUDGET_S:
        torch.cuda.synchronize(); t0 = time.perf_counter()
        logits, loss, gfinite = step()
        torch.cuda.synchronize(); times.append(time.perf_counter() - t0)
        bad["logits"] += int(not torch.isfinite(logits).all())
        bad["loss"] += int(not torch.isfinite(loss))
        bad["grad"] += int(not gfinite)
    s = sorted(times)
    summary["modes"][name] = {
        "amp_fp16": amp, "warmup_steps": WARMUP, "measured_iterations": len(times),
        "valid_windows": int(mask.sum()), "total_windows": mask.numel(),
        "sec_per_step_mean": sum(times) / len(times), "sec_per_step_median": s[len(s) // 2],
        "sec_per_step_max": s[-1], "sec_per_study_synthetic": sum(times) / len(times) / SHAPE[0],
        "nonfinite_counts": bad, "all_finite": sum(bad.values()) == 0,
        "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
        "peak_reserved_mib": torch.cuda.max_memory_reserved() / 2**20,
        "last_loss": float(loss)}
    del model, opt, x
    torch.cuda.empty_cache()


def main():
    t_start = time.perf_counter()
    if not torch.cuda.is_available():
        summary["failures"].append("CUDA not available: enable GPU in kernel settings")
        save(); raise SystemExit("CUDA not available")
    props = torch.cuda.get_device_properties(0)
    summary["device"] = {"name": props.name, "total_vram_mib": props.total_memory / 2**20,
                         "capability": list(torch.cuda.get_device_capability(0)),
                         "count": torch.cuda.device_count(), "cuda": torch.version.cuda}
    print(summary["device"], summary["torch"], summary["torchvision"], flush=True)
    for name, amp in (("fp32", False), ("amp_fp16", True)):
        if time.perf_counter() - t_start > TOTAL_BUDGET_S:
            summary["failures"].append(f"skipped {name}: total budget")
            continue
        try:
            run_mode(name, amp, torch.device("cuda"))
            print(name, summary["modes"][name], flush=True)
        except Exception:
            summary["failures"].append(f"{name}: {traceback.format_exc()}")
        save()
    summary["script_seconds"] = time.perf_counter() - t_start
    summary["completed"] = not summary["failures"] and all(m["all_finite"] for m in summary["modes"].values())
    save()
    if not summary["completed"]:
        raise SystemExit("pilot finished with failures or nonfinite values; see pilot_summary.json")


if __name__ == "__main__":
    main()
