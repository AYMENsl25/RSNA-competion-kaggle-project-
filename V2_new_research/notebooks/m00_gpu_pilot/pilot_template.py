"""Synthetic-only M00 GPU engineering pilot. No real data, labels, checkpoints or submission.
Timings are for random tensors on one Kaggle GPU; they are NOT training/inference time estimates."""
import json, sys, time, traceback
from pathlib import Path

import torch
import torchvision
# __M00_MODEL_SOURCE__

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
