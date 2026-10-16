"""Live training progress bar. Usage: python src/progress.py [outputs/e1_v3.log] [epochs]  (Ctrl+C to quit)"""
import os, re, sys, time, json

log = sys.argv[1] if len(sys.argv) > 1 else "outputs/e1_v3.log"
total_ep = int(sys.argv[2]) if len(sys.argv) > 2 else 12

def bar(frac, width=40):
    n = int(frac * width)
    return "█" * n + "░" * (width - n)

while True:
    try:
        lines = open(log, encoding="utf-8", errors="ignore").read().splitlines()
    except FileNotFoundError:
        lines = []
    steps = [re.search(r"ep (\d+) step (\d+)/(\d+).*?'temp_c': (\d+)", l) for l in lines]
    steps = [m for m in steps if m]
    epochs = [json.loads(l) for l in lines if l.startswith('{"epoch"')]
    os.system("cls" if os.name == "nt" else "clear")
    print(f"Training log: {log}\n")
    done = len(epochs)
    ep_frac = 0.0
    if steps:
        ep, s, n, temp = (int(x) for x in steps[-1].groups())
        if ep > done:
            ep_frac = s / n
            print(f"Epoch {ep}/{total_ep}  {bar(ep_frac)} {100*ep_frac:5.1f}%  step {s}/{n}  GPU {temp}°C")
    overall = (done + ep_frac) / total_ep
    print(f"Overall      {bar(overall)} {100*overall:5.1f}%  ({done}/{total_ep} epochs done)")
    if epochs:
        per_ep = sum(e["train_seconds"] + e["val_seconds"] for e in epochs) / done
        print(f"ETA          ~{per_ep * (total_ep - done - ep_frac) / 60:.0f} min\n")
        print(" epoch | macro AUC (public) | macro AUC (v3) | loss")
        for e in epochs:
            print(f"  {e['epoch']:>4} | {e['public_macro']:18.4f} | {e['v3_macro']:14.4f} | {e['train_loss']:.4f}")
        best = max(epochs, key=lambda e: e["public_macro"])
        print(f"\nBest so far: epoch {best['epoch']}  macro {best['public_macro']:.4f}   (E01 reference: 0.8215)")
    if done >= total_ep:
        print("\nTraining finished."); break
    time.sleep(10)
