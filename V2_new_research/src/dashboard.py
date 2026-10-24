"""Local experiment dashboard: live progress + results of every train_v3 run.

Usage: python src/dashboard.py   then open http://localhost:8050  (auto-refreshes every 15 s)
Reads outputs/*/log.json (finished epochs), outputs/<run>.log (live step), and Kaggle submissions.
"""
import csv, glob, html, io, json, os, re, subprocess, time
from http.server import BaseHTTPRequestHandler, HTTPServer

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
NAMES = {"n5_uniform16": "N5 · uniform LR 16 epochs", "n5_differential16": "N5 · encoder/head LR 16 epochs", "n4_native384": "N4 · native384 / input320 attention", "e1_v3": "E1 · new preprocessing", "e45_v3": "E4+E5 · relabel + soft targets",
         "e2_v3": "E2 · slab 12 + attention (rejected)", "e6_v3": "E6 · E4+E5 + augmentation + 16 ep + avg last 3",
         "e7_320": "E7 · E4+E5 at 320 px", "e8_cnxt": "E8 · ConvNeXt-Tiny", "e3_cnxt6": "E3 · ConvNeXt-Tiny + 6 slots",
         "n1_attn_random": "N1 · ConvNeXt-Tiny + random-window attention (E8 control)",
         "n3b_context": "N3b · sequence and position contextual MIL",
         "n3a_attn6": "N3a · six-slot random-window attention (N1 control)"}
REFS = [("E01 (old baseline)", 0.8215)]
_lb = {"t": 0, "rows": []}
_seen = {}  # log path -> (size, time size last changed)


def leaderboard():
    if time.time() - _lb["t"] > 300:  # ponytail: polls Kaggle at most every 5 min
        try:
            out = subprocess.run(["kaggle", "competitions", "submissions", "rsna-knee-abnormality-detection", "--csv"],
                                 capture_output=True, text=True, timeout=60).stdout
            _lb["rows"] = list(csv.DictReader(io.StringIO(out[out.find("ref"):])))
        except Exception:
            pass
        _lb["t"] = time.time()
    return _lb["rows"]


KERNELS = [  # Kaggle runs since 2026-10-17 (public-weights track)
    ("aymenslimani/rsna-knee-f0-fork-0958", "F0 · public 0.958 stack, unchanged (reproduction)", None),
    ("aymenslimani/rsna-knee-k1-raptor-ft", "K1-A · Raptor-384 fine-tune: consensus labels + parent distillation + aug", "k1_raptor_ft/armA"),
    ("aymenslimani/rsna-knee-k1-raptor-ft", "K1-B · Raptor-384 fine-tune: same targets + dual view (global + 1.5x zoom)", "k1_raptor_ft/armB"),
    ("aymenslimani/rsna-knee-k2-fresh-effv2", "K2-A · fresh EfficientNetV2-S (ImageNet), 1.5x joint-zoom view, labels only", "k2_fresh/armA"),
    ("aymenslimani/rsna-knee-k2-fresh-effv2", "K2-B · fresh EfficientNetV2-S (ImageNet), global view, labels only", "k2_fresh/armB"),
    ("aymenslimani/rsna-knee-k3-compart-seresnext", "K3-A · fresh SE-ResNeXt-50, medial/lateral compartment tiles, EMA, 256→320", "k3_compart/armA"),
    ("aymenslimani/rsna-knee-k3-compart-seresnext", "K3-B · fresh EfficientNetV2-S, compartment tiles, EMA, 256→320", "k3_compart/armB"),
    ("aymenslimani/rsna-knee-k4-warm-compart", "K4-A · Raptor encoder warm start, compartment tiles, fresh head, labels only, enc LR 4e-5", "k4_warm_compart/armA"),
    ("aymenslimani/rsna-knee-k4-warm-compart", "K4-B · same, enc LR 2e-4 (leave the parent basin)", "k4_warm_compart/armB"),
    ("aymenslimani/rsna-knee-t1-teacher-convnexts-2", "T1 · 4-fold ConvNeXt-S teacher (folds 0,2 on GPU0)", "t1_teacher/armA/fold0"),
    ("aymenslimani/rsna-knee-t1-teacher-convnexts-2", "T1 · 4-fold ConvNeXt-S teacher (folds 1,3 on GPU1)", "t1_teacher/armB/fold1"),
    ("aymenslimani/rsna-knee-s1-students-v3", "S1-A · Raptor encoder + compartment view + labels v3", "s1_students/armA"),
    ("aymenslimani/rsna-knee-s1-students-v3", "S1-B · ConvNeXt-S ImageNet + labels v3 (teacher recipe, full data)", "s1_students/armB"),
]
_ks = {}


def kernel_status(ref):
    c = _ks.get(ref)
    if c is None or time.time() - c[0] > 300:  # ponytail: polls Kaggle at most every 5 min per kernel
        try:
            out = subprocess.run(["kaggle", "kernels", "status", ref], capture_output=True, text=True, timeout=60).stdout
            m = re.search(r'status "KernelWorkerStatus\.(\w+)"', out)
            _ks[ref] = (time.time(), m.group(1).lower() if m else "unknown")
        except Exception:
            _ks[ref] = (time.time(), "unknown")
    return _ks[ref][1]


def kaggle_panel():
    rows = ""
    for ref, desc, out in KERNELS:
        note = ""
        log = os.path.join(ROOT, "outputs", out, "log.jsonl") if out else None
        if log and os.path.exists(log):  # filled after `kaggle kernels output` downloads the run
            ev = [json.loads(x) for x in open(log, encoding="utf-8") if x.strip()]
            gold = [e for e in ev if e.get("ev") == "gold"]
            plan = next((e for e in ev if e.get("ev") == "plan"), None)
            note = " · ".join([f"gold58 {e['tag']} {e['macro']:.4f}" for e in gold[-3:]] +
                              ([f"{plan['epochs']} epochs × {plan['epoch_min']} min"] if plan else []))
        rows += (f"<tr><td>{html.escape(desc)}</td><td>{html.escape(kernel_status(ref))}</td>"
                 f"<td>{html.escape(note) or '—'}</td></tr>")
    return ("<section><h2>Kaggle runs · public-weights track</h2><div class='cfg'>Reference LB: public 0.958 stack "
            "(nartaa CoAtNet + heliosli R6 + goodpjw reader) · top 0.969. Gold-58 numbers are optimistic for the "
            "Raptor parent (its SWA epochs were picked on gold-58); the LB decides.</div>"
            "<table><tr><th>Run</th><th>Kaggle status</th><th>Latest</th></tr>" + rows + "</table></section>")


def runs():
    res = []
    for p in sorted(glob.glob(os.path.join(ROOT, "outputs", "*", "log.json")), key=os.path.getmtime):
        try:
            d = json.load(open(p))
        except Exception:
            continue
        if "epochs" not in d or "args" not in d:
            continue
        name = os.path.basename(os.path.dirname(p))
        live = None
        logf = os.path.join(ROOT, "outputs", name + ".log")
        if os.path.exists(logf):
            for line in reversed(open(logf, encoding="utf-8", errors="ignore").read().splitlines()[-50:]):
                m = re.search(r"ep (\d+) step (\d+)/(\d+).*?'temp_c': (\d+)", line)
                if m:
                    live = [int(x) for x in m.groups()]
                    break
            # Windows doesn't refresh mtime of a file held open by the trainer: track real size growth instead.
            with open(logf, "rb") as fh:
                size = fh.seek(0, 2)
            prev = _seen.get(logf)
            if prev is None:
                _seen[logf] = (size, os.path.getmtime(logf))
            elif prev[0] != size:
                _seen[logf] = (size, time.time())
            running = time.time() - _seen[logf][1] < 2400  # step lines arrive every ~3 min
        else:
            running = False
        res.append({"name": name, "d": d, "live": live, "running": running})
    return res


def bar(frac, color="#2a78d6"):
    return (f'<div class="bar"><div style="width:{100*frac:.1f}%;background:{color}"></div></div>'
            f'<span class="pct">{100*frac:.1f}%</span>')


def pipeline_panel():
    folder = os.path.join(ROOT, "outputs", "n4_pipeline")
    try:
        with open(os.path.join(folder, "run_status.json")) as f:
            status = json.load(f)
        labels = {"building_and_downloading": "Preparing full-resolution cache", "training": "Training",
                  "completed": "Completed", "early_stopped": "Stopped after early validation", "failed": "Needs attention"}
        detail = []
        for k in range(8):
            path = os.path.join(folder, f"s{k}.json")
            if os.path.exists(path):
                with open(path) as f:
                    state = json.load(f).get("state", "pending")
            else:
                state = "pending"
            detail.append(f"Shard {k+1}: {state.replace('_', ' ')}")
        title = html.escape(labels.get(status.get("state"), status.get("state", "Preparing")))
        note = "Local training starts automatically after all eight caches pass verification."
        if status.get("state") == "training":
            note = "All eight caches verified. N4 training is running."
        elif status.get("state") == "early_stopped":
            note = "N4 stopped after the requested epoch checkpoint. See N5 experiment progress below."
        if status.get("state") == "failed":
            note = str(status.get("error", "Inspect the pipeline log."))
        return f"<section><h2>N4 · {title}</h2><div>{html.escape(' · '.join(detail))}</div><p>{html.escape(note)}</p></section>"
    except (OSError, ValueError):
        return ""


def page():
    rs = runs()
    out = []
    for r in rs:
        d, eps = r["d"], r["d"]["epochs"]
        total, done = d["args"]["epochs"], len(eps)
        frac = done / total
        state = "✅ finished" if done >= total else ("🟢 running" if r["running"] else "⏸ paused")
        livehtml = ""
        if r["live"] and done < total and r["live"][0] > done:
            ep, s, n, temp = r["live"]
            frac = (done + s / n) / total
            livehtml = f"<div>Epoch {ep}/{total} {bar(s/n, '#eb6834')} step {s}/{n} · GPU {temp}°C</div>"
        eta = ""
        if eps and done < total:
            per = sum(e["train_seconds"] + e["val_seconds"] for e in eps) / done
            eta = f" · ETA ~{per*(total - total*frac)/60:.0f} min"
        best = max(eps, key=lambda e: e["val"]["public"]["macro"]) if eps else None
        a = d["args"]
        cfg = (f"backbone={a.get('backbone','resnet18')} labels={a['labels']} target={a['target']} "
               f"res={a['res']} head={a.get('head','max')} slots={a.get('slots',3)} "
               f"slab={a.get('slab') or '-'} train_windows={a['train_windows']} "
               f"eval_windows={a.get('eval_windows',16)} mirror={a['mirror']}")
        out.append(f"""<section><h2>{html.escape(NAMES.get(r['name'], r['name']))} <small>{state}</small></h2>
<div class="cfg">{html.escape(cfg)}</div>
<div>Overall {bar(frac)} {done}/{total} epochs{eta}</div>{livehtml}
{f"<div class='best'>Best macro AUC: <b>{best['val']['public']['macro']:.4f}</b> (epoch {best['epoch']}) · v3 labels {best['val']['v3']['macro']:.4f}</div>" if best else ''}
<canvas id="c_{r['name']}" height="90"></canvas></section>""")

    finals = [r for r in rs if r["d"]["epochs"]]
    targets = list(finals[0]["d"]["epochs"][-1]["val"]["public"]["auc"]) if finals else []
    rows = ""
    for t in targets + ["macro"]:
        vals = [(r["d"]["epochs"][-1]["val"]["public"]["auc"].get(t) if t != "macro"
                 else r["d"]["epochs"][-1]["val"]["public"]["macro"]) for r in finals]
        top = max(v for v in vals if v is not None)
        rows += f"<tr><td>{t}</td>" + "".join(
            f"<td class='{'hi' if v == top else ''}'>{v:.3f}</td>" for v in vals) + "</tr>"
    table = ("<table><tr><th>Finding (latest epoch, public labels)</th>" +
             "".join(f"<th>{html.escape(NAMES.get(r['name'], r['name']))}</th>" for r in finals) + "</tr>" + rows + "</table>")

    lb = "".join(f"<tr><td>{html.escape(x.get('description',''))}</td><td>{x.get('date','')[:16]}</td>"
                 f"<td><b>{x.get('publicScore') or '⏳ pending'}</b></td></tr>" for x in leaderboard())
    series = {r["name"]: [[e["epoch"], round(e["val"]["public"]["macro"], 4)] for e in r["d"]["epochs"]] for r in rs}
    return f"""<!doctype html><html><head><meta charset="utf-8"><meta http-equiv="refresh" content="15">
<title>RSNA Knee Experiments</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.min.js"></script>
<style>
:root{{--bg:#fcfcfb;--fg:#0b0b0b;--mut:#52514e;--card:#fff;--line:#e1e0d9}}
@media (prefers-color-scheme:dark){{:root{{--bg:#151515;--fg:#eee;--mut:#aaa;--card:#1f1f1f;--line:#333}}}}
body{{font-family:system-ui,sans-serif;background:var(--bg);color:var(--fg);max-width:1100px;margin:auto;padding:16px}}
section{{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:12px 16px;margin:12px 0}}
h2{{margin:4px 0;font-size:18px}} small{{color:var(--mut);font-weight:normal}} .cfg{{color:var(--mut);font-size:12px;margin-bottom:6px}}
.bar{{display:inline-block;width:320px;max-width:60vw;height:12px;background:var(--line);border-radius:6px;vertical-align:middle;overflow:hidden}}
.bar div{{height:100%}} .pct{{margin:0 8px;font-variant-numeric:tabular-nums}} .best{{margin-top:6px}}
table{{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums;font-size:14px}}
td,th{{border-bottom:1px solid var(--line);padding:4px 8px;text-align:right}} td:first-child,th:first-child{{text-align:left}}
.hi{{font-weight:bold;color:#2a9d50}}
</style></head><body>
<h1>RSNA Knee · experiments</h1><div class="cfg">Auto-refresh 15 s · reference {', '.join(f'{n} {v}' for n, v in REFS)}</div>
{kaggle_panel()}
{pipeline_panel()}
<section><h2>Kaggle leaderboard submissions</h2><table><tr><th>Submission</th><th>Date</th><th>Public LB</th></tr>{lb}</table></section>
{''.join(out)}
<section><h2>Per-finding AUC comparison</h2>{table}</section>
<script>
const S={json.dumps(series)};
for (const [k,v] of Object.entries(S)) {{ const el=document.getElementById('c_'+k); if(!el||!v.length) continue;
 new Chart(el,{{type:'line',data:{{labels:v.map(p=>p[0]),datasets:[{{label:'val macro AUC',data:v.map(p=>p[1]),borderColor:'#2a78d6',tension:.2}},
 {{label:'E01 0.8215',data:v.map(()=>0.8215),borderColor:'#898781',borderDash:[4,4],pointRadius:0}}]}},
 options:{{animation:false,plugins:{{legend:{{display:true}}}},scales:{{x:{{title:{{display:true,text:'epoch'}}}}}}}}}}); }}
</script></body></html>"""


class H(BaseHTTPRequestHandler):
    def do_GET(self):
        body = page().encode()
        self.send_response(200); self.send_header("Content-Type", "text/html; charset=utf-8"); self.end_headers()
        self.wfile.write(body)

    def log_message(self, *a):
        pass


if __name__ == "__main__":
    print("Dashboard: http://localhost:8050")
    HTTPServer(("127.0.0.1", 8050), H).serve_forever()
