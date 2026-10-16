"""Local experiment dashboard: live progress + results of every train_v3 run.

Usage: python src/dashboard.py   then open http://localhost:8050  (auto-refreshes every 15 s)
Reads outputs/*/log.json (finished epochs), outputs/<run>.log (live step), and Kaggle submissions.
"""
import csv, glob, html, io, json, os, re, subprocess, time
from http.server import BaseHTTPRequestHandler, HTTPServer

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
NAMES = {"e1_v3": "E1 · new preprocessing", "e45_v3": "E4+E5 · relabel + soft targets",
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
