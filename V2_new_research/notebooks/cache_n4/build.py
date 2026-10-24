"""Build a private CPU pilot for true native-resolution N4; no training or gold evaluation."""
import hashlib
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent
sources={n:(ROOT/'src'/n).read_text(encoding='utf-8') for n in ('m00_data.py','canon.py')}
t=(ROOT/'notebooks/cache_v3/cache_template.py').read_text(encoding='utf-8')
t=t.replace('from canon import OUT_PX, canon_series  # noqa: E402', '''from canon import canon_series as _canon_series
OUT_PX = 384

def canon_series(paths):
    return _canon_series(paths, out_px=OUT_PX)
''')
t=t.replace('studies = sorted({r["StudyInstanceUID"] for r in all_rows})','studies = sorted({r["StudyInstanceUID"] for r in all_rows if part.get(r["StudyInstanceUID"]) in ("candidate_train", "candidate_validation")})')
t=t.replace('set(sorted(studies, key=h)[:20])','set(sorted((u for u in studies if part[u] == "candidate_train"), key=h)[:50])')
t=t.replace('summary["parity"] = parity(rows, starts, counts, str_index)', '''check_ids = [i for i, r in enumerate(index) if r["ok"]][:20]
        summary["parity"] = parity([rows[i] for i in check_ids], [starts[i] for i in check_ids],
                                  [counts[i] for i in check_ids], [str_index[i] for i in check_ids])
        assert summary["parity"]["series_checked"] == 20
        assert summary["parity"]["mismatch"] == 0
        assert summary["parity"]["indep_corr_min"] > .98
        assert summary["excluded_studies"] == 0
        summary["config"] = {"out_px":384,"model_input":320,"crop_mm":160,"slots":6,"control":"N3a", "training_epochs":12}
        summary["source_sha256"] = {n: hashlib.sha256(text.encode()).hexdigest() for n,text in SOURCES.items()}''')
code=t.replace('__SHARD__','0').replace('__NSHARD__','8').replace('__PILOT__','True').replace('__SOURCES__',repr(sources))
d=HERE/'pilot';d.mkdir(parents=True,exist_ok=True)
(d/'cache.py').write_text(code,encoding='utf-8')
meta=json.loads((ROOT/'notebooks/cache_v3/pilot/kernel-metadata.json').read_text())
meta.update(id='aymenslimani/rsna-knee-n4-native-384-train-only-pilot',title='RSNA knee N4 native 384 train-only pilot')
(d/'kernel-metadata.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
print('Built private CPU-only 50-training-study pilot; raw DICOM -> 384; unchanged 160mm FOV')

# Full shards preserve the exact successful pilot preprocessing.
for k in range(8):
    d = HERE / f"s{k}"
    d.mkdir(exist_ok=True)
    full = code.replace("SHARD, NSHARD, PILOT = 0, 8, True", f"SHARD, NSHARD, PILOT = {k}, 8, False", 1)
    full = full.replace('    if PILOT:\n        tp = time.time()', '    summary["config"] = {"out_px":384,"model_input":320,"crop_mm":160,"slots":6,"control":"N3a","training_epochs":12}\n    summary["source_sha256"] = {n: hashlib.sha256(text.encode()).hexdigest() for n,text in SOURCES.items()}\n    if PILOT:\n        tp = time.time()')
    (d / "cache.py").write_text(full, encoding="utf-8")
    fullmeta = dict(meta, id=f"aymenslimani/rsna-knee-n4-native384-s{k}", title=f"RSNA knee N4 native384 s{k}")
    (d / "kernel-metadata.json").write_text(json.dumps(fullmeta, indent=2), encoding="utf-8")
    print("Built", d)
