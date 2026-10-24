"""Build F1 = the unchanged public 0.958 notebook (F0) + Part C (one S1 student rank-blended at a fixed weight).

python build.py <A|B> [weight]   -> kernel/f1.ipynb + kernel/kernel-metadata.json
Part C never touches Parts A/B: it backs up submission.csv, runs infer_c.py in its own process with a deadline,
and restores the F0 submission on any failure, NaN, constant column or incomplete coverage.
"""
import base64, json, sys
from pathlib import Path

HERE = Path(__file__).parent
ROOT = HERE.parents[1]
arm = sys.argv[1]
weight = float(sys.argv[2]) if len(sys.argv) > 2 else 0.2
assert arm in ("A", "B") and 0 < weight < 0.5
f0 = ROOT / "notebooks/f0_fork_0958"
core = Path(r"C:\Users\msi\AppData\Local\Temp\claude\c--Users-msi-Downloads-kaggle-project-RSNA"
            r"\c4da0b6f-92c0-4eb5-a356-03a489448c2a\scratchpad\dl\lq\zoo_raptor_core.py")
b64 = lambda p: base64.b64encode(p.read_bytes()).decode()

cell = f'''import time as _c_time
if _c_time.time() - globals().get('NB_T0', _c_time.time()) > 7.6 * 3600:
    print('Part C skipped: time budget used, the F0 submission is kept', flush=True)
else:
    # ---- Part C: S1 student (arm {arm}, labels v3) rank-blended into F0 at a fixed weight ----
    import base64 as _c_b64, glob as _c_glob, os as _c_os, shutil as _c_sh, subprocess as _c_sp, sys as _c_sys
    import pandas as _c_pd
    C_W = {weight}
    _c_dir = "/kaggle/working/own_c"
    _c_os.makedirs(_c_dir, exist_ok=True)
    open(_c_dir + "/zoo_raptor_core.py", "wb").write(_c_b64.b64decode("{b64(core)}"))
    open(_c_dir + "/infer_c.py", "wb").write(_c_b64.b64decode("{b64(HERE / 'infer_c.py')}"))
    _c_pub, _c_bak = "/kaggle/working/submission.csv", "/kaggle/working/_f0_submission.csv"
    _c_sh.copy(_c_pub, _c_bak)
    try:
        _c_ck = sorted(p for pat in ["/kaggle/input/*/arm{arm}/t1f-1_last.pt", "/kaggle/input/*/*/arm{arm}/t1f-1_last.pt",
                                     "/kaggle/input/*/*/*/arm{arm}/t1f-1_last.pt"] for p in _c_glob.glob(pat))
        assert _c_ck, "student checkpoint not attached"
        _c_data = next(d for d in ["/kaggle/input/competitions/rsna-knee-abnormality-detection",
                                   "/kaggle/input/rsna-knee-abnormality-detection"] if _c_os.path.exists(d + "/test.csv"))
        _c_deadline = globals().get('NB_T0', _c_time.time()) + 8.6 * 3600
        _c_t0 = _c_time.time()
        _c_sp.run([_c_sys.executable, _c_dir + "/infer_c.py", _c_dir, _c_ck[0], _c_data, "/kaggle/working/_student.csv",
                   str(_c_deadline)], check=True, timeout=max(60, _c_deadline - _c_time.time() + 300))
        _c_f0 = _c_pd.read_csv(_c_bak, dtype={{"StudyInstanceUID": str}})
        _c_st = _c_pd.read_csv("/kaggle/working/_student.csv", dtype={{"StudyInstanceUID": str}})
        _c_st = _c_st.set_index("StudyInstanceUID").loc[_c_f0.StudyInstanceUID].reset_index()
        _c_lab = [c for c in _c_f0.columns if c != "StudyInstanceUID"]
        assert _c_st[_c_lab].notna().all().all(), "student did not cover every study"
        assert (_c_st[_c_lab].nunique() > 1).all(), "student predictions are constant"
        _c_r = lambda d: d[_c_lab].rank(method="average", pct=True)
        _c_out = _c_f0.copy()
        _c_out[_c_lab] = ((1 - C_W) * _c_r(_c_f0) + C_W * _c_r(_c_st)).rank(method="average", pct=True)
        _c_out.to_csv(_c_pub, index=False)
        print(f"Part C: student arm {arm} blended at {{C_W}} in {{_c_time.time() - _c_t0:.0f}}s; "
              f"mean rank corr with F0 {{_c_r(_c_f0).corrwith(_c_r(_c_st)).mean():.3f}}", flush=True)
    except Exception as _c_e:
        _c_sh.copy(_c_bak, _c_pub)
        print("Part C FAILED, F0 submission kept:", repr(_c_e), flush=True)
'''

nb = json.loads((f0 / "fork.ipynb").read_text(encoding="utf-8"))
nb["cells"].append({"cell_type": "markdown", "metadata": {}, "source": [
    "---\n# Part C · our S1 student (labels v3, teacher out-of-fold denoising)\n\n",
    f"Rank-blended into the public 0.958 submission at a fixed weight of {weight}. On any failure the F0 submission is kept."]})
nb["cells"].append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": cell})
out = HERE / "kernel"
out.mkdir(exist_ok=True)
(out / "f1.ipynb").write_text(json.dumps(nb), encoding="utf-8")
meta = json.loads((f0 / "kernel-metadata.json").read_text(encoding="utf-8"))
meta.update(id="aymenslimani/rsna-knee-f1-f0-plus-student", title="RSNA knee F1 F0 plus student", code_file="f1.ipynb",
            kernel_sources=["aymenslimani/rsna-knee-s1-students-v3"])
json.dump(meta, open(out / "kernel-metadata.json", "w"), indent=2)
print("built", out, "arm", arm, "weight", weight)
