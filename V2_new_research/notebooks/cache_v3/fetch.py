"""Resumable download of cache shard kernel outputs.

The kaggle CLI (2.2.4) reads each file fully into RAM (response.content), which breaks on ~10 GB
slices.npy. This re-lists fresh signed URLs on every attempt (they expire) and appends with an
HTTP Range request from the current file size, then verifies sha256.
Usage: python fetch.py OUT_DIR 0 1 2 3 4
"""
import subprocess, sys, time
from pathlib import Path
import requests
from kaggle.api.kaggle_api_extended import KaggleApi
from kagglesdk.kernels.types.kernels_api_service import ApiListKernelSessionOutputRequest

api = KaggleApi(); api.authenticate()

def urls(k):
    req = ApiListKernelSessionOutputRequest()
    req.user_name, req.kernel_slug = "aymenslimani", f"rsna-knee-v3-cache-s{k}"
    with api.build_kaggle_client() as kaggle:
        return {f.file_name: f.url for f in kaggle.kernels.kernels_api_client.list_kernel_session_output(req).files}

def fetch(k, name, path):
    for attempt in range(200):
        have = path.stat().st_size if path.exists() else 0
        try:
            r = requests.get(urls(k)[name], headers={"Range": f"bytes={have}-"}, stream=True, timeout=60)
            if r.status_code == 416:  # already complete
                return
            if r.status_code == 200 and have:  # server ignored Range: never truncate, just retry
                raise IOError("range ignored")
            r.raise_for_status()
            with open(path, "ab") as f:
                for chunk in r.iter_content(8 << 20):
                    f.write(chunk)
            return
        except Exception as e:
            print(f"s{k} {name} retry {attempt} at {have/2**30:.2f} GiB: {e}", flush=True)
            time.sleep(5)
    raise RuntimeError(f"s{k} {name} failed")

out = Path(sys.argv[1])
for k in sys.argv[2:]:
    d = out / f"s{k}"
    if (d / "OK").exists():
        continue
    d.mkdir(parents=True, exist_ok=True)
    for name in urls(k):
        fetch(k, name, d / name)
        print(f"s{k} {name} done", flush=True)
    ok = subprocess.run(["sha256sum", "-c", "checksums.sha256"], cwd=d).returncode == 0
    if ok:
        (d / "OK").touch()
    print(f"s{k} {'VERIFIED' if ok else 'CHECKSUM FAIL'}", flush=True)
