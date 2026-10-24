"""Explicit N4 pipeline: private CPU cache builds -> verified downloads -> local training.

Uploads require the explicit --upload-full-shards flag. No scheduler or automation is installed.
Run only after the user approves all eight full-cache notebook uploads.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
STATE=ROOT/'outputs/n4_pipeline'
PY=ROOT/'.venv/Scripts/python.exe'

def save(path: Path, value: dict):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value,indent=2),encoding='utf-8')
    tmp.replace(path)

def digest(path: Path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(16*1024*1024),b''):h.update(block)
    return h.hexdigest()

def cli(args, timeout=180):
    env=os.environ.copy()
    env['PYTHONPATH']=str(ROOT.parent/'.tools/kaggle')
    env['KAGGLE_CONFIG_DIR']=str(ROOT.parent/'.kaggle')
    p=subprocess.run([str(PY),'-m','kaggle',*args],cwd=ROOT,env=env,
                     text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=timeout)
    if p.returncode:raise RuntimeError(f'Kaggle command failed: {p.stdout[-2000:]}')
    return p.stdout

def verify_cache(folder: Path, expected_shard: int):
    import numpy as np
    s=json.loads((folder/'summary.json').read_text())
    assert s['shard']==expected_shard and s['nshard']==8 and s['pilot'] is False
    assert s['excluded_studies']==0 and s['series_failed']==0
    assert s['config']==dict(out_px=384,model_input=320,crop_mm=160,slots=6,control='N3a',training_epochs=12)
    for name,expected in s['source_sha256'].items():
        assert hashlib.sha256((ROOT/'src'/name).read_text(encoding='utf-8').encode()).hexdigest()==expected, f'Source mismatch: {name}'
    for name,expected in s['sha256'].items():
        assert digest(folder/name)==expected, f'Checksum mismatch: {folder/name}'
    a=np.load(folder/'slices.npy',mmap_mode='r')
    assert a.dtype==np.uint8 and a.shape[1:]==(384,384) and a.nbytes==s['slices_bytes']
    return s

def build_and_download(k: int):
    record=STATE/f's{k}.json'
    old=json.loads(record.read_text()) if record.exists() else {}
    folder=ROOT/f'outputs/cache_n4/s{k}'
    kernel=f'aymenslimani/rsna-knee-n4-native384-s{k}'
    def update(state,**extra):
        old.update(state=state,shard=k,kernel=kernel,time=time.time(),**extra)
        save(record,old);print(f's{k}: {state}',flush=True)
    if old.get('state')=='verified':
        verify_cache(folder,k);return
    if not old.get('uploaded'):
        update('uploading')
        output=cli(['kernels','push','-p',str(ROOT/f'notebooks/cache_n4/s{k}')])
        if 'successfully pushed' not in output:raise RuntimeError(output)
        update('building',uploaded=True,push_receipt=output)
    deadline=time.monotonic()+4*3600
    last=None
    while time.monotonic()<deadline:
        status=cli(['kernels','status',kernel]).strip()
        if status!=last:
            print(status,flush=True);last=status
        if 'KernelWorkerStatus.COMPLETE' in status:break
        if any(s in status for s in ('KernelWorkerStatus.ERROR','KernelWorkerStatus.CANCEL')):
            raise RuntimeError(status)
        time.sleep(45)
    else:raise TimeoutError(f'{kernel}: four-hour build deadline exceeded')
    folder.mkdir(parents=True,exist_ok=True)
    update('downloading_metadata')
    cli(['kernels','output',kernel,'-p',str(folder),'--file-pattern',r'(summary\.json|index\.csv|checksums\.sha256)$'])
    s=json.loads((folder/'summary.json').read_text())
    assert s['series_failed']==0 and s['excluded_studies']==0
    data=folder/'slices.npy'
    if not data.exists():
        assert shutil.disk_usage(ROOT).free>s['slices_bytes']+15*2**30,'Not enough free disk for shard plus reserve'
    update('downloading_pixels',bytes=s['slices_bytes'])
    for attempt in range(1,4):
        try:
            result=subprocess.run([str(PY),'-u','src/download_n4_cache.py',kernel,str(folder)],cwd=ROOT,timeout=6*3600+300)
            if result.returncode:raise RuntimeError(f'Resumable download exited {result.returncode}')
            receipt='Resumable HTTP Range download completed'
            verify_cache(folder,k)
            update('verified',download_receipt=receipt);return
        except (RuntimeError,AssertionError,subprocess.TimeoutExpired) as exc:
            update('download_retry',attempt=attempt,error=str(exc)[-2000:])
            if attempt==3:raise
            time.sleep(15)

def main():
    ap=argparse.ArgumentParser(__doc__)
    ap.add_argument('--upload-full-shards',action='store_true',required=True,
                    help='Explicit permission boundary: uploads all eight private full-cache notebooks')
    args=ap.parse_args()
    os.chdir(ROOT);STATE.mkdir(parents=True,exist_ok=True)
    lock=STATE/'pipeline.lock'
    try:fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY)
    except FileExistsError:raise RuntimeError('Pipeline lock exists; verify its PID before restarting')
    os.write(fd,str(os.getpid()).encode());os.close(fd)
    try:
        gpu=json.loads((ROOT/'outputs/n4_native384/gpu_feasibility.json').read_text())
        assert gpu['passed'] and gpu['grad_checkpointing']
        if (ROOT/'outputs/n4_native384/last.pt').exists():
            raise RuntimeError('Existing N4 checkpoint: use explicit reviewed resume, not a new pipeline')
        save(STATE/'run_status.json',dict(state='building_and_downloading',pid=os.getpid(),started=time.time()))
        # Two jobs at once keep CPU concurrency and local download/RAM use bounded.
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures=[pool.submit(build_and_download,k) for k in range(8)]
            for f in as_completed(futures):f.result()
        save(STATE/'run_status.json',dict(state='training',pid=os.getpid(),time=time.time()))
        env=os.environ.copy();env.update(OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')
        with (ROOT/'outputs/n4_native384.log').open('a',encoding='utf-8') as out, (ROOT/'outputs/n4_native384.stderr.log').open('a',encoding='utf-8') as err:
            p=subprocess.run([str(PY),'-u','src/run_n4.py'],cwd=ROOT,env=env,stdout=out,stderr=err)
        if p.returncode:raise RuntimeError(f'N4 trainer exited with code {p.returncode}; inspect stderr')
        save(STATE/'run_status.json',dict(state='completed',pid=os.getpid(),time=time.time()))
    except BaseException as exc:
        save(STATE/'run_status.json',dict(state='failed',pid=os.getpid(),error=str(exc),time=time.time()))
        raise
    finally:lock.unlink(missing_ok=True)

if __name__=='__main__':main()
