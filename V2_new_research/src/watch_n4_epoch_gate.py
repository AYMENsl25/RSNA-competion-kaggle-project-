"""One-time N4 epoch3/4 gate requested by user; no recurring app schedule."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/n4_epoch_gate'
PS=Path(os.environ['SystemRoot'])/'System32/WindowsPowerShell/v1.0/powershell.exe'

def decide(candidate,control):
    by={e['epoch']:e for e in control}
    for e in candidate:
        ep=e['epoch']
        if ep in (3,4):
            gaps={label:e['val'][label]['macro']-by[ep]['val'][label]['macro'] for label in ('public','v3')}
            if all(v < -.03 for v in gaps.values()):return 'stop',ep,gaps
    if any(e['epoch']>=4 for e in candidate):return 'continue',4,{}
    return 'wait',None,{}

def save(name,data):
    p=OUT/name;temp=p.with_suffix('.tmp');temp.write_text(json.dumps(data,indent=2));temp.replace(p)

def stop_n4(expected_pid):
    code=r'''
$all=@(Get-CimInstance Win32_Process | Where-Object {$_.Name -eq 'python.exe'})
$roots=@($all | Where-Object {$_.CommandLine -match 'src[/\\](run_n4|run_n4_pipeline)\.py(?:\s|$)'})
if (-not ($roots.ProcessId -contains EXPECTED)) {throw 'Expected N4 PID/command not found; no processes stopped'}
$ids=@($roots.ProcessId)
do {$next=@($all | Where-Object {$_.ParentProcessId -in $ids -and $_.ProcessId -notin $ids});$ids+=@($next.ProcessId)} while ($next.Count -gt 0)
Stop-Process -Id @($roots.ProcessId) -Force -ErrorAction SilentlyContinue
$children=@($ids | Where-Object {$_ -notin $roots.ProcessId})
if ($children.Count) {Stop-Process -Id $children -Force -ErrorAction SilentlyContinue}
Write-Output ('Stopped N4 tree PIDs: '+($ids -join ','))
'''.replace('EXPECTED',str(int(expected_pid)))
    result=subprocess.run([str(PS),'-NoProfile','-Command',code],capture_output=True,text=True)
    if result.returncode:raise RuntimeError(result.stderr)
    print(result.stdout,flush=True)

def main():
    os.chdir(ROOT);OUT.mkdir(parents=True,exist_ok=True)
    lock=OUT/'guard.lock'
    fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY);os.write(fd,str(os.getpid()).encode());os.close(fd)
    try:
        save('run_status.json',dict(state='waiting_for_epoch3_or4',pid=os.getpid(),rule='Both public and v3 AUC >0.03 below N3a at same epoch'))
        control=json.loads((ROOT/'outputs/n3a_attn6/log.json').read_text())['epochs']
        deadline=time.monotonic()+2*3600
        while time.monotonic()<deadline:
            try:epochs=json.loads((ROOT/'outputs/n4_native384/log.json').read_text())['epochs']
            except (OSError,ValueError):time.sleep(5);continue
            action,epoch,gaps=decide(epochs,control)
            if action=='continue':
                save('run_status.json',dict(state='N4_continues',reviewed_epoch=epoch));return
            if action=='stop':
                assert (ROOT/'outputs/n4_native384/last.pt').exists()
                decision=dict(state='early_stopped',epoch=epoch,gaps=gaps,reason='User requested switch to N5 if epoch3/4 remains low',time=time.time())
                save('decision.json',decision)
                status=json.loads((ROOT/'outputs/n4_native384/run_status.json').read_text())
                assert status['state']=='running',status
                stop_n4(status['pid'])
                for folder in ('n4_native384','n4_pipeline'):
                    p=ROOT/f'outputs/{folder}/run_status.json';p.write_text(json.dumps(decision,indent=2))
                save('run_status.json',dict(state='starting_N5',decision=decision,pid=os.getpid()))
                env=os.environ.copy();env.update(OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')
                with (ROOT/'outputs/n5_lr.log').open('w') as out,(ROOT/'outputs/n5_lr.stderr.log').open('w') as err:
                    p=subprocess.Popen([sys.executable,'-u','src/run_n5.py'],cwd=ROOT,env=env,stdout=out,stderr=err)
                save('run_status.json',dict(state='N5_launched',n5_launcher_pid=p.pid,decision=decision))
                print(f'N5 launched PID {p.pid}',flush=True);return
            time.sleep(10)
        raise TimeoutError('N4 epoch3/4 gate timed out; N4 was left running')
    except BaseException as exc:
        save('run_status.json',dict(state='failed',error=str(exc)));raise
    finally:lock.unlink(missing_ok=True)

if __name__=='__main__':main()
