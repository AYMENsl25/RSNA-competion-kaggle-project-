"""CPU-only Linux process-tree deadline test for the embedded M00 watchdog."""
import ast
import os
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

code = "import os, signal, sys, time\nparent = int(sys.argv[1]); deadline = float(sys.argv[2])\ndef identity(pid):\n    try:\n        fields = open('/proc/%d/stat' % pid).read().rsplit(')', 1)[1].split()\n        return int(fields[1]), fields[19]\n    except (OSError, IndexError, ValueError):\n        return None\ndef tree():\n    children = {}\n    for name in os.listdir('/proc'):\n        if name.isdigit():\n            state = identity(int(name))\n            if state and int(name) != os.getpid(): children.setdefault(state[0], []).append((int(name), state[1]))\n    found = []; pending = [parent]\n    while pending:\n        for pid, start in children.get(pending.pop(), []):\n            found.append((pid, start)); pending.append(pid)\n    return found\ndef alive(pid, start):\n    state = identity(pid)\n    return state is not None and state[1] == start\ndef send(pid, sig):\n    try: os.kill(pid, sig)\n    except (ProcessLookupError, OSError): pass\nparent_state = identity(parent)\nif parent_state is None: sys.exit(0)\nwhile time.monotonic() < deadline - 60:\n    if os.getppid() != parent: sys.exit(0)\n    time.sleep(min(1, max(0, deadline - 60 - time.monotonic())))\nif os.getppid() != parent: sys.exit(0)\ntracked = tree()\nsend(parent, signal.SIGTERM)\ntime.sleep(max(0, deadline - time.monotonic()))\nall_children = {(pid, start) for pid, start in tracked + tree()}\nfor pid, start in all_children:\n    if alive(pid, start): send(pid, signal.SIGKILL)\nif alive(parent, parent_state[1]): send(parent, signal.SIGKILL)\n"
compile(code, "<watchdog>", "exec")
if not Path("/proc/self/stat").exists():
    print("watchdog syntax passed; process-tree execution requires Linux")
    raise SystemExit(0)

if "--worker" in sys.argv:
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    while True:
        time.sleep(1)

if "--parent" in sys.argv:
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    worker = subprocess.Popen([sys.executable, __file__, "--worker"])
    Path(sys.argv[2]).write_text(str(worker.pid))
    deadline = time.monotonic() + 3
    fast = code.replace("deadline - 60", "deadline - 2")
    subprocess.Popen([sys.executable, "-c", fast, str(os.getpid()), str(deadline)])
    while True:
        time.sleep(1)

with tempfile.TemporaryDirectory() as temp:
    pidfile = Path(temp) / "worker.pid"
    parent = subprocess.Popen([sys.executable, __file__, "--parent", str(pidfile)])
    try:
        parent.wait(timeout=8)
        assert parent.returncode == -signal.SIGKILL, parent.returncode
        worker_pid = int(pidfile.read_text())
        time.sleep(0.1)
        try:
            state = Path(f"/proc/{worker_pid}/stat").read_text().rsplit(")", 1)[1].split()[0]
        except FileNotFoundError:
            state = "gone"
        assert state in ("Z", "gone"), f"worker survived: {state}"
        print("watchdog killed SIGTERM-ignoring parent and worker by deadline")
    finally:
        if parent.poll() is None:
            parent.kill()
        if pidfile.exists():
            pid = int(pidfile.read_text())
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
