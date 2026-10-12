"""CPU-only Linux process-tree deadline test for the embedded M00 watchdog."""
import ast
import os
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

template = Path(__file__).resolve().parents[2] / "notebooks/m00_full_train/train_template.py"
if not template.exists():
    template = Path(__file__).with_name("watchdog_source.py")
tree = ast.parse(template.read_text(encoding="utf-8"))
code = next(ast.literal_eval(node.value) for node in tree.body
            if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "WATCHDOG_CODE"
                                                     for t in node.targets))
if "--emit-standalone" in sys.argv:
    source = Path(__file__).read_text(encoding="utf-8")
    start, end = source.index("template = "), source.index('\ncompile(code, "<watchdog>"') + 1
    standalone = source[:start] + "code = " + repr(code) + "\n" + source[end:]
    output = Path(sys.argv[sys.argv.index("--emit-standalone") + 1])
    output.write_text(standalone, encoding="utf-8")
    print(output)
    raise SystemExit(0)
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
