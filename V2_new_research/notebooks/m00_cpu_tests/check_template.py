"""Run synthetic M00 correctness tests in Kaggle's offline CPU runtime."""

import json
import platform
from pathlib import Path

import pytest
import torch
import torchvision


# __EMBEDDED_TEST_FILES__
summary = {
    "synthetic_only": True,
    "torch": torch.__version__,
    "torchvision": torchvision.__version__,
    "python": platform.python_version(),
}
code = pytest.main([str(root / "tests" / "test_m00.py"), "-q", "--disable-warnings"])
summary["pytest_exit_code"] = int(code)
summary["passed"] = int(code) == 0
Path("/kaggle/working/m00_cpu_test_summary.json").write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2))
if code:
    raise SystemExit(int(code))
