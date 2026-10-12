"""Build isolated P02 offline CPU and bounded GPU pilot kernels from frozen P00 source."""
import ast
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src" / "p02_model.py"
BASE = ROOT / "notebooks"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build(kind):
    original = BASE / ("p00_cpu_check" if kind == "cpu" else "p00_gpu_pilot")
    target = BASE / ("p02_cpu_check" if kind == "cpu" else "p02_gpu_pilot")
    target.mkdir(exist_ok=True)
    original_script = original / ("check.py" if kind == "cpu" else "pilot.py")
    code = original_script.read_text(encoding="utf-8")
    assert code.count("COMPETITION_ROOTS = (") == 1
    model_source = SRC.read_text(encoding="utf-8").replace("from p00_model import P00\n", "")
    code = code.replace("COMPETITION_ROOTS = (", f"# Embedded p02_model.py SHA256: {sha(SRC)}\n{model_source}\n\nCOMPETITION_ROOTS = (")
    code = code.replace("'p00_model.py':", f"'p02_model.py': '{sha(SRC)}', 'p00_model.py':", 1)
    code = code.replace("P00()", "P02()")
    if kind == "gpu":
        # The successful P00 Kaggle v3 pilot and full run used fork workers.
        assert code.count('multiprocessing_context="spawn"') == 1
        code = code.replace('multiprocessing_context="spawn"', 'multiprocessing_context="fork"')
    if kind == "cpu":
        code = code.replace('"status": "passed",', '"status": "passed", "model": "P02",')
        gate = '''    # Two findings must select different valid windows; padding must never win.
    class TinyEncoder(nn.Module):
        def forward(self, images):
            return images[:, :, 0, 0]
    tiny = P02.__new__(P02)
    nn.Module.__init__(tiny)
    tiny.encoder = TinyEncoder()
    tiny.head = nn.Linear(3, 2, bias=False)
    with torch.no_grad():
        tiny.head.weight.copy_(torch.tensor([[1., 0., 0.], [0., 1., 0.]]))
    sample = torch.zeros((1, 1, 3, 3, 1, 1), requires_grad=True)
    with torch.no_grad():
        sample[0, 0, 0, :, 0, 0] = torch.tensor([5., 1., 0.])
        sample[0, 0, 1, :, 0, 0] = torch.tensor([2., 7., 0.])
        sample[0, 0, 2, :, 0, 0] = torch.tensor([999., 999., 0.])
    valid = torch.tensor([[[True, True, False]]])
    actual = tiny(sample, valid)
    assert torch.equal(actual, torch.tensor([[5., 7.]]))
    assert torch.equal(tiny(sample[:, :, [1, 0, 2]], valid), actual)
    actual.sum().backward()
    assert sample.grad[0, 0, 0, 0, 0, 0] == 1
    assert sample.grad[0, 0, 1, 1, 0, 0] == 1
    assert sample.grad[0, 0, 2].abs().sum() == 0
'''
        anchor = '    print(json.dumps({"status": "passed",'
        assert code.count(anchor) == 1
        code = code.replace(anchor, gate + anchor)
    else:
        code = code.replace('p00_gpu_pilot_summary.json', 'p02_gpu_pilot_summary.json')
        code = code.replace('"P00 projected full runtime gate failed"', '"P02 projected full runtime gate failed"')
    ast.parse(code)
    (target / original_script.name).write_text(code, encoding="utf-8")
    metadata = json.loads((original / "kernel-metadata.json").read_text(encoding="utf-8"))
    metadata["id"] = metadata["id"].replace("p00", "p02")
    metadata["title"] = metadata["title"].replace("P00", "P02")
    (target / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")


def build_full():
    original = BASE / "p00_full_train"
    target = BASE / "p02_full_train"
    target.mkdir(exist_ok=True)
    base_path = original / "train.py"
    code = base_path.read_text(encoding="utf-8")
    assert code.count("COMPETITION_ROOTS = (") == 1
    model_source = SRC.read_text(encoding="utf-8").replace("from p00_model import P00\n", "")
    code = code.replace("COMPETITION_ROOTS = (", f"# Embedded p02_model.py SHA256: {sha(SRC)}\n{model_source}\n\nCOMPETITION_ROOTS = (")
    code = code.replace("'p00_model.py':", f"'p02_model.py': '{sha(SRC)}', 'p00_full_train.py': '{sha(base_path)}', 'p00_model.py':", 1)
    code = code.replace("P00()", "P02()")
    for old, new in (("p00_full_train_summary.json", "p02_full_train_summary.json"),
                     ("p00_best.pt", "p02_best.pt"),
                     ("p00_val_predictions_epoch", "p02_val_predictions_epoch")):
        code = code.replace(old, new)
    code = code.replace("P00 full training:", "P02 full training:", 1)
    code = code.replace('"model": "P00 ImageNet-pretrained ResNet18 2.5D (3 adjacent slices), shared max over valid windows",',
                        '"model": "P02 ImageNet-pretrained ResNet18 2.5D, per-finding max of valid window logits",')
    assert 'multiprocessing_context="fork"' in code
    assert 'MAX_EPOCHS, LR = 0, 4, 2, 1, 3, 3e-4' in code
    ast.parse(code)
    (target / "train.py").write_text(code, encoding="utf-8")
    metadata = json.loads((original / "kernel-metadata.json").read_text(encoding="utf-8"))
    metadata["id"] = metadata["id"].replace("p00", "p02")
    metadata["title"] = metadata["title"].replace("P00", "P02")
    (target / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    for kind in ("cpu", "gpu"):
        build(kind)
    build_full()
