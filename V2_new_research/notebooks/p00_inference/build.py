"""Bundle exact trained model/preprocessing sources into one Kaggle script."""
from pathlib import Path
import hashlib

ROOT = Path(__file__).resolve().parent
sources = {name: (ROOT / name).read_text(encoding="utf-8") for name in ("m00_data.py", "m00_model.py")}
template = (ROOT / "infer_template.py").read_text(encoding="utf-8")
template = template.replace("from m00_data import ROLES, build_study, load_series, select_series\nfrom m00_model import M00, TARGETS\n", "")
template = template.replace(
    "if sha256(Path(__file__).with_name(filename)) != digest:",
    "if hashlib.sha256(EMBEDDED_SOURCES[filename].encode('utf-8')).hexdigest() != digest:",
)
marker = "CHECKPOINT_SHA256 = "
embed = "EMBEDDED_SOURCES = " + repr(sources) + "\n"
embed += "exec(EMBEDDED_SOURCES['m00_data.py'], globals())\nexec(EMBEDDED_SOURCES['m00_model.py'], globals())\n\n"
assert template.count(marker) == 1
standalone = template.replace(marker, embed + marker)
(ROOT / "infer.py").write_text(standalone, encoding="utf-8")
print("built", ROOT / "infer.py", hashlib.sha256(standalone.encode()).hexdigest())
