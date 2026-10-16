"""Bundle the exact E01 training preprocessing/model sources into one Kaggle script.
Writes infer.py (GPU submission kernel) and preflight/infer.py (same script, CPU kernel)."""
from pathlib import Path
import hashlib

ROOT = Path(__file__).resolve().parent
sources = {name: (ROOT / name).read_text(encoding="utf-8") for name in ("m00_data.py", "m00_model.py")}
template = (ROOT / "infer_template.py").read_text(encoding="utf-8")
imports = "from m00_data import ROLES, build_study, load_series, select_series\nfrom m00_model import M00, TARGETS\n"
check = "if sha256(Path(__file__).with_name(filename)) != digest:"
marker = "CHECKPOINT_NAME = "
assert template.count(imports) == template.count(check) == template.count(marker) == 1
template = template.replace(imports, "").replace(
    check, "if hashlib.sha256(EMBEDDED_SOURCES[filename].encode('utf-8')).hexdigest() != digest:")
embed = "EMBEDDED_SOURCES = " + repr(sources) + "\n"
embed += "exec(EMBEDDED_SOURCES['m00_data.py'], globals())\nexec(EMBEDDED_SOURCES['m00_model.py'], globals())\n\n"
standalone = template.replace(marker, embed + marker)
for path in (ROOT / "infer.py", ROOT / "preflight" / "infer.py"):
    path.write_text(standalone, encoding="utf-8")
print("built infer.py", hashlib.sha256(standalone.encode()).hexdigest())
