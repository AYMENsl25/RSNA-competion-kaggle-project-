"""Plain-assert checks for src/canon.py on data/dicom_pilot. Run: python tests/test_canon.py
(needs numpy, pydicom, torch)."""
import copy
import glob
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.append(str(ROOT / ".tools"))  # vendored pydicom fallback
import pydicom  # noqa: E402

from canon import OUT_PX, canon_series  # noqa: E402

BASE = pydicom.dcmread(sorted(glob.glob(str(ROOT / "data/dicom_pilot/*.dcm")))[0])


def series(img, r, c, spacing, n=6, gap=3.0, shuffle=True):
    """n copies of img on a regular stack; IPP chosen so the stack is consistent with (r, c)."""
    r, c = np.asarray(r, float), np.asarray(c, float)
    out = []
    for k in range(n):
        d = copy.deepcopy(BASE)
        d.Rows, d.Columns = img.shape
        d.PixelData = img.astype(np.uint16).tobytes()
        d.PixelSpacing = [spacing, spacing]
        d.ImageOrientationPatient = [*r, *c]
        d.ImagePositionPatient = list(np.array([10.0, -20.0, 30.0]) + k * gap * np.cross(r, c))
        d.InstanceNumber = k + 1
        out.append(d)
    return out[::-1] if shuffle else out


def bright_extent(x):
    m = x > 127
    rows, cols = np.flatnonzero(m.any(1)), np.flatnonzero(m.any(0))
    return rows[-1] - rows[0] + 1, cols[-1] - cols[0] + 1


# 1. real pilot pixels: dtype/shape, monotone positions, laterality tag
img = BASE.pixel_array
x, meta = canon_series(series(img, BASE.ImageOrientationPatient[:3], BASE.ImageOrientationPatient[3:], 0.1562))
assert x.dtype == np.uint8 and x.shape == (6, OUT_PX, OUT_PX), (x.dtype, x.shape)
assert np.all(np.diff(meta["positions"]) > 0) and np.allclose(np.diff(meta["positions"]), 3.0, atol=1e-3)
assert meta["laterality"] == "L" and meta["laterality_source"] == "tag"
assert meta["plane_geom"] == "Axial" and x.max() == 255 and x.min() == 0

# 2. crop size: 80 mm bright square -> 128 px of 256 (160 mm FOV), for pad (0.1562) and crop (0.5) cases
for sp in (0.1562, 0.5):
    side = int(round(80 / sp))
    a = np.full((960, 960), 50, np.uint16)
    s = (960 - side) // 2
    a[s:s + side, s:s + side] = 1000
    x, meta = canon_series(series(a, [1, 0, 0], [0, 1, 0], sp))
    h, w = bright_extent(x[0])
    assert abs(h - 128) <= 2 and abs(w - 128) <= 2, (sp, h, w)

# 3. canonical orientation: flipped / transposed acquisitions give the same image
a = np.full((400, 400), 50, np.uint16)
a[20:120, 40:100] = 1000  # asymmetric marker near top-left
ref, mref = canon_series(series(a, [1, 0, 0], [0, 1, 0], 0.4))
for (r, c, b) in (([-1, 0, 0], [0, 1, 0], a[:, ::-1]),   # cols run right->left
                  ([1, 0, 0], [0, -1, 0], a[::-1]),      # rows run posterior->anterior
                  ([0, 1, 0], [1, 0, 0], a.T)):          # rows/cols swapped
    x, m = canon_series(series(b, r, c, 0.4))
    assert np.array_equal(x, ref), (r, c)
    assert np.all(np.diff(m["positions"]) > 0)

# 4. laterality fallback from image-centre LPS x
d = series(img, [1, 0, 0], [0, 1, 0], 0.1562)
for e in d:
    del e.Laterality
    e.ImagePositionPatient = [-200.0, -75.0, e.ImagePositionPatient[2]]  # centre x ~ -125 mm
assert canon_series(d)[1]["laterality"] == "R"

# 5. paths work too (single real slice)
x, meta = canon_series(sorted(glob.glob(str(ROOT / "data/dicom_pilot/*.dcm")))[:1])
assert x.shape == (1, OUT_PX, OUT_PX)
print("test_canon OK")
