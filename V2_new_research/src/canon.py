"""V3 canonical series preprocessing, shared by the cache builder and offline inference.

canon_series(datasets or paths) -> (uint8 [N,256,256], meta):
  IPP-projected slice order (m00_data.order_slices), slices ascending along +LPS axis of the
  dominant normal; in-plane flips/transposes only (no interpolation) so that
    sagittal: cols -> +y (posterior), rows -> -z (inferior)
    coronal:  cols -> +x (patient left), rows -> -z
    axial:    cols -> +x, rows -> +y
  160 mm physical square centre crop/pad, antialiased area resize to 256 px (0.625 mm/px),
  foreground (Otsu) p0.5-p99.8 -> uint8. Laterality is only reported (meta), never applied.
"""
import numpy as np
import torch
import torch.nn.functional as F

from m00_data import SeriesError, order_slices, slice_pixels, slice_position

OUT_PX, CROP_MM, LO_Q, HI_Q, LAT_MM = 256, 160.0, 0.5, 99.8, 20.0
AXES = np.eye(3)
# normal axis -> (target col direction, target row direction) in LPS
TARGETS = {0: (AXES[1], -AXES[2]), 1: (AXES[0], -AXES[2]), 2: (AXES[0], AXES[1])}
PLANE = {0: "Sagittal", 1: "Coronal", 2: "Axial"}


def _otsu(v):
    hist, edges = np.histogram(v, 256)
    c = (edges[:-1] + edges[1:]) / 2
    w0 = np.cumsum(hist).astype(float)
    w1 = w0[-1] - w0
    s0 = np.cumsum(hist * c)
    with np.errstate(divide="ignore", invalid="ignore"):
        var = w0 * w1 * (s0 / w0 - (s0[-1] - s0) / w1) ** 2
    return c[int(np.nanargmax(var[:-1]))]


def _fit(a, n, axis, fill):
    """Centre crop or symmetric pad `a` to length n along axis."""
    h = a.shape[axis]
    if h < n:
        pad = [(0, 0)] * a.ndim
        pad[axis] = ((n - h) // 2, n - h - (n - h) // 2)
        return np.pad(a, pad, constant_values=fill)
    s = (h - n) // 2
    return a[(slice(None),) * axis + (slice(s, s + n),)]


def _laterality(ds, r, c, sp):
    for tag in ("Laterality", "ImageLaterality"):
        v = str(getattr(ds, tag, "") or "").strip().upper()[:1]
        if v in ("L", "R"):
            return v, "tag", float("nan")
    ctr = (np.asarray(ds.ImagePositionPatient, float) + r * sp[1] * (int(ds.Columns) - 1) / 2
           + c * sp[0] * (int(ds.Rows) - 1) / 2)
    x = float(ctr[0])
    return ("L" if x > LAT_MM else "R" if x < -LAT_MM else "U"), "ipp_x", x


def canon_series(datasets, *, out_px=OUT_PX):
    if not isinstance(out_px, int) or out_px < 1:
        raise ValueError("out_px must be a positive integer")
    if datasets and not hasattr(datasets[0], "pixel_array"):
        import pydicom
        datasets = [pydicom.dcmread(str(p)) for p in datasets]
    ds = order_slices(datasets)
    if slice_position(ds[0]) is None:
        raise SeriesError("no orientation tags; cannot canonicalise")
    iop = np.asarray(ds[0].ImageOrientationPatient, float)
    r0, c0 = iop[:3], iop[3:]
    n = np.cross(r0, c0)
    ax = int(np.argmax(np.abs(n)))
    if n[ax] < 0:  # ascending along +axis
        ds, n = ds[::-1], -n
    pos = np.array([float(np.asarray(d.ImagePositionPatient, float) @ n) for d in ds])
    sp = np.asarray(ds[0].PixelSpacing, float)  # (row spacing, col spacing)
    lat, lat_src, ctr_x = _laterality(ds[len(ds) // 2], r0, c0, sp)

    tc, tr = TARGETS[ax]
    transpose = abs(r0 @ tc) < abs(c0 @ tc)
    r, c = (c0, r0) if transpose else (r0, c0)
    if transpose:
        sp = sp[::-1]
    flip_c, flip_r = bool(r @ tc < 0), bool(c @ tr < 0)
    ch, cw = (int(round(CROP_MM / s)) for s in sp)
    if len({(int(d.Rows), int(d.Columns)) for d in ds}) != 1:
        raise SeriesError("inconsistent slice shapes")

    out, bad = [], 0
    for d in ds:
        a = slice_pixels(d)
        if not np.isfinite(a).all() or a.max() <= a.min():
            bad += 1
            a = np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)
        if transpose:
            a = a.T
        a = a[::-1 if flip_r else 1, ::-1 if flip_c else 1]
        fill = float(a.min())
        a = _fit(_fit(a, ch, 0, fill), cw, 1, fill)
        t = torch.from_numpy(np.ascontiguousarray(a, dtype=np.float32))[None, None]
        mode = "area" if min(ch, cw) >= out_px else "bilinear"
        out.append(F.interpolate(t, size=(out_px, out_px), mode=mode,
                                 **({"align_corners": False} if mode == "bilinear" else {}))[0, 0].numpy())
    vol = np.stack(out)
    v = vol.ravel()[:: max(1, vol.size // 2_000_000)]
    fg = v[v > _otsu(v)]
    if fg.size < 0.01 * v.size:
        fg = v
    lo, hi = (float(x) for x in np.percentile(fg, [LO_Q, HI_Q]))
    if not hi > lo:  # flat foreground: fall back to all pixels
        lo, hi = (float(x) for x in np.percentile(v, [LO_Q, HI_Q]))
    if not hi > lo:
        raise SeriesError("degenerate intensity range")
    img = np.round(np.clip((vol - lo) / (hi - lo), 0, 1) * 255).astype(np.uint8)
    meta = {
        "plane_geom": PLANE[ax], "n": len(ds), "positions": pos.tolist(),
        "row_spacing": float(sp[0]), "col_spacing": float(sp[1]),
        "thickness": float(getattr(ds[0], "SliceThickness", None) or "nan"),
        "laterality": lat, "laterality_source": lat_src, "centre_x_mm": ctr_x,
        "lo": lo, "hi": hi, "transpose": bool(transpose), "flip_rows": flip_r, "flip_cols": flip_c,
        "bad_slices": bad, "orig_shape": (int(ds[0].Rows), int(ds[0].Columns)),
    }
    return img, meta
