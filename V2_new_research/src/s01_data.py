"""S01 multi-series coverage on top of the unchanged M00 data path.

Sole change vs P00: per plane, up to SERIES_PER_ROLE distinct series (same fluid-sensitive >
fat-suppressed > smallest-UID key as m00 select_series), sharing the unchanged per-plane window
budget (4 train / 16 validation -> 2+2 / 8+8; one series keeps all). Windows are concatenated
within each role so tensors stay [3, W, 3, 224, 224]. Each series is decoded independently with
m00 load_series; any failure propagates as a study failure, never a substitute series.
"""
import torch

from m00_data import INFER_WINDOWS, ROLES, SIZE, TRAIN_WINDOWS, StudyError, series_windows

SERIES_PER_ROLE = 2


def select_series_multi(rows, k=SERIES_PER_ROLE):
    """rows: train_series.csv rows for ONE study. -> {role: [up to k distinct UIDs, best first]}.
    A UID appearing under two planes raises StudyError (no silent reassignment)."""
    keyed = {r: {} for r in ROLES}
    for row in rows:
        role = str(row["Anatomical_Plane"]).strip().capitalize()
        if role not in keyed:
            continue
        uid = str(row["SeriesInstanceUID"])
        key = (-int(row["Fluid_Sensitive"] or 0), -int(row["Fat_Suppression"] or 0), uid)
        keyed[role][uid] = min(key, keyed[role].get(uid, key))
    out = {r: [u for _, u in sorted((v, u) for u, v in keyed[r].items())[:k]] for r in ROLES}
    chosen = [u for r in ROLES for u in out[r]]
    if len(chosen) != len(set(chosen)):
        raise StudyError("series UID selected for more than one plane")
    return out


def split_budget(total, n_series):
    """Divide a per-plane window budget across selected series: (4,2)->[2,2], (16,1)->[16]."""
    if n_series == 0:
        return []
    base, extra = divmod(total, n_series)
    return [base + (i < extra) for i in range(n_series)]


def build_study_multi(role_series, train, rng=None):
    """role_series: {role: [(vol, ok), ...] in selection order}. -> (x [3,W,3,224,224], mask [3,W]).
    Train draws use one seeded rng sequentially across series; validation is deterministic."""
    n = TRAIN_WINDOWS if train else INFER_WINDOWS
    xs, ms = [], []
    for r in ROLES:
        series = role_series.get(r) or []
        if series:
            parts = [series_windows(vol, ok, share, rng if train else None)
                     for (vol, ok), share in zip(series, split_budget(n, len(series)))]
            x, m = torch.cat([p[0] for p in parts]), torch.cat([p[1] for p in parts])
        else:
            x, m = torch.zeros(n, 3, SIZE, SIZE), torch.zeros(n, dtype=torch.bool)
        xs.append(x), ms.append(m)
    mask = torch.stack(ms)
    if not mask.any():
        raise StudyError("study has no valid window in any role")
    return torch.stack(xs), mask
