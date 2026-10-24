"""F2: deterministic windows from existing N4 caches; no reports or expert labels.

Only candidate partitions are eligible. Selected series never get replaced on failure.
Both heads receive exactly the same features, valid masks and study ordering.
"""
import csv
import hashlib
from pathlib import Path

import numpy as np

TARGETS = ("ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA",
           "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture")
PLANES = ("Sagittal", "Coronal", "Axial")
SPLIT_SHA = "dbe879ab5cb719e09c0f9aec612d5b6b53648d6f32dd8d405a075482339b3819"
LABEL_SHA = "96444cd3c7d396bae4a71454317d9cbe0689759b5278b30939781e04d0774537"


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def cohort(split_path, labels_path, strict=True):
    """Parse proxy values only for non-excluded studies; hash checks pin P00 inputs."""
    if strict and (sha(split_path) != SPLIT_SHA or sha(labels_path) != LABEL_SHA):
        raise ValueError("frozen split/weak-label hash mismatch")
    partitions, groups, counts = {}, {}, {}
    with open(split_path, newline="", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            u, g, p = r["StudyInstanceUID"], r["group_id"], r["partition"]
            if u in partitions or p not in {"candidate_train", "candidate_validation", "excluded"}:
                raise ValueError("duplicate study or invalid partition")
            if g in groups and groups[g] != p:
                raise ValueError("group crosses split")
            partitions[u], groups[g] = p, p
            counts[p] = counts.get(p, 0) + 1
    if strict and counts != {"candidate_train": 3457, "candidate_validation": 885, "excluded": 65}:
        raise ValueError("frozen cohort counts mismatch")
    values, seen = {}, set()
    with open(labels_path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != ["StudyInstanceUID", *TARGETS]:
            raise ValueError("weak-label schema mismatch")
        for r in reader:
            u = r["StudyInstanceUID"]
            if u in seen or u not in partitions:
                raise ValueError("weak-label duplicate/unknown study")
            seen.add(u)
            if partitions[u] == "excluded":
                continue  # Do not parse excluded label values.
            p = np.array([float(r[c]) for c in TARGETS])
            if not np.isfinite(p).all() or (p < 0).any() or (p > 1).any():
                raise ValueError("invalid proxy probability")
            values[u] = ((p > .5).astype(np.float32), p != .5)
    eligible = {u for u, p in partitions.items() if p != "excluded"}
    if not eligible <= values.keys():
        raise ValueError("missing eligible weak labels")
    uids = sorted(u for u in eligible if values[u][1].any())
    if strict:
        n = [sum(partitions[u] == p for u in uids) for p in ("candidate_train", "candidate_validation")]
        if n != [3428, 863]:
            raise ValueError("eligible cohort mismatch")
    return uids, partitions, values


def select(rows):
    """One distinct series per (plane, fluid) slot, same rule for A and B."""
    candidates, seen = {}, set()
    for r in rows:
        uid = r["series"]
        if uid in seen:
            raise ValueError("duplicate series UID")
        seen.add(uid)
        plane, fluid = r["plane"], int(r["fluid"] or 0)
        if plane not in PLANES or fluid not in (0, 1):
            raise ValueError("invalid series slot metadata")
        slot = 2 * PLANES.index(plane) + (1 - fluid)
        key = (-int(r["fatsat"] or 0), uid)
        if slot not in candidates or key < candidates[slot][0]:
            candidates[slot] = (key, r)
    return {s: r for s, (_, r) in candidates.items()}


def centers_geometry(row, windows):
    """Physical centers and gaps, without assuming registration between series."""
    if row["ok"] != "1" or int(row["bad_slices"] or 0):
        raise ValueError("selected series failed cache QC")
    pos = np.array([float(v) for v in row["positions"].split(";")])
    if len(pos) != int(row["length"]) or len(pos) < 3 or not np.isfinite(pos).all():
        raise ValueError("invalid selected-series positions")
    if not (np.diff(pos) > 0).all() or row["plane_geom"] != row["plane"]:
        raise ValueError("nonascending geometry or plane mismatch")
    centers = np.unique(np.rint(np.linspace(1, len(pos) - 2, windows)).astype(int))
    extent = pos[-1] - pos[0]
    geometry = np.stack(((pos[centers] - np.median(pos)) / 50,
                         (pos[centers] - pos[centers - 1]) / 10,
                         (pos[centers + 1] - pos[centers]) / 10,
                         np.full(len(centers), extent / 100),
                         (pos[centers] - pos[0]) / extent - .5), axis=1).astype(np.float32)
    return centers, geometry


def cache_index(root, eligible, partitions):
    """Verify small index digests; record, but do not rehash 120 GB of slice arrays."""
    by_study, arrays, provenance = {}, {}, {}
    all_series = set()
    for shard in range(8):
        folder = Path(root) / f"s{shard}"
        checks = dict(line.split()[::-1] for line in (folder / "checksums.sha256").read_text().splitlines())
        index = folder / "index.csv"
        if sha(index) != checks["index.csv"]:
            raise ValueError("cache index integrity failure")
        a = np.load(folder / "slices.npy", mmap_mode="r", allow_pickle=False)
        if a.dtype != np.uint8 or a.ndim != 3 or a.shape[1:] != (384, 384):
            raise ValueError("N4 cache shape/dtype mismatch")
        arrays[shard] = a
        provenance[f"s{shard}"] = {"index_sha256_verified": checks["index.csv"],
                                    "slices_sha256_recorded_not_rehashed": checks["slices.npy"],
                                    "shape": list(a.shape)}
        with index.open(newline="", encoding="utf-8-sig") as f:
            for r in csv.DictReader(f):
                u = r["study"]
                if u not in eligible:
                    continue
                if r["exclude"] not in ("", "0", "False", "false") or r["partition"] != partitions[u]:
                    raise ValueError("cache exclusion/partition mismatch")
                if r["series"] in all_series:
                    raise ValueError("series UID appears more than once")
                all_series.add(r["series"])
                start, length = int(r["start"]), int(r["length"])
                if start < 0 or length < 0 or start + length > len(a):
                    raise ValueError("cache range out of bounds")
                by_study.setdefault(u, []).append(dict(r, shard=shard))
    if set(by_study) != set(eligible):
        raise ValueError("cache does not cover eligible cohort")
    return by_study, arrays, provenance
