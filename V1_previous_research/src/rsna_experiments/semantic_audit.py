"""Header-only MRI protocol, semantic-slot and laterality audit.

This module never decodes PixelData and never changes supervision. It is a
go/no-go gate for later semantic multi-series models and laterality ablations.
"""
from pathlib import Path
import csv
import json
import re

import numpy as np

from .common import UID, locate, sha256, write_json
from .preprocessing import RECIPES, _folders, _headers, _laterality

SLOTS = ("SAG_FLUID_FS", "COR_FLUID_FS", "AX_FLUID_FS",
         "SAG_FLUID_NOFS", "COR_T1", "SAG_T1")


def _read_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        return list(reader), reader.fieldnames or []


def _write_csv(path, rows, fields):
    with Path(path).open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _slot(row, record):
    plane = record["plane"]
    fluid_and_fs, fs, fluid = record["preference"]
    text = " ".join(str(row.get(key, "")) for key in
                    ("SeriesDescription", "ProtocolName", "SequenceName"))
    text += " " + " ".join(str(getattr(record["ds"], key, "")) for key in
                            ("SeriesDescription", "ProtocolName", "SequenceName"))
    explicit_t1 = bool(re.search(r"(?:^|[^a-z0-9])t1(?:w|weighted)?(?:$|[^a-z0-9])", text, re.I))
    if fluid_and_fs:
        return f"{plane}_FLUID_FS"
    if plane == "SAG" and fluid and not fs:
        return "SAG_FLUID_NOFS"
    if plane in {"COR", "SAG"} and explicit_t1 and not fs:
        return f"{plane}_T1"
    return ""


def audit(config) -> Path:
    comp = Path(config["comp_dir"]).resolve(strict=True)
    manifest = Path(locate("experiment_manifest.csv", roots=config.get("input_roots"),
                           override=config.get("manifest_path"))).resolve(strict=True)
    out = Path(config["out_dir"]).resolve()
    if out.is_relative_to(comp) or manifest.is_relative_to(out):
        raise ValueError("Audit output must be separate from immutable inputs")
    out.mkdir(parents=True, exist_ok=True)
    manifest_rows, fields = _read_csv(manifest)
    if UID not in fields or not manifest_rows:
        raise ValueError("Prepared manifest is empty or missing StudyInstanceUID")
    uids = [row[UID] for row in manifest_rows]
    if len(set(uids)) != len(uids):
        raise ValueError("Prepared manifest contains duplicate studies")
    metadata = {}
    series_csv = comp / "train_series.csv"
    if series_csv.exists():
        rows, series_fields = _read_csv(series_csv)
        if not {UID, "SeriesInstanceUID"}.issubset(series_fields):
            raise ValueError("train_series.csv is missing UID columns")
        metadata = {(row[UID], row["SeriesInstanceUID"]): row for row in rows}

    series_rows, study_rows = [], []
    counts = {slot: 0 for slot in SLOTS}
    laterality_counts = {"L": 0, "R": 0, "unresolved": 0, "conflict": 0}
    for uid in uids:
        found = {slot: [] for slot in SLOTS}
        study_lateralities = set()
        for folder in _folders(comp, uid):
            row = metadata.get((uid, folder.name), {})
            result = {UID: uid, "SeriesInstanceUID": folder.name, "plane": "", "slot": "",
                      "slices": 0, "crop130_feasible": False, "laterality": "",
                      "laterality_source": "", "status": "error", "error": ""}
            try:
                records = _headers(folder, row, uid, folder.name, RECIPES["p0"])
                result.update(plane=records[0]["plane"], slices=len(records), status="ok")
                spacing = records[0]["spacing"]
                result["crop130_feasible"] = bool(spacing is not None and
                    np.all(np.rint(130 / spacing).astype(int) <= np.asarray(records[0]["shape"])))
                slot = _slot(row, records[0])
                result["slot"] = slot
                if slot:
                    found[slot].append(folder.name)
                try:
                    lat, source = _laterality(records)
                    result.update(laterality=lat, laterality_source=source)
                    study_lateralities.add(lat)
                except ValueError as exc:
                    result["error"] = str(exc)
            except Exception as exc:
                result["error"] = f"{type(exc).__name__}: {exc}"
            series_rows.append(result)
        if len(study_lateralities) == 1:
            lat = next(iter(study_lateralities)); laterality_counts[lat] += 1
        elif len(study_lateralities) > 1:
            lat = "conflict"; laterality_counts[lat] += 1
        else:
            lat = "unresolved"; laterality_counts[lat] += 1
        study = {UID: uid, "laterality": lat}
        for slot in SLOTS:
            study[slot] = len(found[slot])
            counts[slot] += int(bool(found[slot]))
        study_rows.append(study)

    series_fields = [UID, "SeriesInstanceUID", "plane", "slot", "slices",
                     "crop130_feasible", "laterality", "laterality_source", "status", "error"]
    _write_csv(out / "semantic_slot_series.csv", series_rows, series_fields)
    _write_csv(out / "semantic_slot_studies.csv", study_rows, list(study_rows[0]))
    receipt = {"schema": "rsna-semantic-audit-v1", "complete": True,
               "manifest_path": str(manifest), "manifest_sha256": sha256(manifest),
               "studies": len(study_rows), "series": len(series_rows),
               "slot_study_counts": counts, "laterality_study_counts": laterality_counts,
               "pixel_data_decoded": False,
               "warning": "Absence of fat suppression is never interpreted as T1."}
    write_json(out / "semantic_audit.json", receipt)
    return out
