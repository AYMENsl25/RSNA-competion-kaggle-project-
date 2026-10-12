"""Audit a small DICOM sample without exporting identifying tags or pixels."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pydicom


def vector(values: object, expected: int) -> np.ndarray:
    array = np.asarray([float(value) for value in values], dtype=np.float64)
    if array.shape != (expected,) or not np.isfinite(array).all():
        raise ValueError(f"Expected {expected} finite values")
    return array


def digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()[:12]


def audit(paths: list[Path]) -> dict:
    records: list[dict] = []
    failures: list[dict] = []
    for path in paths:
        file_key = digest(path.name)
        try:
            dcm = pydicom.dcmread(path)
            orientation = vector(dcm.ImageOrientationPatient, 6)
            row, column = orientation[:3], orientation[3:]
            normal = np.cross(row, column)
            position = vector(dcm.ImagePositionPatient, 3)
            spacing = vector(dcm.PixelSpacing, 2)
            if np.any(spacing <= 0):
                raise ValueError("Nonpositive pixel spacing")
            row_norm = float(np.linalg.norm(row))
            col_norm = float(np.linalg.norm(column))
            row_col_dot = float(np.dot(row, column))
            if max(abs(row_norm - 1), abs(col_norm - 1), abs(row_col_dot)) > 1e-3:
                raise ValueError("Invalid orientation direction cosines")
            pixels = dcm.pixel_array
            if pixels.ndim != 2 or pixels.shape != (int(dcm.Rows), int(dcm.Columns)):
                raise ValueError("Decoded pixels do not match single-frame dimensions")
            if not np.isfinite(pixels).all():
                raise ValueError("Decoded pixels contain nonfinite values")
            records.append({
                "file_key": file_key,
                "_filename_internal": path.name,
                "study_key": digest(str(dcm.StudyInstanceUID)),
                "series_key": digest(str(dcm.SeriesInstanceUID)),
                "sop_matches_filename": path.stem == str(dcm.SOPInstanceUID),
                "transfer_syntax": str(dcm.file_meta.TransferSyntaxUID),
                "rows": int(dcm.Rows),
                "columns": int(dcm.Columns),
                "pixel_spacing_mm": [round(float(x), 6) for x in spacing],
                "normal": [round(float(x), 6) for x in normal],
                "projected_position_mm": round(float(np.dot(position, normal)), 6),
                "row_norm": round(row_norm, 6),
                "column_norm": round(col_norm, 6),
                "row_column_dot": round(row_col_dot, 6),
                "pixel_min": int(pixels.min()),
                "pixel_max": int(pixels.max()),
                "pixel_std": round(float(pixels.std()), 3),
                "laterality_tag_present": bool(getattr(dcm, "Laterality", None)),
                "patient_id_tag_present": bool(getattr(dcm, "PatientID", None)),
                "pixel_sha256": hashlib.sha256(pixels.tobytes()).hexdigest(),
            })
        except Exception as exc:  # record every parser or decoder failure
            failures.append({"file_key": file_key, "error_type": type(exc).__name__, "reason": str(exc)[:160]})

    groups: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        groups[record["series_key"]].append(record)
    group_checks = []
    for key, group in groups.items():
        positions = sorted(item["projected_position_mm"] for item in group)
        gaps = [round(b - a, 6) for a, b in zip(positions, positions[1:])]
        lex = sorted(group, key=lambda item: item["_filename_internal"])
        spatial = sorted(group, key=lambda item: item["projected_position_mm"])
        group_checks.append({
            "series_key": key,
            "sampled_slices": len(group),
            "unique_matrix_shapes": len({(item["rows"], item["columns"]) for item in group}),
            "unique_pixel_spacings": len({tuple(item["pixel_spacing_mm"]) for item in group}),
            "filename_order_matches_spatial_order": [x["file_key"] for x in lex] == [x["file_key"] for x in spatial],
            "duplicate_position_pairs_within_0p01mm": sum(gap <= 0.01 for gap in gaps),
            "sampled_gap_mm": gaps,
        })
    for record in records:
        record.pop("_filename_internal")
    return {
        "scope": "Small Kaggle CLI downloaded sample; no population-level geometry claim",
        "files_seen": len(paths),
        "decoded": len(records),
        "failures": failures,
        "transfer_syntax_counts": dict(Counter(item["transfer_syntax"] for item in records)),
        "sop_filename_mismatches": sum(not item["sop_matches_filename"] for item in records),
        "laterality_tag_present_count": sum(item["laterality_tag_present"] for item in records),
        "patient_id_tag_present_count": sum(item["patient_id_tag_present"] for item in records),
        "series_checks": group_checks,
        "slice_records": records,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    paths = sorted(args.input.glob("*.dcm"))
    result = audit(paths)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"Decoded {result['decoded']}/{result['files_seen']} sampled DICOMs; {len(result['series_checks'])} series")


if __name__ == "__main__":
    main()
