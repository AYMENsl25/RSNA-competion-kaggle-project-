"""Offline competition I/O smoke. Outputs constant values; never submit as a model."""

from __future__ import annotations

import csv
import json
import math
import shutil
import time
import importlib.util
from collections import Counter, defaultdict
from pathlib import Path

import pydicom
from pydicom.pixels import get_decoder
from pydicom.uid import JPEGLosslessSV1, JPEG2000Lossless


start = time.perf_counter()
data = Path("/kaggle/input/competitions/rsna-knee-abnormality-detection")
if not (data / "test.csv").exists():
    data = Path("/kaggle/input/rsna-knee-abnormality-detection")
assert (data / "test.csv").exists(), "Competition test.csv is not mounted"


def read(name: str) -> tuple[list[str], list[dict[str, str]]]:
    with (data / name).open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), list(reader)


test_columns, test = read("test.csv")
series_columns, series = read("test_series.csv")
sample_columns, sample = read("sample_submission.csv")
uid_column = "StudyInstanceUID"
assert test_columns == [uid_column]
assert len(sample_columns) == 13 and sample_columns[0] == uid_column
assert [row[uid_column] for row in sample] == [row[uid_column] for row in test]
assert len({row[uid_column] for row in test}) == len(test)

series_by_study = defaultdict(list)
for row in series:
    series_by_study[row[uid_column]].append(row["SeriesInstanceUID"])
failed = []
decoded = 0
test_transfer_syntaxes = Counter()
test_first_instance_failures = Counter()
for row in series:
    uid, sid = row[uid_column], row["SeriesInstanceUID"]
    try:
        files = sorted((data / "test_series" / uid / sid).glob("*.dcm"))
        if not files:
            raise FileNotFoundError("Missing DICOM series")
        header = pydicom.dcmread(files[0], stop_before_pixels=True)
        if str(header.StudyInstanceUID) != uid or str(header.SeriesInstanceUID) != sid:
            raise ValueError("UID/path mismatch")
        test_transfer_syntaxes[str(header.file_meta.TransferSyntaxUID)] += 1
    except Exception as exc:
        test_first_instance_failures[type(exc).__name__] += 1
for index, row in enumerate(test):
    uid = row[uid_column]
    try:
        assert series_by_study[uid], "No test series metadata"
        sid = sorted(series_by_study[uid])[0]
        files = sorted((data / "test_series" / uid / sid).glob("*.dcm"))
        assert files, "No mounted DICOM files for test series"
        dcm = pydicom.dcmread(files[0])
        assert str(dcm.StudyInstanceUID) == uid and str(dcm.SeriesInstanceUID) == sid
        pixels = dcm.pixel_array
        assert pixels.ndim == 2 and pixels.shape == (int(dcm.Rows), int(dcm.Columns))
        decoded += 1
    except Exception as exc:
        failed.append({"study_index": index, "error_type": type(exc).__name__})

output = Path("/kaggle/working/submission.csv")
with output.open("w", newline="", encoding="utf-8") as handle:
    writer = csv.writer(handle)
    writer.writerow(sample_columns)
    for row in test:
        writer.writerow([row[uid_column]] + [0.5] * 12)

with output.open(newline="", encoding="utf-8") as handle:
    reader = csv.DictReader(handle)
    assert list(reader.fieldnames or []) == sample_columns
    predictions = list(reader)
assert len(predictions) == len(test)
assert [row[uid_column] for row in predictions] == [row[uid_column] for row in test]
assert all(math.isfinite(float(row[column])) and 0 <= float(row[column]) <= 1 for row in predictions for column in sample_columns[1:])

summary = {
    "scope": "Offline I/O and one-DICOM-per-visible-study decode smoke; constant outputs are not model predictions",
    "visible_test_studies": len(test),
    "visible_test_series": len(series),
    "studies_with_first_series_decoded": decoded,
    "failed_studies": len(failed),
    "failure_types": failed,
    "submission_columns": sample_columns,
    "submission_rows": len(predictions),
    "internet_disabled_by_metadata": True,
    "gpu_enabled_by_metadata": False,
    "elapsed_seconds": round(time.perf_counter() - start, 2),
    "pydicom_version": pydicom.__version__,
    "jpeg_lossless_decoder_plugins": list(get_decoder(JPEGLosslessSV1).available_plugins),
    "jpeg2000_lossless_decoder_plugins": list(get_decoder(JPEG2000Lossless).available_plugins),
    "llama_cpp_python_installed": importlib.util.find_spec("llama_cpp") is not None,
    "llama_cli_executable": shutil.which("llama-cli"),
    "test_first_instance_transfer_syntaxes": dict(test_transfer_syntaxes),
    "test_first_instance_header_failures": dict(test_first_instance_failures),
}
Path("/kaggle/working/smoke_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
print(json.dumps({k: summary[k] for k in ("visible_test_studies", "studies_with_first_series_decoded", "failed_studies", "elapsed_seconds")}))
