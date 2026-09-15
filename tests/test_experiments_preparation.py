"""Synthetic fresh01 contracts: no competition data, training, or score claims."""
import json

import numpy as np
import pandas as pd
import pytest

from rsna_experiments.common import UID, TARGETS, YC, WC, fingerprint, sha256
from rsna_experiments.preparation import prepare


@pytest.fixture
def inputs(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    records = []
    for i in range(18):
        row = {UID: f"{i:03d}", "group": f"g{i // 2:02d}", "fold": (i // 2) % 3,
               "report": "private source text must not enter training"}
        for j, (yc, wc) in enumerate(zip(YC, WC)):
            row[yc] = float((i + j) % 2)
            row[wc] = 0.0 if j == 11 or (i + j) % 4 == 0 else 1.0
        records.append(row)
    pd.DataFrame(records).to_csv(source / "fresh_manifest.csv", index=False)
    config = {"manifest_path": str(source / "fresh_manifest.csv"),
              "out_dir": str(tmp_path / "prepared"), "max_train": 0, "max_val": 0}
    refresh_metadata(config)
    return config


def refresh_metadata(config, **extra):
    from pathlib import Path
    path = Path(config["manifest_path"])
    metadata = {"targets": list(TARGETS), "n_folds": 3,
                "group_method": "normalized_report_hash_fallback", "manifest_sha256": sha256(path)}
    metadata.update(extra)
    path.with_name("fresh_preparation.json").write_text(json.dumps(metadata), encoding="utf-8")


def manifest(config):
    return pd.read_csv(config["manifest_path"], dtype={UID: str, "group": str})


def save_source(config, frame):
    frame.to_csv(config["manifest_path"], index=False)
    refresh_metadata(config)


def output(config):
    out = prepare(config)
    return pd.read_csv(out / "experiment_manifest.csv", dtype={UID: str, "group": str}), json.loads((out / "preparation_meta.json").read_text())


def audit_file(config):
    from pathlib import Path
    frame = manifest(config)
    audit = frame[[UID]].copy()
    audit["report"] = [f"unique synthetic report {i}" for i in range(len(frame))]
    for target in TARGETS:
        audit[target] = "unmentioned"
    path = Path(config["manifest_path"]).with_name("label_audit.csv")
    audit.to_csv(path, index=False)
    return path, audit


def corrections_file(config, tmp_path, **overrides):
    row = {UID: "002", "target": TARGETS[0], "label": .25, "weight": .8,
           "reviewer": "synthetic reviewer", "reason": "synthetic manual review"}
    row.update(overrides)
    path = tmp_path / "corrections.csv"
    pd.DataFrame([row]).to_csv(path, index=False)
    config["corrections_path"] = str(path)
    return path


def test_preserves_frozen_source_and_provenance(inputs):
    before = manifest(inputs)
    selected, meta = output(inputs)
    pd.testing.assert_frame_equal(selected, before[selected.columns], check_dtype=False)
    assert "report" not in selected
    assert meta["schema"] == "rsna-experiments-preparation-v1"
    assert meta["manifest_sha256"] == sha256(prepare(inputs) / "experiment_manifest.csv")
    assert meta["cohort_id"] == fingerprint(selected[[UID, "group", "fold"]].to_dict("records"))
    assert meta["supervision_id"] == fingerprint(selected[[UID] + list(YC) + list(WC)].to_dict("records"))
    assert meta["patient_grouping"] == "unknown" and meta["exploratory"]
    assert meta["readiness"] == "manual_review_required"
    assert meta["source_provenance"]["manifest_hash_verified"]
    assert meta["train_rows"] == 12 and meta["validation_rows"] == 6
    assert not meta["audit"]["available"]


def test_deterministic_selection_ignores_order_and_labels(inputs, tmp_path):
    inputs.update(max_train=5, max_val=3)
    first, first_meta = output(inputs)
    frame = manifest(inputs).sample(frac=1, random_state=100)
    frame[list(YC)] = 1 - frame[list(YC)]
    save_source(inputs, frame)
    inputs["out_dir"] = str(tmp_path / "second")
    second, second_meta = output(inputs)
    pd.testing.assert_frame_equal(first[[UID, "group", "fold"]], second[[UID, "group", "fold"]])
    assert first_meta["cohort_id"] == second_meta["cohort_id"]
    assert first_meta["supervision_id"] != second_meta["supervision_id"]
    assert len(second) == 8
    assert not set(second.loc[second.fold.eq(0), "group"]) & set(second.loc[second.fold.ne(0), "group"])


@pytest.mark.parametrize("fault,match", [
    ("duplicate", "duplicate UID"), ("group_null", "null/empty group"),
    ("fold_null", "null/empty fold"), ("uid_null", "null/empty StudyInstanceUID"),
    ("fold_fraction", "source fold"), ("fold_negative", "source fold"),
    ("fold_range", "outside source"), ("overlap", "grouping overlap"),
    ("nan", "finite"), ("inf", "finite"), ("weight", "within"), ("label", "within"),
])
def test_rejects_invalid_source_before_output(inputs, fault, match, tmp_path):
    frame = manifest(inputs)
    if fault == "duplicate":
        frame = pd.concat([frame, frame.iloc[[0]]])
    elif fault in {"group_null", "fold_null", "uid_null"}:
        col = {"group_null": "group", "fold_null": "fold", "uid_null": UID}[fault]
        frame[col] = frame[col].astype(object)
        frame.loc[0, col] = ""
    elif fault.startswith("fold_"):
        frame["fold"] = frame["fold"].astype(float)
        frame.loc[0, "fold"] = {"fold_fraction": .5, "fold_negative": -1, "fold_range": 3}[fault]
    elif fault == "overlap":
        frame.loc[2, "group"] = frame.loc[0, "group"]
    else:
        frame.loc[0, WC[0] if fault == "weight" else YC[0]] = {"nan": np.nan, "inf": np.inf, "weight": -1, "label": 2}[fault]
    save_source(inputs, frame)
    with pytest.raises(ValueError, match=match):
        prepare(inputs)
    assert not (tmp_path / "prepared").exists()


def test_rejects_source_tampering_and_target_order(inputs):
    frame = manifest(inputs)
    frame.loc[0, YC[0]] = .123
    frame.to_csv(inputs["manifest_path"], index=False)
    with pytest.raises(ValueError, match="SHA256"):
        prepare(inputs)
    refresh_metadata(inputs, targets=list(reversed(TARGETS)))
    with pytest.raises(ValueError, match="targets/order"):
        prepare(inputs)


def test_counts_and_training_only_prior_bce(inputs):
    selected, meta = output(inputs)
    for row in meta["target_fold_counts"]:
        assert row["positive"] + row["negative"] + row["unknown"] + row["soft"] == row["n"]
    yc, wc = YC[0], WC[0]
    train, val = selected.loc[selected.fold.ne(0)], selected.loc[selected.fold.eq(0)]
    expected = (train[yc] * train[wc]).sum() / train[wc].sum()
    diagnostic = meta["constant_prediction_comparison"]["targets"][0]
    assert diagnostic["training_prior"] == pytest.approx(expected)
    loss = -(val[yc] * np.log(expected) + (1 - val[yc]) * np.log1p(-expected))
    assert diagnostic["validation_bce"] == pytest.approx((loss * val[wc]).sum() / val[wc].sum())
    unknown = meta["constant_prediction_comparison"]["targets"][11]
    assert unknown["training_prior"] is None and unknown["validation_bce"] is None


def test_explicit_training_correction_keeps_validation_and_cohort(inputs, tmp_path):
    original, before = output(inputs)
    inputs["out_dir"] = str(tmp_path / "corrected")
    corrections_file(inputs, tmp_path)
    corrected, after = output(inputs)
    pd.testing.assert_frame_equal(original.loc[original.fold.eq(0)], corrected.loc[corrected.fold.eq(0)])
    assert corrected.set_index(UID).at["002", YC[0]] == .25
    assert corrected.set_index(UID).at["002", WC[0]] == .8
    assert after["cohort_id"] == before["cohort_id"]
    assert after["supervision_id"] != before["supervision_id"]
    assert after["corrections"][0]["reviewer"] == "synthetic reviewer"
    assert after["validation_labels_preserved"]
    assert any(row["soft"] for row in after["target_fold_counts"])


@pytest.mark.parametrize("overrides,match", [
    ({UID: "000"}, "validation edits"), ({UID: "nonexistent"}, "invalid UID"),
    ({"target": "not a target"}, "invalid target"), ({"label": 1.1}, "within"),
    ({"weight": float("inf")}, "finite"), ({"reviewer": " "}, "reviewer"),
    ({"reason": ""}, "reason"),
])
def test_rejects_invalid_corrections(inputs, tmp_path, overrides, match):
    corrections_file(inputs, tmp_path, **overrides)
    with pytest.raises(ValueError, match=match):
        prepare(inputs)


def test_rejects_duplicate_corrections(inputs, tmp_path):
    path = corrections_file(inputs, tmp_path)
    correction = pd.read_csv(path, dtype={UID: str})
    pd.concat([correction, correction]).to_csv(path, index=False)
    with pytest.raises(ValueError, match="duplicate UID/target"):
        prepare(inputs)


def test_no_silent_ignored_correction_outside_cohort(inputs, tmp_path):
    inputs["max_train"] = 1
    selected, _ = output(inputs)
    uid = next(u for u in manifest(inputs).loc[lambda f: f.fold.ne(0), UID] if u not in set(selected[UID]))
    inputs["out_dir"] = str(tmp_path / "corrected")
    corrections_file(inputs, tmp_path, **{UID: uid})
    with pytest.raises(ValueError, match="outside selected"):
        prepare(inputs)


def test_normalized_report_leakage_checked_before_cohort_sampling(inputs):
    path, audit = audit_file(inputs)
    audit.loc[0, "report"] = "  DUPLICATE \n synthetic Report "
    audit.loc[2, "report"] = "duplicate synthetic report"
    audit.to_csv(path, index=False)
    inputs.update(max_train=1, max_val=1)
    with pytest.raises(ValueError, match="collision across training/validation"):
        prepare(inputs)


def test_same_side_collisions_reported_and_sample_bounded(inputs):
    path, audit = audit_file(inputs)
    audit.loc[2, "report"] = audit.loc[4, "report"] = "same training report"
    audit.loc[0, "report"] = audit.loc[6, "report"] = ""  # Empty reports do not imply linkage.
    audit.loc[1, "report"] = "x" * 3000
    audit.to_csv(path, index=False)
    inputs["audit_sample_per_status"] = 2
    _, meta = output(inputs)
    assert meta["audit"]["cross_group_collision_count"] == 1
    assert meta["audit"]["empty_report_count"] == 2
    samples = pd.read_csv(prepare(inputs) / "target_status_audit_sample.csv", keep_default_na=False)
    assert samples.groupby(["target", "status", "source_status"]).size().max() <= 2
    assert samples.report.str.len().max() <= 2000
    assert samples.manual_review_required.all()
    assert len(samples) <= 12 * 4 * 5 * 2


@pytest.mark.parametrize("fault,match", [("missing", "missing .* source UIDs"), ("duplicate", "duplicate UID"), ("status", "invalid target status")])
def test_invalid_audit_fails_closed(inputs, fault, match):
    path, audit = audit_file(inputs)
    if fault == "missing":
        audit = audit.iloc[1:]
    elif fault == "duplicate":
        audit = pd.concat([audit, audit.iloc[[0]]])
    else:
        audit.loc[0, TARGETS[0]] = "arbitrary report text"
    audit.to_csv(path, index=False)
    with pytest.raises(ValueError, match=match):
        prepare(inputs)


def test_output_identity_rejects_changed_run(inputs):
    prepare(inputs)
    inputs["seed"] = 99
    with pytest.raises((ValueError, FileExistsError)):
        prepare(inputs)


def test_nonempty_unknown_output_preserved(inputs, tmp_path):
    out = tmp_path / "prepared"
    out.mkdir()
    existing = out / "user-data.txt"
    existing.write_text("preserve me")
    with pytest.raises(ValueError, match="nonempty"):
        prepare(inputs)
    assert existing.read_text() == "preserve me"
    assert not (out / "experiment_manifest.csv").exists()


def test_prior_does_not_fit_validation_labels(inputs, tmp_path):
    _, before = output(inputs)
    frame = manifest(inputs)
    frame.loc[frame.fold.eq(0), list(YC)] = .37
    save_source(inputs, frame)
    inputs["out_dir"] = str(tmp_path / "new_validation")
    _, after = output(inputs)
    first = before["constant_prediction_comparison"]["targets"]
    second = after["constant_prediction_comparison"]["targets"]
    assert [t["training_prior"] for t in first] == [t["training_prior"] for t in second]
    assert [t["train_bce"] for t in first] == [t["train_bce"] for t in second]


def test_patient_identifier_is_provenance_not_invented_linkage(inputs):
    refresh_metadata(inputs, group_method="PatientID")
    source = manifest(inputs)
    selected, meta = output(inputs)
    assert selected["group"].tolist() == source["group"].tolist()
    assert meta["grouping_method"] == "PatientID"
    assert meta["patient_grouping"] == "source_patient_identifier_unverified"
    assert meta["exploratory"]


def test_explicit_missing_audit_is_not_silently_skipped(inputs, tmp_path):
    inputs["label_audit_path"] = str(tmp_path / "missing_audit.csv")
    with pytest.raises(FileNotFoundError):
        prepare(inputs)


def test_audit_can_include_studies_without_images(inputs):
    path, audit = audit_file(inputs)
    extra = audit.iloc[[0]].copy()
    extra[UID] = "study excluded by fresh01 because no series exists"
    pd.concat([audit, extra]).to_csv(path, index=False)
    selected, meta = output(inputs)
    assert len(selected) == 18
    assert meta["audit"]["available"]


def test_integer_valued_csv_floats_preserve_folds(inputs):
    frame = manifest(inputs)
    frame["fold"] = frame["fold"].astype(float)
    save_source(inputs, frame)
    selected, _ = output(inputs)
    assert selected["fold"].tolist() == frame["fold"].tolist()


@pytest.mark.parametrize("key,value", [("fold", -1), ("fold", 8), ("max_train", -.5), ("max_val", 2.5), ("seed", True), ("audit_sample_per_status", 21)])
def test_invalid_configuration(inputs, key, value):
    inputs[key] = value
    with pytest.raises(ValueError):
        prepare(inputs)
