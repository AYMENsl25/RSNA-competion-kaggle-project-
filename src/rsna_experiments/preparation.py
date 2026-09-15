"""Freeze an exploratory cohort without changing fresh01 folds or validation labels.

``prepare(config)`` returns the output directory. Required keys: manifest_path,
out_dir. Optional: preparation_path (defaults to the manifest's sibling JSON),
label_audit_path (auto-detected sibling CSV), corrections_path, fold=0,
max_train=512, max_val=240, seed=42, audit_sample_per_status=3 (1..20).
Zero cohort limits mean all rows. Manual audit remains required after preparation.
"""
from __future__ import annotations

import json
from pathlib import Path
import re

import numpy as np
import pandas as pd

from .common import UID, TARGETS, YC, WC, fingerprint, locate, safe_output, sha256, write_json


SCHEMA = "rsna-experiments-preparation-v1"
IDENTIFIERS = [UID, "group", "fold"]


def _integer(value, name, minimum=0):
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer >= {minimum}")
    try:
        result = int(value)
        if float(value) != result or result < minimum:
            raise ValueError
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be an integer >= {minimum}") from exc
    return result


def _required(frame, columns, context):
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f"{context}: missing columns {missing}")


def _nonempty(frame, columns, context):
    for col in columns:
        if frame[col].isna().any() or frame[col].astype(str).str.strip().eq("").any():
            raise ValueError(f"{context}: null/empty {col}")
        if not frame[col].astype(str).eq(frame[col].astype(str).str.strip()).all():
            raise ValueError(f"{context}: surrounding whitespace in {col}")


def _unit_values(frame, columns, context):
    try:
        values = frame[columns].apply(pd.to_numeric, errors="raise").to_numpy(dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{context}: numeric values required") from exc
    if not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
        raise ValueError(f"{context}: values must be finite and within [0,1]")
    frame[columns] = values


def _validate_manifest(frame, metadata, fold):
    _required(frame, IDENTIFIERS + list(YC) + list(WC), "manifest")
    if frame.empty:
        raise ValueError("manifest is empty")
    _nonempty(frame, [UID, "group", "fold"], "manifest")
    if frame[UID].duplicated().any():
        raise ValueError("manifest: duplicate UID")
    try:
        numeric_folds = pd.to_numeric(frame["fold"], errors="raise")
    except (TypeError, ValueError) as exc:
        raise ValueError("source fold must be an integer >= 0") from exc
    frame["fold"] = [_integer(v, "source fold") for v in numeric_folds]
    if metadata.get("n_folds") is not None:
        n_folds = _integer(metadata["n_folds"], "n_folds", 2)
        if (frame["fold"] >= n_folds).any() or fold >= n_folds:
            raise ValueError("fold outside source n_folds")
    if frame.groupby("group")["fold"].nunique().gt(1).any():
        raise ValueError("grouping overlap: a frozen group occurs in multiple folds")
    if not frame["fold"].eq(fold).any() or not frame["fold"].ne(fold).any():
        raise ValueError("requested fold must have both training and validation rows")
    _unit_values(frame, list(YC) + list(WC), "manifest y/w")


def _select(frame, limit, seed):
    """Rank identifiers only; labels, row order and report status never select rows."""
    ranked = frame.copy()
    ranked["_rank"] = [fingerprint({"seed": seed, UID: u, "group": g, "fold": int(f)})
                       for u, g, f in ranked[IDENTIFIERS].itertuples(index=False, name=None)]
    ranked = ranked.sort_values(["_rank", UID])
    return (ranked.head(limit) if limit else ranked).drop(columns="_rank")


def _counts(frame):
    rows = []
    for fold, subset in frame.groupby("fold", sort=True):
        for target, ycol, wcol in zip(TARGETS, YC, WC):
            known = subset[wcol].gt(0)
            rows.append({"fold": int(fold), "target": target, "n": len(subset),
                         "positive": int((known & subset[ycol].eq(1)).sum()),
                         "negative": int((known & subset[ycol].eq(0)).sum()),
                         "unknown": int((~known).sum()),
                         "soft": int((known & subset[ycol].between(0, 1, inclusive="neither")).sum()),
                         "weight_sum": float(subset[wcol].sum())})
    return rows


def _prior_comparison(frame, fold):
    """Training-only weighted priors; undefined targets remain explicit nulls."""
    train = frame.loc[frame["fold"].ne(fold)]
    result = []
    for target, ycol, wcol in zip(TARGETS, YC, WC):
        weight = float(train[wcol].sum())
        prior = float((train[ycol] * train[wcol]).sum() / weight) if weight else None
        row = {"target": target, "training_weight": weight, "training_prior": prior}
        for name, mask in (("train", frame["fold"].ne(fold)), ("validation", frame["fold"].eq(fold))):
            subset = frame.loc[mask]
            denom = float(subset[wcol].sum())
            if prior is None or not denom:
                bce = None
            else:
                p = np.clip(prior, 1e-7, 1 - 1e-7)
                losses = -subset[ycol] * np.log(p) - (1 - subset[ycol]) * np.log1p(-p)
                bce = float((losses * subset[wcol]).sum() / denom)
            row[f"{name}_bce"] = bce
        result.append(row)
    return {"purpose": "constant-prediction diagnostic, no scoring claim",
            "prior_scope": "selected training rows after explicit corrections",
            "bce": "per-target sum(weight * soft BCE) / sum(weight)",
            "clip_epsilon": 1e-7, "targets": result}


def _audit(path, source, selected, fold, seed, sample_limit):
    columns = [UID, "group", "fold", "target", "status", "source_status", "label", "weight",
               "report", "report_truncated", "manual_review_required"]
    audit = None
    summary = {"available": path is not None, "cross_group_collision_count": 0,
               "cross_group_collisions": [], "report_normalization": "lowercase, collapse whitespace, strip",
               "manual_review_required": True}
    if path is not None:
        audit = pd.read_csv(path, dtype=str, keep_default_na=False)
        _required(audit, [UID, "report"] + list(TARGETS), "label audit")
        _nonempty(audit, [UID], "label audit")
        if audit[UID].duplicated().any():
            raise ValueError("label audit: duplicate UID")
        statuses = {"positive", "negative", "uncertain", "unmentioned", "conflict"}
        if not audit[list(TARGETS)].isin(statuses).all().all():
            raise ValueError("label audit: invalid target status")
        missing = set(source[UID]) - set(audit[UID])
        if missing:
            raise ValueError(f"label audit: missing {len(missing)} source UIDs")
        joined = source[IDENTIFIERS].merge(audit[[UID, "report"]], on=UID, validate="one_to_one")
        joined["_normalized"] = joined["report"].map(lambda s: re.sub(r"\s+", " ", s.lower()).strip())
        summary["empty_report_count"] = int(joined["_normalized"].eq("").sum())
        # Empty reports contain no evidence and must not imply shared patient identity.
        collisions = []
        for report, group in joined.loc[joined["_normalized"].ne("")].groupby("_normalized", sort=True):
            if group["group"].nunique() <= 1:
                continue
            if group["fold"].eq(fold).any() and group["fold"].ne(fold).any():
                raise ValueError("normalized report collision across training/validation groups")
            collisions.append({"report_fingerprint": fingerprint(report), "study_count": len(group),
                               "group_count": int(group["group"].nunique()),
                               "folds": sorted(int(f) for f in group["fold"].unique())})
        summary["cross_group_collision_count"] = len(collisions)
        summary["cross_group_collisions"] = collisions
        audit = audit.set_index(UID)
    samples = []
    ordered = _select(selected, 0, seed)
    for target, ycol, wcol in zip(TARGETS, YC, WC):
        used = {}
        for record in ordered.to_dict("records"):
            y, w, uid = record[ycol], record[wcol], record[UID]
            status = "unknown" if w == 0 else "positive" if y == 1 else "negative" if y == 0 else "soft"
            source_status = str(audit.at[uid, target]) if audit is not None else "unavailable"
            key = (status, source_status)
            if used.get(key, 0) >= sample_limit:
                continue
            used[key] = used.get(key, 0) + 1
            report = str(audit.at[uid, "report"]) if audit is not None else ""
            samples.append({UID: uid, "group": record["group"], "fold": int(record["fold"]),
                            "target": target, "status": status, "source_status": source_status,
                            "label": y, "weight": w, "report": report[:2000],
                            "report_truncated": len(report) > 2000, "manual_review_required": True})
    summary["sample_rows"] = len(samples)
    summary["sample_per_target_status"] = sample_limit
    summary["sample_report_character_limit"] = 2000
    return summary, pd.DataFrame(samples, columns=columns)


def _correct(frame, source, path, fold):
    if path is None:
        return frame, []
    corrections = pd.read_csv(path, dtype=str, keep_default_na=False)
    fields = [UID, "target", "label", "weight", "reviewer", "reason"]
    _required(corrections, fields, "corrections")
    _nonempty(corrections, [UID, "target", "reviewer", "reason"], "corrections")
    if corrections.duplicated([UID, "target"]).any():
        raise ValueError("corrections: duplicate UID/target")
    _unit_values(corrections, ["label", "weight"], "corrections")
    if not set(corrections["target"]).issubset(TARGETS):
        raise ValueError("corrections: invalid target")
    if not set(corrections[UID]).issubset(set(source[UID])):
        raise ValueError("corrections: invalid UID")
    if set(corrections[UID]) & set(source.loc[source["fold"].eq(fold), UID]):
        raise ValueError("corrections: validation edits are forbidden")
    if not set(corrections[UID]).issubset(set(frame[UID])):
        raise ValueError("corrections: UID outside selected training cohort")
    indexed = frame.set_index(UID).copy()
    changes = []
    for item in corrections[fields].sort_values([UID, "target"]).to_dict("records"):
        i = list(TARGETS).index(item["target"])
        uid = item[UID]
        changes.append({**item, "previous_label": float(indexed.at[uid, YC[i]]),
                        "previous_weight": float(indexed.at[uid, WC[i]])})
        indexed.at[uid, YC[i]] = item["label"]
        indexed.at[uid, WC[i]] = item["weight"]
    return indexed.reset_index()[frame.columns], changes


def prepare(config) -> Path:
    """Validate fresh01 provenance and write a fixed, auditable experiment cohort."""
    if not config.get("manifest_path") or not config.get("out_dir"):
        raise ValueError("manifest_path and out_dir are required")
    fold = _integer(config.get("fold", 0), "fold")
    seed = _integer(config.get("seed", 42), "seed")
    max_train = _integer(config.get("max_train", 512), "max_train")
    max_val = _integer(config.get("max_val", 240), "max_val")
    sample_limit = _integer(config.get("audit_sample_per_status", 3), "audit_sample_per_status", 1)
    if sample_limit > 20:
        raise ValueError("audit_sample_per_status must be <= 20")
    manifest_path = Path(locate("fresh_manifest.csv", override=config["manifest_path"])).resolve()
    metadata_path = Path(config.get("preparation_path") or manifest_path.with_name("fresh_preparation.json")).resolve()
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if not isinstance(metadata, dict) or metadata.get("targets") != list(TARGETS):
        raise ValueError("fresh preparation targets/order do not match common.TARGETS")
    source_hash = sha256(manifest_path)
    if metadata.get("manifest_sha256") != source_hash:
        raise ValueError("source manifest SHA256 does not match fresh_preparation metadata")
    source = pd.read_csv(manifest_path, dtype=str, keep_default_na=False)
    _validate_manifest(source, metadata, fold)
    # Allowlist: arbitrary input columns (especially report text) never enter training.
    source = source[IDENTIFIERS + list(YC) + list(WC)].copy()
    selected = pd.concat([_select(source.loc[source["fold"].ne(fold)], max_train, seed),
                          _select(source.loc[source["fold"].eq(fold)], max_val, seed)])
    selected = selected.sort_values(IDENTIFIERS).reset_index(drop=True)
    cohort_id = fingerprint(selected[IDENTIFIERS].to_dict("records"))
    corrections_path = Path(config["corrections_path"]).resolve() if config.get("corrections_path") else None
    selected, changes = _correct(selected, source, corrections_path, fold)
    validation = source.set_index(UID).loc[selected.loc[selected["fold"].eq(fold), UID], list(YC) + list(WC)]
    if not validation.equals(selected.loc[selected["fold"].eq(fold)].set_index(UID)[list(YC) + list(WC)]):
        raise ValueError("validation supervision changed")
    supervision_id = fingerprint(selected[[UID] + list(YC) + list(WC)].to_dict("records"))
    audit_override = config.get("label_audit_path") or config.get("audit_path")
    audit_path = Path(audit_override).resolve() if audit_override else manifest_path.with_name("label_audit.csv")
    if not audit_override and not audit_path.exists():
        audit_path = None
    audit_summary, samples = _audit(audit_path, source, selected, fold, seed, sample_limit)
    grouping_method = metadata.get("group_method", metadata.get("grouping_method", "unknown"))
    if not isinstance(grouping_method, str) or not grouping_method.strip():
        grouping_method = "unknown"
    patient_evidence = grouping_method in {"PatientID", "patient_id", "PatientId"}
    provenance = {"manifest": {"path": str(manifest_path), "sha256": source_hash},
                  "preparation": {"path": str(metadata_path), "sha256": sha256(metadata_path)},
                  "label_audit": {"path": str(audit_path), "sha256": sha256(audit_path)} if audit_path else None,
                  "corrections": {"path": str(corrections_path), "sha256": sha256(corrections_path)} if corrections_path else None,
                  "manifest_hash_verified": True}
    identity = {"schema": SCHEMA, "cohort_id": cohort_id, "supervision_id": supervision_id,
                "fold": fold, "seed": seed, "max_train": max_train, "max_val": max_val,
                "audit_sample_per_status": sample_limit, "source_provenance": provenance}
    out = Path(config["out_dir"]).resolve()
    # Never overwrite source inputs, even if safe_output permits an identity reuse.
    outputs = {out / name for name in ("experiment_manifest.csv", "preparation_meta.json", "run_identity.json", "target_status_audit_sample.csv")}
    if outputs & {p for p in (manifest_path, metadata_path, audit_path, corrections_path) if p is not None}:
        raise ValueError("output would overwrite a source input")
    safe_output(out, identity)
    output_manifest = out / "experiment_manifest.csv"
    selected.to_csv(output_manifest, index=False)
    samples.to_csv(out / "target_status_audit_sample.csv", index=False)
    result = {"schema": SCHEMA, "cohort_id": cohort_id, "supervision_id": supervision_id,
              "fold": fold, "manifest_sha256": sha256(output_manifest), "grouping_method": grouping_method,
              "targets": list(TARGETS), "source_provenance": provenance,
              "patient_grouping": "source_patient_identifier_unverified" if patient_evidence else "unknown",
              "exploratory": True, "readiness": "manual_review_required", "manual_review_required": True,
              "source_rows": len(source), "train_rows": int(selected["fold"].ne(fold).sum()),
              "validation_rows": int(selected["fold"].eq(fold).sum()),
              "selection": {"method": "seeded identifier fingerprint rank; no label stratification",
                            "seed": seed, "max_train": max_train, "max_val": max_val},
              "source_target_fold_counts": _counts(source), "target_fold_counts": _counts(selected),
              "constant_prediction_comparison": _prior_comparison(selected, fold),
              "audit": audit_summary, "corrections": changes, "validation_labels_preserved": True,
              "warnings": ["Heuristic supervision requires manual target/status review."]}
    if not patient_evidence:
        result["warnings"].append("Patient grouping unknown: exploratory only; report grouping does not establish patient independence.")
    if audit_path is None:
        result["warnings"].append("No report audit supplied; normalized-report leakage audit unavailable.")
    write_json(out / "preparation_meta.json", result)
    return out
