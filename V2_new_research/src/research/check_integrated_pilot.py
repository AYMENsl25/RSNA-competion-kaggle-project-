"""Apply strict post-run engineering gates to the second M00 GPU pilot."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--selection-audit", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    pilot = json.loads(args.summary.read_text(encoding="utf-8"))
    selection = json.loads(args.selection_audit.read_text(encoding="utf-8"))
    phases = pilot.get("phases", {})
    checks = {
        "selected_excluded_zero": selection.get("selected_excluded") == 0,
        "selected_expert_zero": selection.get("selected_expert") == 0,
        "all_phases_recorded": set(phases) == {"train", "val", "infer"},
        "no_reported_failures": not pilot.get("failures_by_exception"),
        "actual_gpu_window_le_one_hour": pilot.get("script_seconds", float("inf")) <= 3600,
        "temporary_checkpoint_measured": pilot.get("checkpoint", {}).get("size_mib", 0) > 0,
    }
    requirements = {"train": (70, 30), "val": (20, 20), "infer": (10, 10)}
    for name, (requested, steps) in requirements.items():
        phase = phases.get(name, {})
        checks[f"{name}_all_requested_seen_and_ok"] = (
            phase.get("studies_requested") == requested and phase.get("studies_seen") == requested
            and phase.get("studies_ok") == requested and phase.get("studies_failed") == 0
        )
        checks[f"{name}_steps_sufficient"] = phase.get("gpu_steps", 0) >= steps
        checks[f"{name}_no_nonfinite"] = phase.get("nonfinite_loss_or_prob") == 0
        checks[f"{name}_not_truncated"] = phase.get("truncated_by_budget") is False
    engineering_sample_pass = all(checks.values())

    # Whole-phase rates include startup/shutdown and are more conservative than
    # the notebook's steady-state projection. Still only a 70/20/10-study sample.
    projections = {"headroom_factor": 1.5, "train_studies_per_epoch": 3457,
                   "validation_studies_per_epoch": 885, "epochs": 8,
                   "assumed_hidden_test_studies": 1300}
    if all(phases.get(n, {}).get("studies_per_hour_wall_incl_startup") for n in requirements):
        train_h = 3457 / phases["train"]["studies_per_hour_wall_incl_startup"]
        val_h = 885 / phases["val"]["studies_per_hour_wall_incl_startup"]
        ckpt_h = pilot.get("checkpoint", {}).get("serialize_seconds", 0) / 3600
        infer_h = 1300 / phases["infer"]["studies_per_hour_wall_incl_startup"]
        projections.update({"m00_eight_epoch_hours_wall_rate_raw": 8 * (train_h + val_h + ckpt_h),
                            "m00_eight_epoch_hours_wall_rate_with_headroom": 1.5 * 8 * (train_h + val_h + ckpt_h),
                            "infer_1300_hours_wall_rate_raw": infer_h,
                            "infer_1300_hours_wall_rate_with_headroom": 1.5 * infer_h,
                            "estimated_training_within_8_gpu_hours": 1.5 * 8 * (train_h + val_h + ckpt_h) <= 8,
                            "estimated_inference_within_9_hours": 1.5 * infer_h <= 9})

    out = {
        "status": "engineering_sample_only_not_full_training_authorization",
        "engineering_sample_pass": engineering_sample_pass,
        "checks": checks,
        "whole_phase_projections": projections,
        "source_sha256": {"pilot_summary": hashlib.sha256(args.summary.read_bytes()).hexdigest(),
                          "selection_audit": hashlib.sha256(args.selection_audit.read_bytes()).hexdigest()},
        "limits": ["Synthetic target masks do not test real weak-label sparsity or all-unobserved batch handling.",
                   "AMP gradient overflow/skipped optimizer steps were not counted by the pilot.",
                   "Pilot caps are cooperative post-batch checks, not hard process timeouts.",
                   "Whole-phase rates from 70/20/10 studies may be unrepresentative; 1.5x is planning headroom, not a confidence bound.",
                   "Patient linkage and report-label quality are separate gates; no AUC or trained candidate resulted."],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"engineering_sample_pass": engineering_sample_pass,
                      "checks_failed": [name for name, ok in checks.items() if not ok],
                      "projections": projections}, indent=2))


if __name__ == "__main__":
    main()
