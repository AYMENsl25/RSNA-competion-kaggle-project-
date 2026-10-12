"""Build an auditable candidate only; no expert values are exported or used to balance folds."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

TARGETS = ("ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA",
           "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture")
UID = "StudyInstanceUID"


def sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def build(train: Path, patients: Path, output: Path, summary: Path,
          seed: str, validation_fraction: float, near_edges: Path,
          image_triage: Path | None = None, image_pairs: Path | None = None,
          image_decisions: Path | None = None, image_audit_summary: Path | None = None,
          n01_assertions: Path | None = None) -> dict:
    if not 0 < validation_fraction < 1:
        raise ValueError("Validation fraction must be between zero and one")
    rows = read_csv(train)
    patient_rows = read_csv(patients)
    by_uid = {row[UID]: row for row in rows}
    patient_by_uid = {row[UID]: row for row in patient_rows}
    if len(by_uid) != len(rows) or len(patient_by_uid) != len(patient_rows):
        raise ValueError("Duplicate UIDs in inputs")
    if set(by_uid) != set(patient_by_uid):
        raise ValueError("Patient census and official train UID sets differ")
    # Read labels only to identify fully labeled rows, never retain their values.
    experts = {row[UID] for row in rows if all(row[target] in {"0", "1"} for target in TARGETS)}
    if len(experts) != 58:
        raise ValueError("Expected 58 fully labeled expert studies")
    parents = {uid: uid for uid in by_uid}

    def root(uid: str) -> str:
        while parents[uid] != uid:
            parents[uid] = parents[parents[uid]]
            uid = parents[uid]
        return uid

    def join(left: str, right: str) -> None:
        left, right = root(left), root(right)
        parents[max(left, right)] = min(left, right)

    report_first: dict[str, str] = {}
    patient_first: dict[str, str] = {}
    report_hashes = {}
    flagged = set()
    for uid in sorted(by_uid):
        report = " ".join(by_uid[uid]["Report"].casefold().split())
        if not report:
            raise ValueError("Empty reports require an explicit grouping policy")
        report_hashes[uid] = key = sha(report)
        if key in report_first:
            join(uid, report_first[key])
        else:
            report_first[key] = uid
        patient = patient_by_uid[uid]["patient_id_sha256"].strip()
        if len(patient) != 64 or any(c not in "0123456789abcdef" for c in patient):
            raise ValueError("Missing or malformed patient hash")
        if patient in patient_first:
            join(uid, patient_first[patient])
        else:
            patient_first[patient] = uid
        flag = patient_by_uid[uid]["expert_or_linked_excluded"]
        if flag not in {"0", "1"}:
            raise ValueError("Malformed exclusion flag")
        if flag == "1":
            flagged.add(uid)
    expert_reports = {report_hashes[uid] for uid in experts}
    linked = {uid for uid in by_uid if report_hashes[uid] in expert_reports} - experts
    if len(linked) != 4 or not (experts | linked) <= flagged:
        raise ValueError("Expected all 58 experts and four linked studies protected by census")
    groups: dict[str, list[str]] = defaultdict(list)
    for uid in sorted(by_uid):
        groups[root(uid)].append(uid)
    def assign(members: list[str]) -> str:
        if set(members) & (experts | flagged):
            return "excluded"
        bucket = int(sha(seed + ":" + sha("\n".join(members))), 16) / 2**256
        return "candidate_validation" if bucket < validation_fraction else "candidate_train"

    before = {uid: assign(members) for members in groups.values() for uid in members}
    edge_metadata = json.loads(near_edges.with_suffix(".audit.json").read_text(encoding="utf-8"))
    if edge_metadata["source_train_sha256"] != file_sha(train) or edge_metadata["edges_sha256"] != file_sha(near_edges):
        raise ValueError("Near-report source or edge artifact hash mismatch")
    if edge_metadata["threshold"] != 0.90:
        raise ValueError("This candidate policy requires near-report threshold 0.90")
    edges = read_csv(near_edges)
    seen = set()
    for edge in edges:
        left, right = edge["uid_a"], edge["uid_b"]
        if left not in by_uid or right not in by_uid or left >= right or (left, right) in seen:
            raise ValueError("Invalid or duplicate near-report UID edge")
        if not 0.90 <= float(edge["jaccard"]) <= 1:
            raise ValueError("Invalid near-report similarity")
        seen.add((left, right))
        join(left, right)
    if len(edges) != edge_metadata["edges"]:
        raise ValueError("Near-report edge count mismatch")
    pre_cross = sum(before[left] != before[right] for left, right in seen)
    pre_train_validation = sum({before[left], before[right]} == {"candidate_train", "candidate_validation"} for left, right in seen)
    groups = defaultdict(list)
    for uid in sorted(by_uid):
        groups[root(uid)].append(uid)
    image_seen = set()
    dropped_image = set()
    image_expert_component_sizes = []
    image_pre_assignment = {uid: assign(members) for members in groups.values() for uid in members}
    image_sources = (image_triage, image_pairs, image_decisions, image_audit_summary)
    if any(path is not None for path in image_sources):
        if not all(path is not None for path in image_sources):
            raise ValueError("All four image triage/provenance inputs are required")
        reconciled = json.loads(image_audit_summary.read_text(encoding="utf-8"))
        for path in (image_pairs, image_decisions):
            if reconciled["source_sha256"][path.name] != file_sha(path):
                raise ValueError("Image confirmation provenance hash mismatch")
        candidate_rows = read_csv(image_pairs)
        decision_rows = read_csv(image_decisions)
        decision_by_hash = {row["pair_sha256"]: row["decision"] for row in decision_rows}
        if len(decision_by_hash) != len(decision_rows):
            raise ValueError("Duplicate image decision hashes")
        triage_rows = read_csv(image_triage)
        candidates = {tuple(sorted((row["study_uid_a"], row["study_uid_b"]))) for row in candidate_rows}
        triage = {}
        for row in triage_rows:
            pair = tuple(sorted((row["study_uid_a"], row["study_uid_b"])))
            if pair in triage or pair[0] == pair[1] or not set(pair) <= set(by_uid):
                raise ValueError("Invalid image triage UID edge")
            expected = decision_by_hash.get(sha("|".join(pair)), "unprocessed")
            if row["decision"] != expected:
                raise ValueError("Image triage decision differs from private evidence")
            triage[pair] = expected
        if len(candidate_rows) != 272 or len(candidates) != 272 or candidates != set(triage):
            raise ValueError("Image triage must cover all 272 candidate pairs")
        if set(decision_by_hash) - {sha("|".join(pair)) for pair in candidates}:
            raise ValueError("Image decision absent from candidates")
        image_seen = {pair for pair, decision in triage.items() if decision != "sampled_slices_dissimilar"}
        dropped_image = candidates - image_seen
        # Report image-only expert component sizes, then union with report and
        # patient components. Final exclusion propagates through every edge.
        image_neighbors = defaultdict(set)
        for left, right in image_seen:
            image_neighbors[left].add(right)
            image_neighbors[right].add(left)
            join(left, right)
        visited = set()
        for uid in sorted(image_neighbors):
            if uid in visited:
                continue
            component, pending = set(), [uid]
            while pending:
                node = pending.pop()
                if node in component:
                    continue
                component.add(node)
                pending.extend(image_neighbors[node] - component)
            visited.update(component)
            if component & experts:
                image_expert_component_sizes.append(len(component))
        groups = defaultdict(list)
        for uid in sorted(by_uid):
            groups[root(uid)].append(uid)
    manifest = []
    for members in sorted(groups.values()):
        group_id = sha("\n".join(members))
        excluded = bool(set(members) & (experts | flagged))
        partition = assign(members)
        for uid in members:
            reason = "expert_holdout" if uid in experts else ("expert_linked_exact_report" if uid in linked else ("census_or_group_exclusion" if excluded else ""))
            manifest.append({UID: uid, "group_id": group_id,
                             "report_sha256": report_hashes[uid],
                             "patient_id_sha256": patient_by_uid[uid]["patient_id_sha256"],
                             "partition": partition, "exclusion_reason": reason,
                             "expert_labels_access_allowed": "0"})
    manifest.sort(key=lambda row: row[UID])
    assignment = {row[UID]: row["partition"] for row in manifest}
    dev = {uid for uid, split in assignment.items() if split != "excluded"}
    checks = {
        "no_expert_in_development": not bool(experts & dev),
        "all_four_report_linked_uids_excluded": not bool(linked & dev),
        "all_census_flagged_uids_excluded": not bool(flagged & dev),
        "uid_sets_complete_and_unique": len(manifest) == len(by_uid),
        "near_report_edges_partition_disjoint": all(assignment[left] == assignment[right] for left, right in seen),
        "unresolved_image_edges_partition_disjoint": all(assignment[left] == assignment[right] for left, right in image_seen),
    }
    for column in ("group_id", "report_sha256", "patient_id_sha256"):
        partitions = defaultdict(set)
        for row in manifest:
            partitions[row[column]].add(row["partition"])
        checks[column + "_partition_disjoint"] = all(len(splits) == 1 for splits in partitions.values())
    checks["all_manifest_expert_access_prohibited"] = all(row["expert_labels_access_allowed"] == "0" for row in manifest)
    if not all(checks.values()):
        raise AssertionError("Candidate split integrity check failed")
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(manifest[0]))
        writer.writeheader()
        writer.writerows(manifest)
    result = {
        "status": "candidate_only_not_frozen_not_training_authorization",
        "candidate_version": 3 if image_triage is not None else 2,
        "review_gate": "Astra review and remaining label/runtime gates required before any freeze or training",
        "algorithm": "Connected components of exact normalized reports, census patient hashes, detected five-word-shingle Jaccard >=0.90 report edges, and all image edges except sampled_slices_dissimilar; seeded SHA-256 group assignment",
        "seed": seed, "validation_fraction": validation_fraction,
        "source_sha256": {"train_csv": file_sha(train), "patient_groups_csv": file_sha(patients), "near_report_edges_csv": file_sha(near_edges)},
        "manifest_sha256": file_sha(manifest_path), "studies": len(rows),
        "experts": len(experts), "exact_report_expert_linked_nonexperts": len(linked),
        "groups": len(groups), "partition_counts": dict(sorted(Counter(assignment.values()).items())),
        "near_report_reconciliation": {
            "threshold": 0.90, "edges": len(edges),
            "pre_reconciliation_cross_partition_edges": pre_cross,
            "pre_reconciliation_train_validation_cross_edges": pre_train_validation,
            "post_reconciliation_cross_partition_edges": sum(assignment[left] != assignment[right] for left, right in seen),
            "largest_component": max(map(len, groups.values())),
            "nonexpert_excluded_after_component_propagation": len(set(by_uid) - dev - experts),
        },
        "checks": checks,
        "image_reconciliation": {
            "candidate_edges": len(image_seen) + len(dropped_image),
            "conservatively_grouped_edges": len(image_seen),
            "sampled_dissimilar_edges_not_used_for_grouping": len(dropped_image),
            "pre_reconciliation_cross_partition_unresolved_edges": sum(image_pre_assignment[left] != image_pre_assignment[right] for left, right in image_seen),
            "pre_reconciliation_cross_fold_unresolved_edges": sum({image_pre_assignment[left], image_pre_assignment[right]} == {"candidate_train", "candidate_validation"} for left, right in image_seen),
            "post_reconciliation_cross_partition_unresolved_edges": sum(assignment[left] != assignment[right] for left, right in image_seen),
            "post_reconciliation_cross_fold_unresolved_edges": sum({assignment[left], assignment[right]} == {"candidate_train", "candidate_validation"} for left, right in image_seen),
            "image_only_expert_component_sizes": sorted(image_expert_component_sizes),
            "combined_expert_components_larger_than_one": sorted(len(members) for members in groups.values() if len(members) > 1 and set(members) & experts),
            "additional_exclusions_vs_report_patient_candidate": sum(image_pre_assignment[uid] != "excluded" and assignment[uid] == "excluded" for uid in by_uid),
        },
        "per_target_expert_access": {target: {"training": False, "tuning": False, "checkpoint_selection": False, "candidate_audit_label_values": False} for target in TARGETS},
        "limitations": [
            "Detected near-report edges reconciled; rare-shingle candidate generation is not an exhaustive near-report census. Image screen and five-slice comparison are sampled, not exhaustive.",
            "Unresolved image edges are grouped conservatively, not classified as true duplicates. Sampled dissimilarity does not prove patient independence or exclude duplication elsewhere.",
            "Templated reports can link unrelated patients; transitive near-report components may overgroup and distort validation composition.",
            "De-identified PatientID hashes may be study-specific; census disjointness does not prove patient independence.",
            "Candidate N01 positive/negative support was checked after reassignment; only three targets meet provisional validation 20/20 support. Independent reviewed reference support and model-training gates remain open.",
            "Access prohibition is a manifest policy and audit assertion, not an operating-system or model-loader access control.",
        ],
    }
    if image_triage is not None:
        result["source_sha256"].update({"image_triage_csv": file_sha(image_triage), "image_candidate_pairs_csv": file_sha(image_pairs), "private_image_decisions_csv": file_sha(image_decisions), "image_reconciled_summary_json": file_sha(image_audit_summary)})
    sizes_by_partition = defaultdict(list)
    for members in groups.values():
        sizes_by_partition[assign(members)].append(len(members))
    result["fold_group_balance"] = {
        partition: {"studies": sum(sizes), "connected_groups": len(sizes),
                    "largest_group": max(sizes),
                    "kish_effective_groups_under_study_weighting": round(sum(sizes)**2 / sum(size**2 for size in sizes), 3)}
        for partition, sizes in sorted(sizes_by_partition.items())}
    result["largest_components"] = [
        {"group_sha256": sha("\n".join(members)), "studies": len(members), "partition": assign(members)}
        for members in sorted(groups.values(), key=lambda members: (-len(members), members[0]))[:5]]
    result["development_validation_study_fraction"] = sum(split == "candidate_validation" for split in assignment.values()) / len(dev)
    if n01_assertions is not None:
        counts = defaultdict(lambda: defaultdict(Counter))
        assertion_rows = read_csv(n01_assertions)
        assertion_uids = [row[UID] for row in assertion_rows]
        if len(set(assertion_uids)) != len(assertion_uids) or not set(assertion_uids) <= set(by_uid):
            raise ValueError("Invalid N01 candidate UID set")
        seen_assertions = set()
        for row in assertion_rows:
            uid = row[UID]
            if uid not in dev:
                continue
            seen_assertions.add(uid)
            for target in TARGETS:
                state = row[target]
                mask = row[target + "_mask"]
                if mask not in {"0", "1"}:
                    raise ValueError("Invalid N01 mask")
                counts[assignment[uid]][target][state] += 1
                if mask == "1":
                    counts[assignment[uid]][target]["observed_mask_cells"] += 1
                    counts[assignment[uid]][target]["observed_" + state] += 1
        result["n01_coverage_reassigned_to_v3"] = {
            "status": "candidate_assertions_only_not_frozen_or_validated_reference",
            "development_studies_with_existing_assertions": len(seen_assertions),
            "development_studies_missing_existing_assertions": len(dev - seen_assertions),
            "counts": {partition: {target: dict(counter) for target, counter in targets.items()} for partition, targets in sorted(counts.items())},
            "candidate_validation_support_at_least_20_per_class": {
                target: counts["candidate_validation"][target]["observed_positive"] >= 20 and counts["candidate_validation"][target]["observed_negative"] >= 20
                for target in TARGETS},
        }
        result["source_sha256"]["n01_candidate_assertions_csv"] = file_sha(n01_assertions)
    (output / "audit.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    summary.parent.mkdir(parents=True, exist_ok=True)
    summary.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", required=True, type=Path)
    parser.add_argument("--patients", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--summary", required=True, type=Path)
    parser.add_argument("--near-edges", required=True, type=Path)
    parser.add_argument("--image-triage", type=Path)
    parser.add_argument("--image-pairs", type=Path)
    parser.add_argument("--image-decisions", type=Path)
    parser.add_argument("--image-audit-summary", type=Path)
    parser.add_argument("--n01-assertions", type=Path)
    parser.add_argument("--seed", default="rsna-v2-candidate-2026-10-09")
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    args = parser.parse_args()
    result = build(args.train, args.patients, args.output, args.summary, args.seed, args.validation_fraction, args.near_edges, args.image_triage, args.image_pairs, args.image_decisions, args.image_audit_summary, args.n01_assertions)
    print(json.dumps({"status": result["status"], "partition_counts": result["partition_counts"], "checks": result["checks"]}, indent=2))


if __name__ == "__main__":
    main()
