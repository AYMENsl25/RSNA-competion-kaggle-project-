"""Audit near reports; optionally export UID edges locally, never report text."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path


TARGETS = (
    "ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA",
    "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's",
    "Contusion", "Fracture",
)


def shingles(text: str) -> set[tuple[str, ...]]:
    words = re.findall(r"\w+", text.casefold())
    if len(words) < 5:
        return {tuple(words)}
    return {tuple(words[i:i + 5]) for i in range(len(words) - 4)}


def audit(path: Path, edges_output: Path | None = None, edge_threshold: float = 0.90) -> dict:
    if not 0 < edge_threshold <= 1:
        raise ValueError("Edge threshold must be in (0, 1]")
    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    expert = {i for i, row in enumerate(rows) if all(row[t] in {"0", "1"} for t in TARGETS)}
    sets = [shingles(row["Report"]) for row in rows]
    postings: dict[tuple[str, ...], list[int]] = defaultdict(list)
    for index, grams in enumerate(sets):
        for gram in grams:
            postings[gram].append(index)
    shared = Counter()
    skipped_common = 0
    for indices in postings.values():
        if len(indices) > 20:
            skipped_common += 1
            continue
        for a, b in combinations(indices, 2):
            shared[(a, b)] += 1

    # Frequent report templates can have only high-frequency shingles and be
    # missed by the rare-shingle index. Always include exact normalized pairs.
    exact_groups: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        exact_groups[" ".join(row["Report"].casefold().split())].append(index)
    forced_exact_pairs = 0
    for indices in exact_groups.values():
        for pair in combinations(indices, 2):
            if pair not in shared:
                forced_exact_pairs += 1
            shared[pair] = max(shared[pair], 3)

    counts = Counter()
    linked_to_expert = {0.90: set(), 0.95: set()}
    edges = []
    for (a, b), overlap in shared.items():
        if overlap < 3:
            continue
        # Recompute the exact overlap because very common shingles were skipped
        # only during candidate generation.
        intersect = len(sets[a] & sets[b])
        union = len(sets[a]) + len(sets[b]) - intersect
        similarity = intersect / union if union else 1.0
        if similarity >= edge_threshold:
            left, right = sorted((rows[a]["StudyInstanceUID"], rows[b]["StudyInstanceUID"]))
            edges.append({"uid_a": left, "uid_b": right, "jaccard": format(similarity, ".17g")})
        for threshold in (0.90, 0.95):
            if similarity >= threshold:
                counts[str(threshold)] += 1
                if a in expert and b not in expert:
                    linked_to_expert[threshold].add(b)
                elif b in expert and a not in expert:
                    linked_to_expert[threshold].add(a)
    result = {
        "scope": "Five-word-shingle Jaccard candidate audit; similarity does not establish patient identity",
        "reports": len(rows),
        "expert_reports": len(expert),
        "candidate_pairs_with_shared_rare_shingles": len(shared),
        "exact_report_pairs_added_after_indexing": forced_exact_pairs,
        "very_common_shingles_skipped_for_candidate_generation": skipped_common,
        "near_duplicate_pair_counts": dict(counts),
        "nonexpert_reports_linked_to_expert_at_0p90": len(linked_to_expert[0.90]),
        "nonexpert_reports_linked_to_expert_at_0p95": len(linked_to_expert[0.95]),
        "limitation": "A low similarity cannot prove patient independence; templated reports can be similar for unrelated patients",
    }
    if edges_output is not None:
        edges_output.parent.mkdir(parents=True, exist_ok=True)
        with edges_output.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["uid_a", "uid_b", "jaccard"])
            writer.writeheader()
            writer.writerows(sorted(edges, key=lambda edge: (edge["uid_a"], edge["uid_b"])))
        result["edge_artifact"] = {
            "threshold": edge_threshold, "edges": len(edges),
            "source_train_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "edges_sha256": hashlib.sha256(edges_output.read_bytes()).hexdigest(),
            "coverage": "Rare-shingle candidates plus forced exact pairs; not an exhaustive all-pairs near-report census",
        }
        edges_output.with_suffix(".audit.json").write_text(json.dumps(result["edge_artifact"], indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--edges-output", type=Path, help="Local ignored UID edge CSV; no report text")
    parser.add_argument("--edge-threshold", type=float, default=0.90)
    args = parser.parse_args()
    result = audit(args.input, args.edges_output, args.edge_threshold)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"Checked {result['reports']} reports; expert-linked at 0.95: {result['nonexpert_reports_linked_to_expert_at_0p95']}")


if __name__ == "__main__":
    main()
