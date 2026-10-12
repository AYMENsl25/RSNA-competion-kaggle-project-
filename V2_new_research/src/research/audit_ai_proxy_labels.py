"""Profile a fixed public report-label candidate against the AI-only B packet.

This never reads the 58 expert label values and prints aggregate metrics only.
"""

import argparse
import csv
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

TARGETS = ("ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA",
           "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture")


def rows(path):
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def auc(labels, scores):
    positive = [s for y, s in zip(labels, scores) if y == 1]
    negative = [s for y, s in zip(labels, scores) if y == 0]
    if not positive or not negative:
        return None
    wins = sum((p > n) + 0.5 * (p == n) for p in positive for n in negative)
    return wins / (len(positive) * len(negative))


def audit(manifest_path, label_path, packet_path, map_path):
    manifest = {r["StudyInstanceUID"]: r for r in rows(manifest_path)}
    labels = {r["StudyInstanceUID"]: r for r in rows(label_path)}
    packet = {r["review_id"]: r for r in rows(packet_path)}
    mapping = rows(map_path)
    if set(labels) != set(manifest) or len(mapping) != len(packet):
        raise ValueError("UID universe or packet size mismatch")
    if len({r["review_id"] for r in mapping}) != len(mapping) or set(packet) != {r["review_id"] for r in mapping}:
        raise ValueError("Review ID map mismatch")
    train = [uid for uid, r in manifest.items() if r["partition"] == "candidate_train"]
    output = {
        "scope": "AI report-assertion proxy; no expert-label values read; not image ground truth",
        "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "public_candidate_sha256": hashlib.sha256(label_path.read_bytes()).hexdigest(),
        "ai_packet_sha256": hashlib.sha256(packet_path.read_bytes()).hexdigest(),
        "train_studies": len(train), "ai_validation_reports": len(mapping), "targets": {},
    }
    for target in TARGETS:
        train_scores = [float(labels[uid][target]) for uid in train]
        if not all(math.isfinite(x) and 0 <= x <= 1 for x in train_scores):
            raise ValueError("Invalid candidate probability")
        observed = [x for x in train_scores if x != 0.5]
        reference, predicted, agreement = [], [], Counter()
        for item in mapping:
            if item["partition"] != "candidate_validation":
                raise ValueError("Non-validation packet map")
            state = packet[item["review_id"]][f"{target} state"]
            score = float(labels[item["StudyInstanceUID"]][target])
            if state in {"positive", "negative"}:
                reference.append(int(state == "positive"))
                predicted.append(score)
                agreement["evaluated"] += 1
                agreement["candidate_masked_at_0p5"] += int(score == 0.5)
                if score != 0.5:
                    agreement["sign_agree"] += int((score > 0.5) == (state == "positive"))
        counts = Counter(reference)
        output["targets"][target] = {
            "train_positive": sum(x > 0.5 for x in observed),
            "train_negative": sum(x < 0.5 for x in observed),
            "train_masked": len(train_scores) - len(observed),
            "ai_B_positive": counts[1], "ai_B_negative": counts[0],
            "ai_B_auc": auc(reference, predicted),
            "ai_B_candidate_masked": agreement["candidate_masked_at_0p5"],
            "ai_B_sign_agreement_on_unmasked": (agreement["sign_agree"] /
                                                 (agreement["evaluated"] - agreement["candidate_masked_at_0p5"])
                                                 if agreement["evaluated"] > agreement["candidate_masked_at_0p5"] else None),
        }
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for name in ("manifest", "labels", "packet", "map", "out"):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    result = audit(args.manifest, args.labels, args.packet, args.map)
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"Audited {result['train_studies']} train studies and {result['ai_validation_reports']} AI-B reports")
