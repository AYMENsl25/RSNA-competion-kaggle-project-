"""Merge two independent AI annotation passes of a report packet, then apply adjudication.

Labels produced here are AI-generated (LLM), report-derived, and NOT clinician reviewed.
Pass/adjudication JSON: {"<review_id>": {"<target>": {"state": ..., "span": ...}, "notes": "..."}}
Adjudication JSON may cover only the disputed cells.

  validate <packet.csv> <pass.json> [--partial]     schema + literal-span check (prints counts only)
  disputes <packet.csv> <p1.json> <p2.json> <out.json>
  finalize <packet.csv> <p1.json> <p2.json> <adj.json> <out.csv> --reviewer-id ID --date YYYY-MM-DD
"""

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from check_review_packet import STATES, TARGETS, read  # noqa: E402

AI_NOTE = "AI-generated report-derived labels (LLM); not clinician reviewed."


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def cell_errors(report, cell):
    state, span = cell.get("state", ""), (cell.get("span") or "").strip()
    if state not in STATES:
        return "invalid_state"
    if state == "not_addressed":
        return "span_for_not_addressed" if span else None
    return None if span and span in report else "missing_or_nonliteral_span"


def validate(rows, ann, partial=False):
    errors = {}
    for row in rows:
        rid = row["review_id"]
        if rid not in ann:
            if not partial:
                errors.setdefault("missing_review_id", []).append(rid)
            continue
        for target in TARGETS:
            if target not in ann[rid]:
                if not partial:
                    errors.setdefault("missing_target", []).append(f"{rid}|{target}")
                continue
            err = cell_errors(row["Report"], ann[rid][target])
            if err:
                errors.setdefault(err, []).append(f"{rid}|{target}")
    extra = set(ann) - {r["review_id"] for r in rows}
    if extra:
        errors["unknown_review_id"] = sorted(extra)
    return errors


def disputes(rows, p1, p2):
    out = {}
    for row in rows:
        rid = row["review_id"]
        for target in TARGETS:
            a, b = p1[rid][target], p2[rid][target]
            bad = cell_errors(row["Report"], a) or cell_errors(row["Report"], b)
            if a["state"] != b["state"] or bad:
                out.setdefault(rid, {})[target] = {"pass1": a, "pass2": b}
    return out


def finalize(rows, p1, p2, adj, reviewer_id, review_date):
    todo = disputes(rows, p1, p2)
    for rid, cells in todo.items():
        for target in cells:
            if target not in adj.get(rid, {}):
                raise SystemExit(f"unadjudicated dispute: {rid}|{target}")
    for row in rows:
        rid = row["review_id"]
        notes = [AI_NOTE]
        for target in TARGETS:
            if target in todo.get(rid, {}):
                cell = adj[rid][target]
            else:
                a, b = p1[rid][target], p2[rid][target]
                cell = a if not cell_errors(row["Report"], a) else b
            if cell_errors(row["Report"], cell):
                raise SystemExit(f"invalid final cell: {rid}|{target}")
            row[f"{target} state"] = cell["state"]
            row[f"{target} evidence span"] = (cell.get("span") or "").strip() if cell["state"] != "not_addressed" else ""
        for src in (p1, p2, adj):
            if src.get(rid, {}).get("notes"):
                notes.append(src[rid]["notes"])
        row["reviewer_id"], row["review_date"] = reviewer_id, review_date
        row["notes"] = " | ".join(notes)
    return rows


def demo():
    rows = [{"review_id": "r1", "Report": "Moderate effusion. Intact ACL."}]
    blank = {t: {"state": "not_addressed", "span": ""} for t in TARGETS}
    p1 = {"r1": {**blank, "Effusion": {"state": "positive", "span": "Moderate effusion"},
                 "ACL": {"state": "negative", "span": "Intact ACL"}}}
    p2 = {"r1": {**blank, "Effusion": {"state": "uncertain", "span": "Moderate effusion"},
                 "ACL": {"state": "negative", "span": "intact acl"}}}
    assert not validate(rows, p1) and validate(rows, p2) == {"missing_or_nonliteral_span": ["r1|ACL"]}
    d = disputes(rows, p1, p2)
    assert set(d["r1"]) == {"Effusion", "ACL"}
    adj = {"r1": {"Effusion": {"state": "positive", "span": "Moderate effusion"},
                  "ACL": {"state": "negative", "span": "Intact ACL"}}}
    out = finalize([dict(rows[0])], p1, p2, adj, "AI", "2026-10-10")[0]
    assert out["Effusion state"] == "positive" and out["ACL evidence span"] == "Intact ACL"
    assert out["MCL evidence span"] == "" and out["notes"].startswith("AI-generated")
    print("demo ok")


if __name__ == "__main__":
    if sys.argv[1:] == ["demo"]:
        demo()
        raise SystemExit
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["validate", "disputes", "finalize"])
    ap.add_argument("paths", nargs="+", type=Path)
    ap.add_argument("--partial", action="store_true")
    ap.add_argument("--reviewer-id")
    ap.add_argument("--date")
    a = ap.parse_args()
    headers, rows = read(a.paths[0])
    if a.cmd == "validate":
        errs = validate(rows, load(a.paths[1]), a.partial)
        print(json.dumps({k: {"count": len(v), "cells": v[:50]} for k, v in errs.items()} or {"status": "ok"}, indent=2))
        raise SystemExit(1 if errs else 0)
    p1, p2 = load(a.paths[1]), load(a.paths[2])
    for p in (p1, p2):
        if validate(rows, p) .keys() & {"missing_review_id", "missing_target", "invalid_state"}:
            raise SystemExit("a pass is incomplete; run validate first")
    if a.cmd == "disputes":
        d = disputes(rows, p1, p2)
        a.paths[3].write_text(json.dumps(d, ensure_ascii=False, indent=1), encoding="utf-8")
        print(json.dumps({"disputed_reports": len(d), "disputed_cells": sum(map(len, d.values()))}))
    else:
        out = finalize(rows, p1, p2, load(a.paths[3]), a.reviewer_id, a.date)
        with a.paths[4].open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=headers)
            w.writeheader()
            w.writerows(out)
        print(f"wrote {len(out)} rows")
