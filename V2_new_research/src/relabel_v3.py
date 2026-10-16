"""E4: severity-aware rule relabel of the public report labels (no API, CPU only).

Host rubric (research/LABEL_RUBRIC.md): Effusion and Baker's are positive only when
moderate/large. Public labels mark mild/small/trace mentions positive (~0.76-0.86).
Rule: if every non-negated mention of the finding carries a mild word and none carries a
moderate+ word, a public positive becomes soft 0.3. Anything else is left unchanged.

Rules were written from the rubric, AI packet A (60 candidate_train reports) and
candidate_train report text only. AI packet B (120 candidate_validation reports) is
evaluation only. The 58 expert rows (and their linked reports, partition 'excluded')
are never relabelled and their label columns are never read.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import unicodedata
from collections import Counter
from pathlib import Path

MILD_VALUE = 0.3
ROOT = Path(__file__).resolve().parents[1]

# Finding terms, 9 languages (EN ES TR HR EL DE BG NL FR). Stems; matched on normalised text.
TERMS = {
    "Effusion": (
        r"effusion|joint fluid|fluid in the joint|intra-?articular fluid|hydrops|hemarthros|haemarthros"
        r"|derrame|efusion|liquido (?:intra)?articular|hemartros"
        r"|eff?uzyon|eklem (?:ici |icinde )?(?:\w+ )?sivi|eklem aralig\w* sivi|sivi miktari|\bsivi\b"
        r"|izljev|hemartroz|tekucin"
        r"|ενδαρθρικ\w* συλλογ\w*(?: υγρου)?|συλλογ\w* υγρου|υγρου|υδραρθρ|αιμαρθρ"
        r"|erguss|hamarthros|gelenkfl\w*"
        r"|излив|течност|хидрартроз|хемартроз"
        r"|vocht|effusie|\w*uitstorting|opzetting|hydrops"
        r"|epanchement|liquide articulaire"
    ),
    "Baker's": (
        r"baker|popliteal\w* cyst\w*|popliteal fossa cyst|gastro\w*[- /]semimembran\w*|semimembran\w*[- /]gastro\w*"
        r"|quiste popl\w*|popliteal kist|poplitealna cist|poplitealn\w* cist"
        r"|ιγνυακ\w* κυστ|poplitealzyste|popliteazyste|бекер|бейкър|поплитеал\w* кист"
        r"|kyste poplite|kyste de baker|popliteacyste"
    ),
}
# Mild-only words. Greek micro sign is normalised to mu, accents stripped.
MILD = (
    r"\b(?:trace|traces|minimal|minimally|small|smaller|mild|mildly|slight|slightly|little|minor|tiny|physiologic\w*|subtle|discrete|low volume|thin|collapsed|colapsed|early|tiny|winzig\w*|kolabir\w*|tank\w*|lamelarn\w*|pocetn\w*|beperkt\w*"
    r"|leve|pequen\w*|escas\w*|minim\w*|discret\w*|ligero|ligera|poca|poco|fisiologic\w*"
    r"|hafif|minimal|az|milimetrik|azami|kucuk|eser|fizyolojik"
    r"|sitn\w*|mali|mala|malo|male|malen\w*|manj\w*|minimaln\w*|blag\w*|blaz\w*|diskretn\w*|fizioloski\w*|neznatn\w*"
    r"|μικρ\w*|ελαχιστ\w*|μολισ|ηπι\w*|λιγ\w*|φυσιολογικ\w*"
    r"|gering\w*|klein\w*|wenig|kaum|etwas|akzentuiert\w*|leicht\w*|diskret\w*|schmal\w*|physiologisch\w*|minimal\w*|spur\w*"
    r"|мал\w*|минимал\w*|лек\w*|незначител\w*|оскъд\w*|физиологич\w*"
    r"|kleine|klein|geringe|licht\w*|beetje|fysiologisch\w*|minimaal"
    r"|petit\w*|minime|faible|leger\w*|physiologique)\b"
)
MODERATE = (
    r"\b(?:moderate\w*|large|larger|massive|extensive|considerable|marked|markedly|significant|prominent|sizable|substantial|big"
    r"|moderad\w*|abundante|importante|grande|marcad\w*|considerable|cuantioso"
    r"|orta|belirgin|fazla|buyuk|yogun|masif|artmis"
    r"|umjeren\w*|obiln\w*|vec\w*|velik\w*|izrazen\w*|znacajn\w*|opsezn\w*"
    r"|μετρι\w*|μεγαλ\w*|σημαντικ\w*|αφθον\w*|εκτεταμ\w*"
    r"|maßig\w*|massig\w*|moderat\w*|ausgepragt\w*|gros\w*|deutlich\w*|erheblich\w*|kraftig\w*|massiv\w*|mittel\w*|reichlich\w*"
    r"|умерен\w*|голям\w*|значител\w*|изразен\w*|обилен\w*|обилн\w*"
    r"|matig\w*|groot|grote|fors\w*|uitgesproken|aanzienlijk\w*|ruim\w*"
    r"|modere\w*|abondant\w*|volumineu\w*|marque\w*)\b"
)
# "increased" (TR artmis) alone signals more than physiological; "hafif derecede artmis" stays mild
# because MILD wins only when no MODERATE word is present -> handle by dropping artmis when hafif present.
NEGATION = (
    r"\b(?:no|not|without|absent|absence|negative for|nor"
    r"|sin|no hay|ausencia|yok|yoktur|izlenmedi|saptanmadi|bez|nema|nije|χωρισ|δεν|ουτε"
    r"|kein\w*|ohne|nicht|без|няма|не|geen|zonder|niet|pas d\w*|absence|sans)\b"
)


def norm(text: str) -> str:
    text = text.replace("µ", "μ").casefold()
    text = "".join(c for c in unicodedata.normalize("NFD", text) if unicodedata.category(c) != "Mn")
    return text.replace("ı", "i")


def sentences(text: str) -> list[str]:
    return [s for s in re.split(r"[;\n]+|\.(?!\d)|(?:^|\s)-\s|\s•\s", text) if s.strip()]


SIZE = r"(\d+(?:[.,]\d+)?)(?:\s*[x*×/]\s*(\d+(?:[.,]\d+)?))?(?:\s*[x*×/]\s*(\d+(?:[.,]\d+)?))?\s*(mm|cm)\b"


def max_size_mm(text: str) -> float | None:
    sizes = [max(float(v.replace(",", ".")) for v in m.groups()[:3] if v) * (10 if m[4] == "cm" else 1)
             for m in re.finditer(SIZE, text)]
    return max(sizes) if sizes else None


def mention_grades(report: str, target: str) -> list[str]:
    """One grade per non-negated mention: 'mild', 'moderate' or 'unspecified'."""
    grades = []
    for sent in sentences(norm(report)):
        for m in re.finditer(TERMS[target], sent):
            before, after = sent[max(0, m.start() - 40):m.start()], sent[m.end():m.end() + 40]
            if re.search(NEGATION, before) or re.match(r"\s*(?:\w+\s){0,2}(?:yok|izlenmedi|saptanmadi|not seen|absent)\b", after):
                continue
            window = before + " " + after
            mild, moderate = re.search(MILD, window), re.search(MODERATE, window)
            if moderate and moderate.group(0) == "artmis" and mild:
                moderate = None  # "hafif derecede artmis" = mildly increased
            if target == "Baker's" and not moderate and (size := max_size_mm(window)) is not None:
                # ponytail: 30 mm cut is a dev-set heuristic (packet A: 16-23 mm read as small), not a rubric number
                moderate, mild = (True, None) if size >= 30 else (None, True)
            grades.append("moderate" if moderate else "mild" if mild else "unspecified")
    return grades


def relabel_value(report: str, target: str, public: float) -> tuple[float, str]:
    if public <= 0.5:
        return public, ""
    grades = mention_grades(report, target)
    if grades and set(grades) == {"mild"}:
        return MILD_VALUE, f"{target}:{public}->{MILD_VALUE} mild-only"
    return public, ""


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def load_reports(train_csv: Path, allowed: set[str]) -> dict[str, str]:
    """UID -> report for allowed UIDs; label columns are never accessed."""
    out = {}
    for row in read_csv(train_csv):
        if row["StudyInstanceUID"] in allowed:
            out[row["StudyInstanceUID"]] = row["Report"]
    return out


def build(labels: list[dict[str, str]], reports: dict[str, str]) -> tuple[list[dict], Counter]:
    rows, changed = [], Counter()
    for row in labels:
        new, log = dict(row), []
        report = reports.get(row["StudyInstanceUID"])
        if report is not None:
            for target in TERMS:
                value, note = relabel_value(report, target, float(row[target]))
                if note:
                    new[target], changed[target] = str(value), changed[target] + 1
                    log.append(note)
        new["change_log"] = "; ".join(log)
        rows.append(new)
    return rows, changed


def agreement(packet: Path, uid_map: Path, labels: dict[str, dict]) -> dict:
    """Sign agreement with adjudicated AI states (positive/negative only); public 0.5 = masked."""
    mp = {r["review_id"]: r["StudyInstanceUID"] for r in read_csv(uid_map)}
    out = {}
    for target in ("Effusion", "Baker's", "Lateral OA", "PF OA", "Medial OA"):
        agree = n = 0
        for r in read_csv(packet):
            state, value = r[f"{target} state"], float(labels[mp[r["review_id"]]][target])
            if state in {"positive", "negative"} and value != 0.5:
                n += 1
                agree += (value > 0.5) == (state == "positive")
        out[target] = {"n": n, "agreement": round(agree / n, 4) if n else None}
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train", type=Path, default=ROOT / "data/metadata/train.csv")
    p.add_argument("--labels", type=Path, default=ROOT / "outputs/public_label_candidate/llm_labels_full.csv")
    p.add_argument("--manifest", type=Path, default=ROOT / "outputs/candidate_split_v3/manifest.csv")
    p.add_argument("--out", type=Path, default=ROOT / "outputs/labels_v3/labels_v3.csv")
    a = p.parse_args()
    manifest = read_csv(a.manifest)
    allowed = {r["StudyInstanceUID"] for r in manifest
               if r["partition"] in {"candidate_train", "candidate_validation"} and r["expert_labels_access_allowed"] == "0"}
    labels = read_csv(a.labels)
    rows, changed = build(labels, load_reports(a.train, allowed))
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with a.out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    part = {r["StudyInstanceUID"]: r["partition"] for r in manifest}
    by_part = Counter((part[r["StudyInstanceUID"]], t) for r in rows for t in TERMS if f"{t}:" in r["change_log"])
    old, new = {r["StudyInstanceUID"]: r for r in labels}, {r["StudyInstanceUID"]: r for r in rows}
    pk = ROOT / "outputs/ai_review_v1/returned"
    mp = ROOT / "outputs/report_review_packets_v1"
    summary = {
        "changed_cells": dict(changed),
        "changed_by_partition": {f"{k[0]}/{k[1]}": v for k, v in sorted(by_part.items())},
        "packet_A_dev_candidate_train": {"before": agreement(pk / "packet_A.csv", mp / "packet_A_private_uid_map.csv", old),
                                          "after": agreement(pk / "packet_A.csv", mp / "packet_A_private_uid_map.csv", new)},
        "packet_B_heldout_candidate_validation": {"before": agreement(pk / "packet_B.csv", mp / "packet_B_private_uid_map.csv", old),
                                                   "after": agreement(pk / "packet_B.csv", mp / "packet_B_private_uid_map.csv", new)},
    }
    (a.out.parent / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
