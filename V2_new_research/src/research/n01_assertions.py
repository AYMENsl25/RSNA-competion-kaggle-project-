"""Unvalidated English report assertions. Not an MRI truth or evaluation reference."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

TARGETS = ('ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 'Medial OA',
           'Lateral OA', 'PF OA', 'Effusion', 'Synovitis', "Baker's", 'Contusion', 'Fracture')
STATES = ('positive', 'negative', 'uncertain', 'not-addressed')
TERMS = {
    'ACL': r'\b(?:acl|anterior cruciate ligament)\b',
    'MCL': r'\b(?:mcl|medial collateral ligament)\b',
    'Medial Meniscus': r'\bmedial menisc(?:us|al)\b',
    'Lateral Meniscus': r'\blateral menisc(?:us|al)\b',
    'Medial OA': r'\bmedial (?:compartment|femorotibial|tibiofemoral)\b',
    'Lateral OA': r'\blateral (?:compartment|femorotibial|tibiofemoral)\b',
    'PF OA': r'\b(?:patellofemoral|patello-femoral)\b',
    'Effusion': r'\b(?:joint effusion|effusion)\b',
    'Synovitis': r'\bsynovitis\b',
    "Baker's": r"\b(?:baker'?s? cyst|popliteal cyst)\b",
    'Contusion': r'\b(?:bone|osseous|marrow) (?:contusion|bruise)\b',
    'Fracture': r'\bfracture\b',
}
UNCERTAIN = r'\b(?:possible|possibly|probable|probably|suspected|suspicious|questionable|equivocal|may|might|cannot exclude|can not exclude|cannot rule out|not excluded)\b'
HISTORY = r'\b(?:history|prior|previous|remote|old|healed|postoperative|post-operative|reconstruction|repair|status post|comparison|indication)\b'
NEGATION = r'\b(?:no|without|absent|absence of|negative for|no evidence of)\b'

def rubric_positive(target: str, clause: str) -> bool:
    """Require explicit threshold evidence; unspecified severity remains uncertain."""
    if target == 'ACL':
        return bool(re.search(r'\b(?:high[- ]grade|complete|full[- ]thickness)\b', clause)
                    and re.search(r'\b(?:tear|rupture|ruptured|torn|discontinuity)\b', clause))
    if target == 'MCL':
        return bool(re.search(r'\bacute\b', clause) and re.search(r'\b(?:high[- ]grade|complete)\b', clause)
                    and re.search(r'\b(?:tear|rupture|ruptured|torn)\b', clause))
    if target in {'Effusion', "Baker's"}:
        return bool(re.search(r'\b(?:moderate|large|massive)\b', clause)
                    and not re.search(r'\b(?:small|mild|trace|minimal)\b', clause))
    if target == 'Fracture':
        return bool(re.search(r'\bacute\b', clause) and re.search(r'\b(?:cortical break|fracture line)\b', clause))
    if target == 'Contusion':
        return bool(re.search(r'\b(?:impact|traumatic|trauma)\b', clause)
                    and re.search(r'\b(?:without|no) (?:a |discrete )?fracture(?: line)?\b', clause))
    return True

def oa_assertion(clause: str, before: str) -> str:
    cartilage = bool(re.search(r'\bcartilage\b', clause))
    loss = bool(re.search(r'\b(?:loss|defect|thinning)\b', clause))
    absent_loss = re.search(r'\b(?:no|without) (?:evidence of )?(?:articular )?cartilage (?:loss|defect|thinning)\b', clause)
    normal_cartilage = re.search(r'\b(?:normal|intact|preserved) (?:medial compartment |lateral compartment |patellofemoral |articular )?cartilage\b', clause)
    if cartilage and (absent_loss or (normal_cartilage and not loss)):
        return 'negative'
    depth = bool(re.search(r'\b(?:high[- ]grade|full[- ]thickness)\b', clause))
    # Numeric size must be a cartilage defect dimension, not an unrelated cyst.
    size = re.search(r'\b(?:area|defect|loss)\s+(?:of |measuring |measures )?(\d+(?:\.\d+)?)\s*(mm|cm)\b', clause)
    extensive = bool(re.search(r'\b(?:moderate|large) area\b', clause))
    if size:
        extensive = float(size[1]) * (10 if size[2] == 'cm' else 1) >= 10
    if cartilage and loss and depth and extensive and not re.search(NEGATION, clause):
        return 'positive'
    return 'uncertain'

def extract(report: str, language: str) -> dict[str, str]:
    """Unsupported language is unobserved (not-addressed), never binary negative.

    Conservative clause scopes: commas, sentences, contrast and newlines terminate
    assertion scope. Shared anatomy and coordinated predicates may be missed.
    """
    states = {target: 'not-addressed' for target in TARGETS}
    if language != 'en':
        return states
    text = report.casefold()
    # Strip explicit historical sections through the next recognised current heading.
    text = re.sub(r'(?ms)^\s*(?:history|clinical history|indication|comparison)\s*:.*?(?=^\s*(?:findings|impression|conclusion)\s*:|\Z)', '', text)
    clauses = re.split(r'[;,\n]+|\.(?!\d)|\b(?:but|however|although|whereas)\b', text)
    for target, term in TERMS.items():
        votes = []
        for clause in clauses:
            matches = list(re.finditer(term, clause))
            for match in matches:
                before, after = clause[:match.start()], clause[match.end():]
                if re.search(HISTORY, clause) or re.search(UNCERTAIN, clause) or '?' in clause:
                    votes.append('uncertain')
                    continue
                if target.endswith('OA'):
                    votes.append(oa_assertion(clause, before))
                    continue
                neg = bool(re.search(NEGATION, before) or re.search(r'^\s*(?:is |are )?(?:absent|not present|not seen)\b', after))
                if target in {'ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus'}:
                    anatomy_count = sum(bool(re.search(TERMS[t], clause)) for t in
                                        ('ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus'))
                    if anatomy_count > 1:
                        votes.append('uncertain')
                        continue
                    lesion = re.search(r'\b(?:tear|torn|rupture|ruptured)\b', clause)
                    normal = re.search(r'\b(?:intact|normal|unremarkable)\b', clause)
                    if normal and (lesion or re.search(r'\bnot\b', clause)):
                        votes.append('uncertain')
                    elif normal and not re.search(NEGATION, before):
                        votes.append('negative')
                    elif lesion:
                        votes.append('negative' if neg else ('positive' if rubric_positive(target, clause) else 'uncertain'))
                    else:
                        votes.append('uncertain')
                else:
                    # A question or a generic "normal" sentence does not establish absence.
                    # Contusion's required "without fracture" is not contusion negation.
                    if target == 'Contusion' and rubric_positive(target, clause):
                        neg = bool(re.search(NEGATION, before))
                    votes.append('negative' if neg else ('positive' if rubric_positive(target, clause) else 'uncertain'))
        unique = set(votes)
        if unique:
            states[target] = next(iter(unique)) if len(unique) == 1 else 'uncertain'
    return states

def run(train: Path, manifest: Path, output: Path, summary: Path) -> dict:
    from langdetect import DetectorFactory, LangDetectException, detect_langs
    DetectorFactory.seed = 0
    with manifest.open(encoding='utf-8-sig', newline='') as handle:
        manifest_rows = list(csv.DictReader(handle))
    if len({r['StudyInstanceUID'] for r in manifest_rows}) != len(manifest_rows):
        raise ValueError('Duplicate manifest UID')
    allowed = {r['StudyInstanceUID']: r for r in manifest_rows
               if r['partition'] in {'candidate_train', 'candidate_validation'}}
    if any(r['exclusion_reason'] or r['expert_labels_access_allowed'] != '0' for r in allowed.values()):
        raise ValueError('Unsafe manifest policy')
    output = output.resolve()
    private_root = Path(__file__).resolve().parents[2] / 'outputs'
    if not output.is_relative_to(private_root.resolve()):
        raise ValueError('Row labels must remain in ignored V2 outputs')
    aggregate = defaultdict(lambda: {t: Counter({s: 0 for s in STATES}) for t in TARGETS})
    rows = []
    with train.open(encoding='utf-8-sig', newline='') as handle:
        reader = csv.DictReader(handle)
        # Never inspect any label-column values. Skip protected rows before report access.
        for source in reader:
            uid = source['StudyInstanceUID']
            if uid not in allowed:
                continue
            report = source['Report']
            if hashlib.sha256(' '.join(report.casefold().split()).encode()).hexdigest() != allowed[uid]['report_sha256']:
                raise ValueError('Manifest report hash mismatch')
            try:
                guess = detect_langs(report)[0]
                language = guess.lang if guess.prob >= .90 else 'und'
            except LangDetectException:
                language = 'und'
            states = extract(report, language)
            key = allowed[uid]['partition'] + '/' + language
            for target, state in states.items():
                aggregate[key][target][state] += 1
            rows.append({'StudyInstanceUID': uid, 'partition': allowed[uid]['partition'],
                         'language': language, 'language_supported': int(language == 'en'),
                         **states, **{t + '_mask': int(states[t] in {'positive', 'negative'}) for t in TARGETS}})
    if len(rows) != len(allowed) or len({r['StudyInstanceUID'] for r in rows}) != len(rows):
        raise ValueError('Candidate reports missing or duplicated')
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    counts = {key: {t: {**dict(c), 'mask_observed': c['positive'] + c['negative'],
                       'mask_excluded': c['uncertain'] + c['not-addressed']} for t, c in targets.items()}
              for key, targets in sorted(aggregate.items())}
    result = {'status': 'candidate_unvalidated_not_frozen_not_training_authorization',
              'studies': len(rows), 'supported_languages': ['en'],
              'unsupported_policy': 'All targets not-addressed, masks zero; language_supported=0',
              'host_rubric_verified': True, 'oa_binary_extraction_enabled': True,
              'host_rubric_source': 'https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/733343',
              'counts_by_partition_language_target': counts,
              'sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (manifest, output, Path(__file__))}}
    summary.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    return result

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ('train', 'manifest', 'output', 'summary'):
        parser.add_argument('--' + arg, required=True, type=Path)
    args = parser.parse_args()
    result = run(args.train, args.manifest, args.output, args.summary)
    print(json.dumps({'status': result['status'], 'studies': result['studies']}))
