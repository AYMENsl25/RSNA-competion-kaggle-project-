"""Exploratory lexical triage, never labels or a complete evidence census."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from n01_assertions import TARGETS, TERMS

# Unvalidated multilingual stems: counts need native-language review. Abbreviations
# are deliberately broad and may refer to another structure or reporting context.
EXTRA = {
    'ACL': r'\b(?:lca|vkb|ön çapraz|anteriorni križni|предна кръстна|πρόσθι.{0,15}χιαστ)\b',
    'MCL': r'\b(?:lcm|innenband|iç yan bağ|medijalni kolateralni|медиал.{0,15}колатерал|έσω πλαγ)\b',
    'Medial Meniscus': r'(?:medial\w*|medijal\w*|inner\w*|iç|медиал\w*|έσω).{0,30}(?:menis\w*|μηνίσ\w*|менис\w*)|(?:menis\w*|μηνίσ\w*|менис\w*).{0,30}(?:medial\w*|medijal\w*|inner\w*|медиал\w*|έσω)',
    'Lateral Meniscus': r'(?:lateral\w*|vanjsk\w*|außen\w*|dış|латерал\w*|έξω).{0,30}(?:menis\w*|μηνίσ\w*|менис\w*)|(?:menis\w*|μηνίσ\w*|менис\w*).{0,30}(?:lateral\w*|vanjsk\w*|außen\w*|латерал\w*|έξω)',
    'Medial OA': r'(?:medial\w*|medijal\w*|медиал\w*|έσω).{0,30}(?:compart\w*|kompart\w*|femorotib\w*|tibiofemor\w*|odjel\w*)',
    'Lateral OA': r'(?:lateral\w*|латерал\w*|έξω).{0,30}(?:compart\w*|kompart\w*|femorotib\w*|tibiofemor\w*|odjel\w*)',
    'PF OA': r'patellofemor\w*|femoropatell\w*|femororrotul\w*|patelofemor\w*|пателофемор\w*|επιγονατιδομηρι\w*',
    'Effusion': r'\b(?:derrame|erguss|effus\w*|efüzyon|izljev|излив|ύδραρθρ\w*)\b',
    'Synovitis': r'\b(?:sinovit\w*|synovit\w*|синовит\w*|υμενίτι\w*)\b',
    "Baker's": r'\bbaker\w*\b|poplite\w*.{0,20}(?:cist\w*|kist\w*|zyst\w*|cyst\w*)',
    'Contusion': r'\b(?:kontuz\w*|contusi\w*|контуз\w*)\b',
    'Fracture': r'\b(?:fractur\w*|fraktur\w*|kırık\w*|prijelom\w*|фрактур\w*|κάταγ\w*)\b',
}
CUES = {
    'severity': r'\b(?:moderate|large|massive|high[- ]grade|complete|full[- ]thickness|moderad\w*|sever\w*|grave\w*|hochgrad\w*|vollständig\w*|komplett\w*|büyük|tam|izražen\w*|umjeren\w*|potpun\w*|пълн\w*|теж\w*|σοβαρ\w*|μεγάλ\w*)\b',
    'low_severity': r'\b(?:small|mild|trace|minimal|low[- ]grade|leve\w*|pequeñ\w*|gering\w*|leicht\w*|küçük|hafif|blag\w*|малък|лека|ήπι\w*)\b',
    'negation': r'\b(?:no|without|absent|negative|not|cannot|sin|ningún\w*|kein\w*|ohne|nicht|yok\w*|izlenmedi|bez|nema|няма|без|χωρίς|δεν|sans|aucun\w*|geen|zonder)\b',
    'uncertainty': r'\b(?:possible|possibly|probable|suspect\w*|equivocal|questionable|cannot exclude|cannot rule out|not excluded|may|might|posible\w*|probable\w*|sospech\w*|möglich\w*|verdacht|şüphe\w*|olası|moguć\w*|suspekt\w*|вероят\w*|възмож\w*|πιθαν\w*|possible\w*|mogelijk\w*)\b|\?',
    'temporal_or_surgical': r'\b(?:history|prior|previous|remote|old|healed|chronic|reconstruction|repair|postoperative|antecedente\w*|crónic\w*|chronisch\w*|alt\w*|kronik|eski|hronič\w*|kronič\w*|хронич\w*|χρόνι\w*)\b',
    'dimension': r'\b\d+(?:[.,]\d+)?\s*(?:mm|cm)\b',
}

def inventory(report: str) -> dict[str, dict[str, int]]:
    """Count mention and cue co-occurrence within a +/-120 character window.

    Windows are not assertion scopes: a cue can concern another finding, and
    "cannot exclude" intentionally hits BOTH negation and uncertainty.
    """
    text = report.casefold()
    result = {}
    for target in TARGETS:
        matches = list(re.finditer('(?:' + TERMS[target] + ')|(?:' + EXTRA[target] + ')', text))
        windows = [text[max(0, m.start()-120):m.end()+120] for m in matches]
        result[target] = {'mention': int(bool(matches)),
                          **{name: int(any(re.search(pattern, window) for window in windows))
                             for name, pattern in CUES.items()}}
    return result

def run(train: Path, manifest: Path, private: Path, summary: Path) -> dict:
    from langdetect import DetectorFactory, LangDetectException, detect_langs
    DetectorFactory.seed = 0
    private_root = Path(__file__).resolve().parents[2] / 'outputs'
    if not private.resolve().is_relative_to(private_root.resolve()):
        raise ValueError('Private diagnostics must be inside ignored outputs')
    with manifest.open(encoding='utf-8-sig', newline='') as h:
        all_manifest = list(csv.DictReader(h))
    if len({r['StudyInstanceUID'] for r in all_manifest}) != len(all_manifest):
        raise ValueError('Duplicate manifest UID')
    eligible = {r['StudyInstanceUID']: r for r in all_manifest if r['partition'] in {'candidate_train', 'candidate_validation'}}
    if any(r['exclusion_reason'] or r['expert_labels_access_allowed'] != '0' for r in eligible.values()):
        raise ValueError('Unsafe candidate manifest')
    counts = defaultdict(lambda: {t: Counter({c: 0 for c in ('mention', *CUES)}) for t in TARGETS})
    language_counts = Counter()
    rows = []
    seen = set()
    with train.open(encoding='utf-8-sig', newline='') as h:
        for source in csv.DictReader(h):
            uid = source['StudyInstanceUID']
            if uid not in eligible:
                continue
            if uid in seen:
                raise ValueError('Duplicate candidate source UID')
            seen.add(uid)
            report = source['Report']
            if hashlib.sha256(' '.join(report.casefold().split()).encode()).hexdigest() != eligible[uid]['report_sha256']:
                raise ValueError('Report hash mismatch')
            try:
                guess = detect_langs(report)[0]
                language = guess.lang if guess.prob >= .90 else 'und'
            except LangDetectException:
                language = 'und'
            key = eligible[uid]['partition'] + '/' + language
            language_counts[key] += 1
            for target, cues in inventory(report).items():
                counts[key][target].update(cues)
                rows.append({'StudyInstanceUID': uid, 'partition': eligible[uid]['partition'],
                             'language': language, 'target': target, **cues})
    if seen != set(eligible):
        raise ValueError('Missing candidate report')
    private.parent.mkdir(parents=True, exist_ok=True)
    with private.open('w', encoding='utf-8', newline='') as h:
        writer = csv.DictWriter(h, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    result = {
        'status': 'potential_lexical_evidence_triage_not_labels_not_validated',
        'studies': len(seen), 'report_counts_by_partition_language': dict(sorted(language_counts.items())),
        'counts_by_partition_language_target': {k: {t: dict(c) for t,c in ts.items()} for k,ts in sorted(counts.items())},
        'unit': 'Number of reports with >=1 lexical hit; overlapping cue counts are not additive',
        'cue_window': '120 characters before/after each target lexical hit, irrespective of clause',
        'limitations': ['Lexicons are unvalidated, incomplete, and not native-language-reviewed.',
                       'Potential upper bound only within detected lexical matches, not an upper bound on actual cohort findings.',
                       'OA compartment mention does not imply cartilage loss; ligament abbreviation may be ambiguous.',
                       'Severity, negation, uncertainty, dimension and temporal cues can concern a different target.',
                       'Negation and uncertainty overlap intentionally; no polarity or observation masks are inferred.',
                       'Language identification is heuristic. No report annotation, MRI truth or target support is established.'],
        'sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (manifest, private, Path(__file__))},
        'expert_values_consulted': False,
    }
    summary.write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    return result

if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('train', 'manifest', 'private', 'summary'):
        p.add_argument('--'+name, required=True, type=Path)
    a = p.parse_args()
    result = run(a.train, a.manifest, a.private, a.summary)
    print(json.dumps({'status': result['status'], 'studies': result['studies']}))
