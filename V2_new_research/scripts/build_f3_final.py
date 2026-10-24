"""Build the predeclared F3 inference-only candidate without modifying F0."""
import ast
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'notebooks/f0_fork_0958'
OUT = ROOT / 'notebooks/f3_weight_shrinkage'


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    notebook = json.loads((BASE / 'fork.ipynb').read_text(encoding='utf-8'))
    original = [''.join(c.get('source', [])) for c in notebook['cells']]
    source = original[12]
    old = "W_RAPTOR_C = {'Medial OA': 0.5, 'Lateral OA': 0.2, 'PF OA': 0.6, 'Synovitis': 0.8, \"Baker's\": 0.35}"
    new = "W_RAPTOR_C = {'Medial OA': 0.5, 'Lateral OA': 0.35, 'PF OA': 0.55, 'Synovitis': 0.65, \"Baker's\": 0.425}"
    assert source.count(old) == 1
    source = source.replace(old, new)
    source = source.replace('# Stage 3:', '# F3: fixed 50% shrinkage toward equal family weights.\n# Stage 3:', 1)
    anchor = '\n\ndef pct(v):'
    assert source.count(anchor) == 1
    source = source.replace(anchor, '''
# Fail closed: a missing family would invalidate this controlled comparison.
assert r224 is not None, 'F3 requires the complete Raptor-224 branch'
assert not rap_fail.any() and not our_fail.any(), 'F3 rejects incomplete family inference'
assert len(set(test_ids)) == len(test_ids), 'Duplicate study IDs'
assert np.isfinite(raw.values).all() and np.isfinite(ours.values).all()

def pct(v):''')
    notebook['cells'][12]['source'] = source.splitlines(keepends=True)
    for cell in notebook['cells']:
        if cell['cell_type'] == 'code':
            cell['outputs'] = []
            cell['execution_count'] = None
    final = '''# F3 final output integrity and run receipt; does not alter predictions.
import hashlib
_final = pd.read_csv(WORK / 'submission.csv', dtype={'StudyInstanceUID': str})
assert list(_final.columns) == cols
assert _final['StudyInstanceUID'].tolist() == test_ids
assert _final['StudyInstanceUID'].is_unique
assert np.isfinite(_final[labels].values).all()
assert ((_final[labels].values >= 0) & (_final[labels].values <= 1)).all()
_receipt = {'experiment': 'F3_weight_shrinkage', 'shrinkage': 0.5,
            'weights': W_RAPTOR_C, 'rows': len(_final),
            'elapsed_seconds': time.time() - NB_T0,
            'submission_sha256': hashlib.sha256((WORK / 'submission.csv').read_bytes()).hexdigest(),
            'validation_scope': 'This run only; visible cases do not establish hidden runtime.'}
(WORK / 'f3_receipt.json').write_text(json.dumps(_receipt, indent=2))
print('F3_COMPLETE', json.dumps(_receipt), flush=True)
'''
    notebook['cells'].append(dict(cell_type='code', metadata={}, source=final.splitlines(keepends=True), outputs=[], execution_count=None))
    for i, cell in enumerate(notebook['cells']):
        if cell['cell_type'] == 'code':
            code = ''.join(cell['source'])
            if not code.startswith('%%'):
                ast.parse(code)
        if i < len(original) and i != 12:
            assert ''.join(cell.get('source', [])) == original[i]
    meta = json.loads((BASE / 'kernel-metadata.json').read_text())
    meta.update(id='aymenslimani/rsna-knee-f3-weight-shrinkage', title='RSNA knee F3 weight shrinkage', code_file='f3.ipynb')
    (OUT / 'f3.ipynb').write_text(json.dumps(notebook, indent=1), encoding='utf-8')
    (OUT / 'kernel-metadata.json').write_text(json.dumps(meta, indent=2), encoding='utf-8')
    manifest = dict(parent_sha256=hashlib.sha256((BASE / 'fork.ipynb').read_bytes()).hexdigest(),
                    candidate_sha256=hashlib.sha256((OUT / 'f3.ipynb').read_bytes()).hexdigest(),
                    predictive_change='50% shrinkage of per-finding Raptor weights toward 0.5',
                    unchanged='All model inference cells, datasets, TTA, reader blend and container',
                    preview_timeout_seconds=1800, submission_status='NOT_SUBMITTED_RUNTIME_GATE_PENDING')
    (OUT / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()
