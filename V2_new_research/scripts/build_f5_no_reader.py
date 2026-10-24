"""F5: remove F0's small-reader stage; preserve all main inference and weights."""
import ast
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'notebooks/f0_fork_0958/fork.ipynb'
OUT = ROOT / 'notebooks/f5_no_reader'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    source = json.loads(BASE.read_text(encoding='utf-8'))
    nb = json.loads(BASE.read_text(encoding='utf-8'))
    assert len(nb['cells']) == 19
    assert 'OWN_W_PER_TARGET' in ''.join(nb['cells'][18]['source'])
    # Part A3 already writes the complete main-stack submission. Drop all Part B.
    nb['cells'] = nb['cells'][:13]
    receipt = '''# F5 main-stack-only integrity checks and receipt; no prediction changes.
import hashlib
assert r224 is not None, 'F5 requires Raptor224'
assert not rap_fail.any() and not our_fail.any(), 'F5 refuses missing main-family predictions'
assert len(set(test_ids)) == len(test_ids), 'duplicate study IDs'
_final = pd.read_csv(WORK / 'submission.csv', dtype={'StudyInstanceUID': str})
assert list(_final.columns) == cols and len(cols) == 13
assert _final['StudyInstanceUID'].tolist() == test_ids
assert _final['StudyInstanceUID'].is_unique
assert np.isfinite(_final[labels].values).all()
assert ((_final[labels].values >= 0) & (_final[labels].values <= 1)).all()
_receipt = {'experiment': 'F5_no_small_reader', 'rows': len(_final),
            'predictive_change': 'omit small-reader stage; original main-stack weights',
            'weights': W_RAPTOR_C, 'raptor224_share': A224,
            'elapsed_seconds': time.time() - NB_T0,
            'submission_sha256': hashlib.sha256((WORK / 'submission.csv').read_bytes()).hexdigest(),
            'scope': 'execution checks only; no measured leaderboard gain'}
(WORK / 'f5_receipt.json').write_text(json.dumps(_receipt, indent=2))
print('F5_COMPLETE', json.dumps(_receipt), flush=True)
'''
    nb['cells'].append({'cell_type': 'code', 'metadata': {}, 'execution_count': None,
                        'outputs': [], 'source': receipt.splitlines(keepends=True)})
    # Remove inherited visual/write-up claims about the reader from this notebook.
    nb['cells'][0] = {'cell_type': 'markdown', 'metadata': {}, 'source': [
        '# F5: main-stack-only ablation\n',
        'Speculative candidate. Removes the small reader from F0; no score claimed.\n']}
    nb['cells'][1] = {'cell_type': 'markdown', 'metadata': {}, 'source': [
        'Main inference cells and family weights match F0 exactly. Final integrity checks fail closed.\n']}
    hashes = {}
    for i, cell in enumerate(nb['cells']):
        if 2 <= i <= 12:
            assert cell['source'] == source['cells'][i]['source']
            hashes[str(i)] = hashlib.sha256(''.join(cell['source']).encode()).hexdigest()
        if cell['cell_type'] == 'code':
            cell['outputs'], cell['execution_count'] = [], None
            code = ''.join(cell['source'])
            if code.startswith('%%writefile '):
                code = code.split('\n', 1)[1]
            ast.parse(code)
    OUT.mkdir(parents=True, exist_ok=False)
    meta = json.loads(BASE.with_name('kernel-metadata.json').read_text())
    meta.update(id='aymenslimani/rsna-knee-f5-no-small-reader',
                title='RSNA knee F5 no small reader', code_file='f5.ipynb')
    # Preserve original offline inputs/container: only the reader execution is removed.
    (OUT / 'f5.ipynb').write_text(json.dumps(nb, indent=1), encoding='utf-8')
    (OUT / 'kernel-metadata.json').write_text(json.dumps(meta, indent=2), encoding='utf-8')
    manifest = {'parent_sha256': digest(BASE), 'candidate_sha256': digest(OUT / 'f5.ipynb'),
                'unchanged_source_cell_sha256': hashes, 'preview_timeout_seconds': 1800,
                'hypothesis': 'removing the small reader may reduce harmful rank perturbations',
                'evidence_of_gain': None, 'training': False, 'status': 'built_not_submitted'}
    (OUT / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()
