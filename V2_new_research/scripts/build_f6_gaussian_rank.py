"""F6: Gaussian-rank main aggregation, retaining F0 inference and reader weights."""
import ast
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'notebooks/f0_fork_0958/fork.ipynb'
OUT = ROOT / 'notebooks/f6_gaussian_rank'


def main():
    nb = json.loads(BASE.read_text(encoding='utf-8'))
    original = [''.join(c.get('source', [])) for c in nb['cells']]
    code = original[12]
    old = "return pd.Series(np.asarray(v, dtype=np.float64)).rank(method='average', pct=True).values"
    new = """from scipy.special import ndtri
    ranks = pd.Series(np.asarray(v, dtype=np.float64)).rank(method='average').values
    # Midranks keep both endpoints strictly inside (0, 1), including n=1.
    return ndtri((ranks - 0.5) / len(ranks))"""
    assert code.count(old) == 1
    code = code.replace(old, new)
    anchor = '\n\ndef pct(v):'
    assert code.count(anchor) == 1
    code = code.replace(anchor, '''
# F6 requires every main branch; no fallback from normal scores to probabilities.
from scipy.special import ndtr
assert r224 is not None and not rap_fail.any() and not our_fail.any()
assert len(set(test_ids)) == len(test_ids)
assert np.isfinite(raw.values).all() and np.isfinite(ours.values).all()

def pct(v):''')
    assert code.count('    out[:, j] = b') == 1
    code = code.replace('    out[:, j] = b', '    out[:, j] = ndtr(b)  # monotonic bounded output for the unchanged reader blend')
    nb['cells'][12]['source'] = code.splitlines(keepends=True)
    reader = 'F6_READER_OK = False\n' + original[18]
    marker = '        _o_final.to_csv(_o_pub, index=False)'
    assert reader.count(marker) == 1
    reader = reader.replace(marker, marker + '\n        F6_READER_OK = True')
    nb['cells'][18]['source'] = reader.splitlines(keepends=True)
    final = '''# F6 output checks and receipt. No score or calibration claim.
import hashlib
assert F6_READER_OK, 'F6 requires successful original reader blend'
_final = pd.read_csv(WORK / 'submission.csv', dtype={'StudyInstanceUID': str})
assert list(_final.columns) == cols and len(cols) == 13
assert _final['StudyInstanceUID'].tolist() == test_ids and _final['StudyInstanceUID'].is_unique
assert np.isfinite(_final[labels].values).all()
assert ((_final[labels].values >= 0) & (_final[labels].values <= 1)).all()
_receipt = {'experiment': 'F6_gaussian_rank', 'rows': len(_final),
            'transform': 'ndtri((average_rank-0.5)/n); original weighted sum; ndtr',
            'main_weights': W_RAPTOR_C, 'raptor224_share': A224,
            'reader_weights': OWN_W_PER_TARGET, 'reader_complete': F6_READER_OK,
            'elapsed_seconds': time.time() - NB_T0,
            'submission_sha256': hashlib.sha256((WORK / 'submission.csv').read_bytes()).hexdigest()}
(WORK / 'f6_receipt.json').write_text(json.dumps(_receipt, indent=2))
print('F6_COMPLETE', json.dumps(_receipt), flush=True)
'''
    nb['cells'].append(dict(cell_type='code', metadata={}, source=final.splitlines(keepends=True), outputs=[], execution_count=None))
    for i in (0, 1, 11):
        nb['cells'][i] = dict(cell_type='markdown', metadata={}, source=[
            '# F6 Gaussian-rank aggregation\n' if i == 0 else
            'Main model ranks are mapped to normal scores before the original weighted sum. Original reader retained. Speculative; no score claimed.\n'])
    hashes = {}
    for i, cell in enumerate(nb['cells']):
        if i < len(original) and i not in (0, 1, 11, 12, 18):
            assert ''.join(cell['source']) == original[i]
            hashes[str(i)] = hashlib.sha256(original[i].encode()).hexdigest()
        if cell['cell_type'] == 'code':
            cell['outputs'], cell['execution_count'] = [], None
            code = ''.join(cell['source'])
            ast.parse(code.split('\n', 1)[1] if code.startswith('%%writefile ') else code)
    OUT.mkdir(parents=True, exist_ok=False)
    meta = json.loads(BASE.with_name('kernel-metadata.json').read_text())
    meta.update(id='aymenslimani/rsna-knee-f6-gaussian-rank', title='RSNA knee F6 Gaussian rank', code_file='f6.ipynb')
    (OUT / 'f6.ipynb').write_text(json.dumps(nb, indent=1), encoding='utf-8')
    (OUT / 'kernel-metadata.json').write_text(json.dumps(meta, indent=2))
    manifest = {'parent_sha256': hashlib.sha256(BASE.read_bytes()).hexdigest(),
                'candidate_sha256': hashlib.sha256((OUT / 'f6.ipynb').read_bytes()).hexdigest(),
                'unchanged_source_cell_sha256': hashes, 'preview_timeout_seconds': 1800,
                'predictive_change': 'Gaussian-rank main aggregation; F0 coefficients and reader retained',
                'evidence_of_gain': None, 'training': False}
    (OUT / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()
