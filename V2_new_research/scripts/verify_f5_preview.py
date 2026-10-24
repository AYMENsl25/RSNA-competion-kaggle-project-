"""Verify downloaded F5 prediction-only artifacts before exact-version submission."""
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FOLDER = ROOT / 'outputs/f5_preview_v1'


def main():
    receipt = json.loads((FOLDER / 'f5_receipt.json').read_text())
    blend = json.loads((FOLDER / 'blend_receipt.json').read_text())
    assert receipt['experiment'] == 'F5_no_small_reader'
    assert blend['raptor_failed'] == blend['ours_failed'] == blend['both_failed'] == 0
    weights = {'Medial OA': .5, 'Lateral OA': .2, 'PF OA': .6, 'Synovitis': .8, "Baker's": .35}
    assert receipt['weights'] == weights and receipt['raptor224_share'] == .35
    final = pd.read_csv(FOLDER / 'submission.csv', dtype={'StudyInstanceUID': str})
    assert len(final) == receipt['rows'] > 0 and len(final.columns) == 13
    assert final.StudyInstanceUID.is_unique
    cols = list(final.columns[1:])
    frames = []
    for name in ('raptor_raw.csv', 'raptor224_sub.csv', 'ours_sub.csv'):
        frame = pd.read_csv(FOLDER / name, dtype={'StudyInstanceUID': str})
        assert frame.StudyInstanceUID.is_unique
        assert set(frame.StudyInstanceUID) == set(final.StudyInstanceUID)
        frame = frame.set_index('StudyInstanceUID').loc[final.StudyInstanceUID, cols]
        assert np.isfinite(frame.values).all()
        frames.append(frame.rank(method='average', pct=True).to_numpy())
    raw, r224, ours = frames
    w = np.array([weights.get(c, .5) for c in cols])
    expected = w * (.65 * raw + .35 * r224) + (1 - w) * ours
    actual = final[cols].to_numpy()
    assert np.isfinite(actual).all() and ((actual >= 0) & (actual <= 1)).all()
    np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-7)
    checksum = hashlib.sha256((FOLDER / 'submission.csv').read_bytes()).hexdigest()
    assert checksum == receipt['submission_sha256']
    result = {'status': 'passed', 'kernel': 'aymenslimani/rsna-knee-f5-no-small-reader',
              'version': 1, 'rows': len(final), 'columns': list(final.columns),
              'max_blend_reconstruction_error': float(np.max(np.abs(actual - expected))),
              'submission_sha256': checksum, 'preview_seconds': receipt['elapsed_seconds'],
              'scope': 'visible preview execution and blend integrity, not score or hidden runtime'}
    (FOLDER / 'verification.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
