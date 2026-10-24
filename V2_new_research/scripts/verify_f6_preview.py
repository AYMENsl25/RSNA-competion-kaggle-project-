"""Independent reconstruction of F6 main and reader blends from prediction files."""
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.special import ndtr, ndtri

ROOT = Path(__file__).resolve().parents[1]
FOLDER = ROOT / 'outputs/f6_preview_v1'


def main():
    receipt = json.loads((FOLDER / 'f6_receipt.json').read_text())
    blend = json.loads((FOLDER / 'blend_receipt.json').read_text())
    assert receipt['experiment'] == 'F6_gaussian_rank' and receipt['reader_complete']
    assert blend['raptor_failed'] == blend['ours_failed'] == blend['both_failed'] == 0
    weights = {'Medial OA': .5, 'Lateral OA': .2, 'PF OA': .6, 'Synovitis': .8, "Baker's": .35}
    own_weights = {"Baker's": .035, 'Contusion': .03, 'Medial OA': .025, 'ACL': .025,
                   'Medial Meniscus': .02, 'MCL': .02, 'Lateral Meniscus': .008,
                   'Fracture': .008, 'Lateral OA': .008, 'PF OA': .008,
                   'Effusion': .004, 'Synovitis': .003}
    assert receipt['main_weights'] == weights and receipt['reader_weights'] == own_weights
    assert receipt['raptor224_share'] == .35
    final = pd.read_csv(FOLDER / 'submission.csv', dtype={'StudyInstanceUID': str})
    assert len(final) == receipt['rows'] > 0 and len(final.columns) == 13
    assert final.StudyInstanceUID.is_unique
    cols = list(final.columns[1:])

    def read(name):
        frame = pd.read_csv(FOLDER / name, dtype={'StudyInstanceUID': str})
        assert frame.StudyInstanceUID.is_unique
        assert set(frame.StudyInstanceUID) == set(final.StudyInstanceUID)
        frame = frame.set_index('StudyInstanceUID').loc[final.StudyInstanceUID, cols]
        assert np.isfinite(frame.values).all()
        return frame

    scores = [ndtri((read(name).rank(method='average').to_numpy() - .5) / len(final))
              for name in ('raptor_raw.csv', 'raptor224_sub.csv', 'ours_sub.csv')]
    raw, r224, ours = scores
    w = np.array([weights.get(c, .5) for c in cols])
    expected_main = ndtr(w * (.65 * raw + .35 * r224) + (1 - w) * ours)
    stack = read('_public_stack.csv')
    np.testing.assert_allclose(stack.to_numpy(), expected_main, rtol=0, atol=1e-7)
    own = read('_own.csv')
    assert ((own.values >= 0) & (own.values <= 1)).all()
    w_reader = np.array([own_weights[c] for c in cols])
    expected_final = pd.DataFrame((1 - w_reader) * stack.rank(method='average', pct=True).to_numpy()
                                 + w_reader * own.rank(method='average', pct=True).to_numpy()
                                 ).rank(method='average', pct=True).to_numpy()
    actual = final[cols].to_numpy()
    assert np.isfinite(actual).all() and ((actual >= 0) & (actual <= 1)).all()
    np.testing.assert_allclose(actual, expected_final, rtol=0, atol=1e-12)
    checksum = hashlib.sha256((FOLDER / 'submission.csv').read_bytes()).hexdigest()
    assert checksum == receipt['submission_sha256']
    result = {'status': 'passed', 'kernel': 'aymenslimani/rsna-knee-f6-gaussian-rank',
              'version': 1, 'rows': len(final), 'columns': list(final.columns),
              'max_main_error': float(np.max(np.abs(stack.to_numpy() - expected_main))),
              'max_final_error': float(np.max(np.abs(actual - expected_final))),
              'submission_sha256': checksum, 'preview_seconds': receipt['elapsed_seconds'],
              'scope': 'visible execution only; no score or hidden runtime claim'}
    (FOLDER / 'verification.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
