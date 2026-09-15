"""Interface/invariance checks with a fake VLM; these are not accuracy measurements."""
from pathlib import Path
import json
import re
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from PIL import Image
from solution.stage2 import _predict_file, _motion_scan, _frame_number, _overview_indices, _json_object


class CandidateEcho:
    def __init__(self):
        self.calls = 0

    def ask(self, images, prompt, max_new_tokens=128):
        self.calls += 1
        candidates = [[int(n) for n in text.split(',')] for text in re.findall(r'\[([0-9, ]+)\]', prompt)]
        collision = candidates[0][len(candidates[0]) // 2]
        entry = candidates[-1][0]
        return json.dumps(dict(collision_frame=collision, entry_frame=entry, entry_side='RIGHT', evasion_space=1))


with tempfile.TemporaryDirectory(dir=ROOT / 'research', prefix='stage2_contract_') as temporary:
    temporary = Path(temporary)
    predictions = []
    for offset in (0, 10000):
        folder = temporary / str(offset)
        folder.mkdir()
        for i in range(23):
            array = np.zeros((96, 160, 3), dtype=np.uint8)
            array[40:65, 20+i:50+i] = 180
            Image.fromarray(array).save(folder / f'frame_{offset + i*7:06d}.png')
        paths, scores, _ = _motion_scan(sorted(folder.glob('*.png')))
        vlm = CandidateEcho()
        prediction, diagnostics = _predict_file(paths, scores, vlm)
        assert vlm.calls == 4 and diagnostics['calls'] == 4
        valid = {_frame_number(path) for path in paths}
        assert prediction['collision_frame'] in valid and prediction['entry_frame'] in valid
        assert prediction['evasion_space'] == 1 and prediction['entry_side'] == 'RIGHT'
        predictions.append(prediction)
    assert predictions[1]['collision_frame'] - predictions[0]['collision_frame'] == 10000
    assert predictions[1]['entry_frame'] - predictions[0]['entry_frame'] == 10000
    one = [paths[0]]
    vlm = CandidateEcho()
    prediction, _ = _predict_file(one, np.zeros(1), vlm)
    assert prediction['collision_frame'] == prediction['entry_frame'] == _frame_number(one[0])
    try:
        _motion_scan([])
    except ValueError:
        pass
    else:
        raise AssertionError('Empty input must fail explicitly')
    scores = np.zeros(50)
    scores[17] = 10
    assert 17 in _overview_indices(scores)
    assert _json_object('```json\n{"entry_frame":3}\n```')['entry_frame'] == 3
print('PASS: original-number offset invariance, sparse existing frame membership, four-call budget, '
      'singleton input, explicit empty-input error, motion proposal inclusion, fenced JSON parsing. '
      'Fake VLM used; no accuracy claim.')
