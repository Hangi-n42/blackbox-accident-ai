"""CPU-only contracts using fake VLM and mocked sheets; no model is loaded."""
import copy
import importlib.util
import json
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '2'
sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
spec = importlib.util.spec_from_file_location('solution.stage2_contact_verify_v6a', Path(__file__).with_name('stage2_contact_verify_v6a.py'))
candidate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(candidate)


class FakeVLM:
    def __init__(self, answer):
        self.answer, self.calls = answer, []

    def ask(self, images, prompt, max_new_tokens):
        self.calls.append(dict(images=images, prompt=prompt, tokens=max_new_tokens))
        return self.answer


class Contracts(unittest.TestCase):
    def setUp(self):
        self.numbers = [i * 7 + 11 for i in range(60)]
        self.paths = [Path(f'frame_{n:06d}.png') for n in self.numbers]
        self.pred = dict(collision_frame=self.numbers[45], entry_frame=39, entry_side='RIGHT', evasion_space=1)
        self.diag = {'collision_replacement': {'base_collision_frame': self.numbers[3]}, 'earlier': {'value': 1}}
        self.sheet = patch.object(candidate.baseline.base, '_sheet', return_value='FAKE_SHEET')
        self.mock_sheet = self.sheet.start()
        self.addCleanup(self.sheet.stop)

    def invoke(self, raw):
        vlm = FakeVLM(raw)
        result, diagnostics = candidate.refine_collision(self.paths, self.pred, self.diag, vlm)
        self.assertEqual(len(vlm.calls), 1)
        self.assertEqual(vlm.calls[0]['tokens'], 48)
        self.assertIn('two local time windows', vlm.calls[0]['prompt'])
        self.assertEqual(self.mock_sheet.call_args.kwargs['columns'], 5)
        return result, diagnostics['contact_verification']

    def test_noncontiguous_numbers_and_both_centers(self):
        result, detail = self.invoke(json.dumps({'collision_frame': self.numbers[3]}))
        self.assertEqual(result['collision_frame'], self.numbers[3])
        self.assertEqual(detail['centers_valid_path_indices'], [3, 45])
        self.assertLessEqual(len(detail['offered_original_frames']), 18)
        self.assertEqual(detail['offered_original_frames'], sorted(set(detail['offered_original_frames'])))
        self.assertTrue({self.numbers[3], self.numbers[45]}.issubset(detail['offered_original_frames']))
        for center, window in zip([3, 45], detail['window_indices']):
            self.assertLessEqual(len(window), 9)
            self.assertTrue(all(max(0,center-10) <= i <= min(59,center+10) for i in window))

    def test_center_inclusion_exhaustive_lengths_and_boundaries(self):
        for length in range(1, 81):
            for center in range(length):
                window = candidate._window_indices(center, length)
                self.assertIn(center, window)
                self.assertLessEqual(len(window), 9)
                self.assertEqual(window, sorted(set(window)))
                self.assertTrue(all(max(0,center-10) <= i <= min(length-1,center+10) for i in window))

    def test_nearest_replacement_earlier_tie(self):
        with patch.object(candidate.baseline.base, '_uniform_indices', return_value=[0,2,4,6,8,10,12,14,16]):
            self.assertEqual(candidate._window_indices(7,30), [0,2,4,7,8,10,12,14,16])

    def test_strict_parser_rejects_bool_float_string_null_outside_and_malformed(self):
        for value in [True,False,32.0,'32',None,-1,999999,3]:
            with self.subTest(value=value):
                result, detail = self.invoke(json.dumps({'collision_frame':value}))
                self.assertEqual(result, self.pred)
                self.assertTrue(detail['fallback'])
        for raw in ['invalid', '{}', '[]', 'null', '{"collision_frame":NaN}', '{"collision_frame":Infinity}', '```json\n{"collision_frame":32}\n```']:
            with self.subTest(raw=raw):
                result, detail = self.invoke(raw)
                self.assertEqual(result, self.pred)
                self.assertTrue(detail['fallback'])
                json.dumps(detail, allow_nan=False)

    def test_valid_source_number_outside_offered_is_rejected(self):
        _, detail = self.invoke('null')
        unoffered = next(n for n in self.numbers if n not in detail['offered_original_frames'])
        result, detail = self.invoke(json.dumps({'collision_frame':unoffered}))
        self.assertEqual(result, self.pred)
        self.assertEqual(detail['fallback_reason'], 'collision_not_offered')

    def test_other_fields_and_input_dictionaries_unchanged(self):
        original_pred, original_diag = copy.deepcopy(self.pred), copy.deepcopy(self.diag)
        result, detail = self.invoke(json.dumps({'collision_frame':self.numbers[3], 'entry_side':'LEFT'}))
        self.assertEqual({k:v for k,v in result.items() if k!='collision_frame'}, {k:v for k,v in self.pred.items() if k!='collision_frame'})
        self.assertEqual(self.pred, original_pred)
        self.assertEqual(self.diag, original_diag)
        self.assertTrue(detail['other_prediction_fields_unchanged'])

    def test_overlapping_windows_and_single_frame(self):
        self.diag['collision_replacement']['base_collision_frame'] = self.pred['collision_frame']
        _, detail = self.invoke('null')
        self.assertLessEqual(len(detail['offered_original_frames']),9)
        self.paths=[Path('frame_000111.png')]
        self.pred['collision_frame']=111
        self.diag['collision_replacement']['base_collision_frame']=111
        result,detail=self.invoke('{"collision_frame":111}')
        self.assertEqual(detail['offered_original_frames'],[111])
        self.assertEqual(result['collision_frame'],111)

    def test_invalid_center_or_path_order_rejected_before_call(self):
        vlm=FakeVLM('null')
        with self.assertRaises(ValueError):
            candidate.refine_collision(self.paths[::-1],self.pred,self.diag,vlm)
        self.pred['collision_frame']=99999
        with self.assertRaises(ValueError):
            candidate.refine_collision(self.paths,self.pred,self.diag,vlm)
        self.assertEqual(vlm.calls,[])

    def test_predict_file_runs_baseline_once_then_exactly_one_verifier(self):
        vlm=FakeVLM('null')
        with patch.object(candidate.baseline,'_predict_file',return_value=(self.pred,self.diag)) as old:
            result, detail=candidate._predict_file(self.paths,'UNUSED_SCORES',vlm)
        old.assert_called_once_with(self.paths,'UNUSED_SCORES',vlm)
        self.assertEqual(len(vlm.calls),1)
        self.assertEqual(result,self.pred)
        self.assertIn('contact_verification',detail)


if __name__ == '__main__':
    unittest.main()
