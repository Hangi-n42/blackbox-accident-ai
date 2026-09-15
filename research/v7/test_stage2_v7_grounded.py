"""Small synthetic CPU contracts; not model accuracy tests or real GT."""
import os
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
os.environ["OPENBLAS_NUM_THREADS"] = "2"
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
FROZEN = ROOT / "artifacts/submissions/verify_v6/model/stage2/code"
sys.path.insert(0, str(FROZEN))
spec = importlib.util.spec_from_file_location("v7_grounded_test_subject", Path(__file__).parent / "solution/stage2_v7_grounded.py")
subject = importlib.util.module_from_spec(spec)
spec.loader.exec_module(subject)
subject.primitives.cv2.setNumThreads(2)


class MockVLM:
    pixel_budget = 1_200_000

    def __init__(self, answers):
        self.answers, self.calls = list(answers), []

    def ask(self, images, prompt, max_new_tokens):
        self.calls.append((images[0].copy(), prompt, max_new_tokens))
        answer = self.answers[len(self.calls)-1]
        return answer if isinstance(answer, str) else json.dumps(answer)


class Contracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.paths = []
        for i in range(257):
            p = Path(cls.tmp.name) / f"frame{100 + i*3:06d}.png"
            Image.new("RGB", (96,54), (i % 256,40,110)).save(p, compress_level=1)
            cls.paths.append(p)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def run_mock(self, answers, scores=None, paths=None):
        paths = self.paths if paths is None else paths
        if scores is None:
            scores = np.zeros(len(paths), dtype=np.float32)
            scores[min(1,len(paths)-1)] = 10
        vlm = MockVLM(answers)
        prediction, diagnostics = subject.predict_from_scan(paths,scores,scores,vlm)
        return prediction, diagnostics, vlm

    def test_invalid_json_preserves_motion_and_reports_first_fallback(self):
        pred, diag, mock = self.run_mock(["garbage"]*4)
        self.assertEqual(pred, dict(collision_frame=103,entry_frame=100,entry_side="LEFT",evasion_space=0))
        self.assertEqual(diag["entry_selection_route"],"invalid_fallback_first")
        self.assertIn(103,diag["collision_candidates"])
        self.assertNotIn(103,diag["retained_motion_peaks"])
        self.assertEqual([c[2] for c in mock.calls],[120,24,24,24])
        self.assertEqual(len(mock.calls),4)

    def test_strict_parser_rejects_nonintegers_and_unoffered(self):
        self.assertEqual(subject._strict_json('{"x":1} trailing'),{})
        for value in [True,1.0,"1",2,None]:
            self.assertEqual(subject._select(value,{1:3},5),(5,False))
        self.assertEqual(subject._select(1,{1:3},5),(3,True))
        self.assertEqual(subject._strict_json('```json\n{"x":1}\n```'),{"x":1})

    def test_long_input_window_width_and_exact_anchors(self):
        numbers = [i*7+900 for i in range(5001)]
        offered = subject.primitives._uniform_indices(0,5000,16)
        coarse = offered[8]
        window, selected, diag = subject._window([numbers[0],numbers[coarse],numbers[-1]],numbers,offered,0)
        self.assertEqual(diag["status"],"oversized_bracket")
        self.assertEqual(window[1]-window[0],625)
        dense=subject._dense(window,[selected,1])
        self.assertIn(coarse,dense)
        self.assertIn(1,dense)
        self.assertLessEqual(len(dense),18)

    def test_entry_after_contact_not_rejected_and_valid_coarse_retained(self):
        overview=subject.primitives._uniform_indices(0,256,16)
        value=100+3*overview[10]
        plan={"entry":[value,value,value]}
        pred,diag,_=self.run_mock([plan,{"collision_frame":103},{"entry_frame":None},{"entry_side":"RIGHT","evasion_space":1}])
        self.assertEqual(pred["entry_frame"],value)
        self.assertTrue(diag["entry_after_contact"])
        self.assertEqual(diag["entry_selection_route"],"valid_coarse_retained")
        self.assertIn(100,diag["entry_candidates"])

    def test_spatial_output_cannot_modify_time_or_reference(self):
        first=[{}, {"collision_frame":103},{"entry_frame":100}]
        a,da,_=self.run_mock(first+[{"entry_side":"LEFT","evasion_space":0}])
        b,db,_=self.run_mock(first+[{"entry_side":"RIGHT","evasion_space":1,"collision_frame":868,"entry_frame":868,"reference_frame":868}])
        for key in ["collision_frame","entry_frame"]:
            self.assertEqual(a[key],b[key])
        self.assertEqual(da["reference"],db["reference"])
        self.assertEqual(da["spatial_context_contact_frame"],a["collision_frame"])

    def test_reference_crop_bound_to_only_reference_frame(self):
        plan={"reference_frame":100,"bbox":[200,200,800,900],"target":"test object"}
        _,diag,mock=self.run_mock([plan,{}, {}, {}])
        self.assertTrue(diag["reference"]["bbox_valid"])
        self.assertFalse(diag["reference"]["identity_verified"])
        for trace in diag["trace"][1:]:
            crops=[p for p in trace["rendering"]["panels"] if p["crop"]]
            self.assertEqual(len(crops),1)
            self.assertEqual(crops[0]["frame"],100)
            self.assertGreaterEqual(trace["rendering"]["panel_occupancy"],.75)
        self.assertIn("true clip start",mock.calls[2][1])
        self.assertIn("local-window first image may be later",mock.calls[2][1])

    def test_invalid_reference_never_crops(self):
        for box in ([800,100,200,300],[0,0,1001,900],[0.,0,500,500]):
            reference=subject._reference({"reference_frame":100,"bbox":box},[100],[0])
            self.assertFalse(reference["bbox_valid"])

    def test_one_frame_and_first_selection_not_observed_gt(self):
        pred,diag,_=self.run_mock([{}, {"collision_frame":100},{"entry_frame":100},{}],paths=self.paths[:1])
        self.assertEqual(pred["entry_frame"],100)
        self.assertEqual(diag["entry_selection_route"],"valid_selected_first")
        self.assertTrue(diag["valid_first_is_not_verified_already_entered"])

    def test_input_offset_and_independent_calls(self):
        answers=[{}, {}, {}, {}]
        a,_,_=self.run_mock(answers,paths=self.paths[:20])
        b,_,_=self.run_mock(answers,paths=self.paths[20:40])
        repeat,_,_=self.run_mock(answers,paths=self.paths[:20])
        self.assertEqual(a,repeat)
        self.assertEqual(b["collision_frame"]-a["collision_frame"],60)
        self.assertEqual(b["entry_frame"]-a["entry_frame"],60)

    def test_bad_inputs_fail_before_vlm(self):
        mock=MockVLM([])
        for paths,scores in [([],[]),([self.paths[0]]*2,[0,1]),(self.paths[:2],[0]),(self.paths[:2],[0,float("nan")])]:
            with self.assertRaises(ValueError):
                subject.predict_from_scan(paths,scores,scores,mock)
        self.assertFalse(mock.calls)


if __name__ == "__main__":
    unittest.main()
