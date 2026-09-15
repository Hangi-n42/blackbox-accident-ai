"""CPU-only synthetic contracts. These tests do not measure model accuracy."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from PIL import Image
from solution import stage2_entry_refine as candidate
from solution.stage2 import _uniform_indices

RECORDS = []


class ScriptedVLM:
    def __init__(self, numbers, entry_index, fine_answer):
        self.numbers, self.entry_index, self.fine_answer = numbers, entry_index, fine_answer
        self.calls = []

    def ask(self, images, prompt, max_new_tokens=128):
        self.calls.append({"prompt": prompt, "max_new_tokens": max_new_tokens})
        call = len(self.calls)
        if call == 1:
            return json.dumps({"collision_frame": self.numbers[-1], "entry_side": "RIGHT"})
        if call == 2:
            return json.dumps({"collision_frame": self.numbers[-1]})
        if call == 3:
            return json.dumps({"entry_frame": self.numbers[self.entry_index]})
        if call == 4:
            return '{"evasion_space": 1}'
        if call > 5:
            raise AssertionError("Exceeded five calls")
        if isinstance(self.fine_answer, Exception):
            raise self.fine_answer
        return self.fine_answer(prompt) if callable(self.fine_answer) else self.fine_answer


def choose_interior(prompt):
    frames = json.loads(re.search(r"Lane entry candidates: (\[[^\]]+\])", prompt)[1])
    return json.dumps({"entry_frame": frames[3]})


def run_synthetic(numbers, entry_index=None, fine_answer=choose_interior):
    entry_index = entry_index if entry_index is not None else _uniform_indices(0, len(numbers)-1, 12)[5]
    paths = [Path(f"frame_{number:08d}.jpg") for number in numbers]
    scores = np.zeros(len(paths), dtype=np.float32)
    scores[-1] = 1
    model = ScriptedVLM(numbers, entry_index, fine_answer)
    # Actual V2 calls and parsing run; image reading/rendering is replaced by a CPU placeholder.
    with patch.object(candidate.base, "_sheet", return_value=Image.new("RGB", (32, 32))):
        prediction, diagnostics = candidate._predict_file(paths, scores, model)
    record = {"input_count": len(numbers), "first_number": numbers[0], "last_number": numbers[-1],
              "calls": len(model.calls), "prediction": prediction,
              "entry_refinement": diagnostics["entry_refinement"]}
    RECORDS.append(record)
    return prediction, diagnostics, model


class EntryRefinementContracts(unittest.TestCase):
    def check_unchanged_fields(self, prediction, last):
        self.assertEqual(prediction["collision_frame"], last)
        self.assertEqual(prediction["entry_side"], "RIGHT")
        self.assertEqual(prediction["evasion_space"], 1)

    def test_long_input_bounds_and_coarse_preservation(self):
        prediction, diagnostic, model = run_synthetic(list(range(600)))
        fine = diagnostic["entry_refinement"]["fine_candidates"]
        lower, upper = diagnostic["entry_refinement"]["interval_frames"]
        self.assertEqual(len(model.calls), 5)
        self.assertEqual(diagnostic["calls"], 5)
        self.assertLessEqual(len(fine), 12)
        self.assertIn(diagnostic["entry_refinement"]["coarse_entry_frame"], fine)
        self.assertTrue(all(lower <= number <= upper for number in fine))
        self.assertIn(prediction["entry_frame"], fine)
        self.assertLess(max(np.diff(fine)), upper-lower)
        self.check_unchanged_fields(prediction, 599)

    def test_nonconsecutive_original_numbers(self):
        numbers = [101+index*7 for index in range(300)]
        prediction, diagnostic, model = run_synthetic(numbers)
        fine = diagnostic["entry_refinement"]["fine_candidates"]
        self.assertTrue(set(fine).issubset(numbers))
        self.assertIn(prediction["entry_frame"], numbers)
        self.assertEqual(len(model.calls), 5)
        self.check_unchanged_fields(prediction, numbers[-1])

    def test_offset_equivariance(self):
        left, left_diag, _ = run_synthetic(list(range(300)))
        right, right_diag, _ = run_synthetic(list(range(1000, 1300)))
        self.assertEqual(right["entry_frame"]-left["entry_frame"], 1000)
        self.assertEqual(right["collision_frame"]-left["collision_frame"], 1000)
        self.assertEqual([n+1000 for n in left_diag["entry_refinement"]["fine_candidates"]],
                         right_diag["entry_refinement"]["fine_candidates"])
        self.assertEqual(left["entry_side"], right["entry_side"])
        self.assertEqual(left["evasion_space"], right["evasion_space"])

    def test_one_frame_no_extra_call(self):
        prediction, diagnostic, model = run_synthetic([901], entry_index=0)
        self.assertEqual(len(model.calls), 4)
        self.assertEqual(diagnostic["calls"], 4)
        self.assertEqual(prediction["entry_frame"], 901)
        self.assertEqual(diagnostic["entry_refinement"]["skip_reason"], "all_precontact_frames_already_shown")
        self.check_unchanged_fields(prediction, 901)

    def test_all_frames_present_no_extra_call(self):
        prediction, diagnostic, model = run_synthetic(list(range(12)), entry_index=5)
        self.assertEqual(len(model.calls), 4)
        self.assertEqual(prediction["entry_frame"], 5)
        self.assertFalse(diagnostic["entry_refinement"]["accepted"])

    def test_invalid_answers_keep_coarse_entry(self):
        cases = ["not JSON", "[]", "{}", '{"entry_frame": true}', '{"entry_frame": 1.5}',
                 '{"entry_frame": "136"}', '{"entry_frame": 999999}',
                 '[{"entry_frame": 125}]', 'prefix {"entry_frame": 125}',
                 RuntimeError("synthetic model failure")]
        for answer in cases:
            with self.subTest(answer=str(answer)):
                prediction, diagnostic, model = run_synthetic(list(range(300)), fine_answer=answer)
                self.assertEqual(prediction["entry_frame"], diagnostic["entry_refinement"]["coarse_entry_frame"])
                self.assertFalse(diagnostic["entry_refinement"]["accepted"])
                self.assertEqual(len(model.calls), 5)
                self.check_unchanged_fields(prediction, 299)

    def test_other_fields_cannot_be_overwritten(self):
        def answer(prompt):
            value = json.loads(choose_interior(prompt))
            value.update(collision_frame=0, entry_side="LEFT", evasion_space=0)
            return json.dumps(value)
        prediction, _, _ = run_synthetic(list(range(300)), fine_answer=answer)
        self.check_unchanged_fields(prediction, 299)

    def test_local_start_is_marked_as_window_boundary(self):
        def answer(prompt):
            self.assertIn("the first image may not be the start of the clip", prompt)
            frames = json.loads(re.search(r"Lane entry candidates: (\[[^\]]+\])", prompt)[1])
            return json.dumps({"entry_frame": frames[0]})
        prediction, diagnostic, _ = run_synthetic(list(range(300)), fine_answer=answer)
        audit = diagnostic["entry_refinement"]
        self.assertEqual(prediction["entry_frame"], audit["interval_frames"][0])
        self.assertTrue(audit["selected_local_window_start"])


if __name__ == "__main__":
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(EntryRefinementContracts))
    report = {"scope": "CPU-only mocked VLM synthetic contracts; NOT model accuracy or timing validation.",
              "gpu_used": False, "torch_imported": "torch" in sys.modules,
              "tests_run": result.testsRun, "failures": len(result.failures), "errors": len(result.errors),
              "passed": result.wasSuccessful(),
              "sources_sha256": {p: hashlib.sha256((ROOT/p).read_bytes()).hexdigest()
                                  for p in ("solution/stage2_entry_refine.py", "solution/stage2_v2.py",
                                            "scripts/test_stage2_entry_refine.py")},
              "synthetic_runs": RECORDS}
    output = ROOT/"research/stage2_entry_refine_contract.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"report": str(output), "passed": result.wasSuccessful(), "tests_run": result.testsRun}))
    sys.exit(0 if result.wasSuccessful() else 1)
