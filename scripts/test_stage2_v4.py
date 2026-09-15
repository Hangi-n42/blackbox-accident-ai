"""CPU synthetic contracts only, not accuracy evaluation."""
import hashlib
import json
from pathlib import Path
import re
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from PIL import Image
from solution import stage2_v4 as candidate
from solution import stage2_motion_collision as baseline


class MockVLM:
    def __init__(self, fine_status="OBSERVED", fine_value=None, coarse_bad=False):
        self.calls = []
        self.fine_status, self.fine_value, self.coarse_bad = fine_status, fine_value, coarse_bad

    def ask(self, images, prompt, max_new_tokens=128):
        self.calls.append((prompt, max_new_tokens))
        match = re.search(r"(?:Available frames|Allowed frames|Lane entry candidates): (\[[^\]]+\])", prompt)
        numbers = json.loads(match[1]) if match else []
        if "Return JSON with status" in prompt:
            if isinstance(self.fine_value, Exception):
                raise self.fine_value
            if self.fine_status == "RAW":
                return self.fine_value
            value = self.fine_value if self.fine_value is not None else numbers[min(len(numbers)-1, len(numbers)//2+1)]
            return json.dumps({"status": self.fine_status, "entry_frame": value})
        if "entry_frame only" in prompt:
            return '{"entry_frame": "unknown"}' if self.coarse_bad else json.dumps({"entry_frame": numbers[len(numbers)//2]})
        if "evasion_space" in prompt:
            return '{"evasion_space": 1}'
        return json.dumps({"collision_frame": numbers[-1], "entry_side": "RIGHT"})


def run(numbers, variant="B", model=None):
    paths = [Path(f"frame_{n}.jpg") for n in numbers]
    scores = np.zeros(len(paths), dtype=np.float32)
    scores[max(0, len(paths)-3)] = 9
    model = model or MockVLM()
    with patch.object(candidate.base, "_sheet", return_value=Image.new("RGB", (8, 8))):
        result, diagnostic = candidate._predict_file(paths, scores, model, variant=variant)
    return result, diagnostic, model


class Contracts(unittest.TestCase):
    def test_a_three_calls_and_motion_context(self):
        prediction, d, model = run(list(range(60)), "A")
        self.assertEqual(len(model.calls), 3)
        self.assertEqual(prediction["collision_frame"], 57)
        self.assertEqual(d["collision_context_frame"], 57)
        self.assertEqual(d["entry_candidates"][-1], 57)
        self.assertEqual(d["space_context_frames"], [55, 57, 59])

    def test_b_only_entry_changes_and_initial_calls_identical(self):
        a, ad, am = run(list(range(60)), "A")
        b, bd, bm = run(list(range(60)), "B")
        for key in ("collision_frame", "entry_side", "evasion_space"):
            self.assertEqual(a[key], b[key])
        self.assertEqual(am.calls, bm.calls[:3])
        self.assertEqual(len(bm.calls), 4)
        self.assertTrue(bd["entry_refinement"]["accepted"])
        self.assertIn(a["entry_frame"], bd["entry_refinement"]["fine_candidates"])

    def test_overview_and_space_prompt_match_frozen_base(self):
        paths = [Path(f"frame_{n}.jpg") for n in range(60)]
        scores = np.zeros(60); scores[57] = 9
        model = MockVLM()
        with patch.object(candidate.base, "_sheet", return_value=Image.new("RGB", (8, 8))):
            old, _ = baseline._predict_file(paths, scores, model)
        new, _, fresh = run(list(range(60)), "A")
        self.assertEqual(model.calls[0], fresh.calls[0])
        self.assertEqual(model.calls[3], fresh.calls[2])
        self.assertEqual(old["collision_frame"], new["collision_frame"])
        self.assertEqual(old["entry_side"], new["entry_side"])

    def test_noncontinuous_original_numbers_offset(self):
        numbers = [7+11*n for n in range(100)]
        result, d, _ = run(numbers)
        shifted, sd, _ = run([n+700 for n in numbers])
        for key in ("collision_frame", "entry_frame"):
            self.assertEqual(shifted[key]-result[key], 700)
            self.assertIn(result[key], numbers)
        self.assertEqual([n+700 for n in d["entry_refinement"]["fine_candidates"]], sd["entry_refinement"]["fine_candidates"])

    def test_long_timeline_window_center_bounds_and_budget(self):
        _, d, m = run(list(range(600)))
        f = d["entry_refinement"]
        self.assertLessEqual(len(f["fine_candidates"]), 12)
        self.assertIn(f["coarse_entry_frame"], f["fine_candidates"])
        self.assertEqual(f["fine_candidates"][0], f["interval_frames"][0])
        self.assertEqual(f["fine_candidates"][-1], f["interval_frames"][-1])
        self.assertEqual(len(m.calls), 4)

    def test_single_and_dense_small_skip(self):
        for numbers in ([1009], list(range(12))):
            _, d, m = run(numbers)
            self.assertEqual(len(m.calls), 3)
            self.assertEqual(d["entry_refinement"]["skip_reason"], "all_precontact_frames_already_shown")

    def test_no_new_frames_skip(self):
        with patch.object(candidate, "_refinement_indices", return_value=([], "no_new_frames_in_selected_interval", (10, 11))):
            _, d, m = run(list(range(60)))
        self.assertEqual(len(m.calls), 3)
        self.assertEqual(d["entry_refinement"]["skip_reason"], "no_new_frames_in_selected_interval")

    def test_invalid_coarse_skips(self):
        _, d, m = run(list(range(60)), model=MockVLM(coarse_bad=True))
        self.assertEqual(len(m.calls), 3)
        self.assertEqual(d["entry_refinement"]["skip_reason"], "invalid_or_unshown_coarse_entry")

    def test_wrong_window_and_uncertain_preserve_coarse(self):
        a, _, _ = run(list(range(60)), "A")
        for status in ("BEFORE_WINDOW", "AFTER_WINDOW", "UNCERTAIN", "observed"):
            with self.subTest(status=status):
                result, d, _ = run(list(range(60)), model=MockVLM(fine_status=status))
                self.assertEqual(a, result)
                self.assertFalse(d["entry_refinement"]["accepted"])

    def test_observed_must_be_integer_shown_and_json_only(self):
        a, _, _ = run(list(range(60)), "A")
        bad = ["{}", "[]", "broken", '{"status":"OBSERVED","entry_frame":true}',
               '{"status":"OBSERVED","entry_frame":30.5}', '{"status":"OBSERVED","entry_frame":"30"}',
               '{"status":"OBSERVED","entry_frame":99999}',
               'prefix {"status":"OBSERVED","entry_frame":30}']
        for value in bad:
            result, d, _ = run(list(range(60)), model=MockVLM(fine_status="RAW", fine_value=value))
            self.assertEqual(result, a)
            self.assertFalse(d["entry_refinement"]["accepted"])

    def test_refinement_exception_preserves_result(self):
        a, _, _ = run(list(range(60)), "A")
        b, d, _ = run(list(range(60)), model=MockVLM(fine_value=RuntimeError("mock")))
        self.assertEqual(a, b)
        self.assertEqual(d["entry_refinement"]["fallback_reason"], "refinement_call_failed")

    def test_window_start_is_not_full_clip_start(self):
        _, d, _ = run(list(range(60)))
        lower = d["entry_refinement"]["fine_candidates"][0]
        self.assertGreater(lower, 0)
        p, d, m = run(list(range(60)), model=MockVLM(fine_value=lower))
        self.assertEqual(p["entry_frame"], lower)
        self.assertTrue(d["entry_refinement"]["selected_local_window_start"])
        self.assertIn("first image may not be the start", m.calls[-1][0])
        self.assertIn("BEFORE_WINDOW", m.calls[-1][0])

    def test_pointwise_independence_shared_model(self):
        shared = MockVLM()
        before, _, _ = run(list(range(60)), model=shared)
        run(list(range(1000, 1100)), model=shared)
        after, _, _ = run(list(range(60)), model=shared)
        self.assertEqual(before, after)

    def test_invalid_arrays(self):
        for paths, scores in (([], []), ([Path("f_2.jpg"), Path("f_1.jpg")], [0, 1]),
                              ([Path("f_1.jpg"), Path("g_1.jpg")], [0, 1]),
                              ([Path("f_1.jpg")], []), ([Path("f_1.jpg")], [float("nan")])):
            with self.assertRaises(ValueError):
                candidate._predict_file(paths, scores, MockVLM())

    def test_missing_images_and_variant(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(candidate, "CandidateVLM") as loader:
            with self.assertRaises(FileNotFoundError):
                candidate.predict_stage2(directory, directory)
            with self.assertRaises(ValueError):
                candidate.predict_stage2(directory, directory, variant="C")
            loader.assert_not_called()


if __name__ == "__main__":
    frozen = json.loads((ROOT/"research/v4_stage2/frozen_inputs.json").read_text())
    verify = lambda: all(hashlib.sha256((ROOT/name).read_bytes()).hexdigest() == value for name, value in frozen["protected_sha256"].items())
    assert verify()
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Contracts))
    report = {"scope": "CPU mocked contracts, not accuracy", "tests": result.testsRun, "passed": result.wasSuccessful(),
              "torch_imported": "torch" in sys.modules, "protected_sources_unchanged": verify(),
              "candidate_sha256": hashlib.sha256((ROOT/"solution/stage2_v4.py").read_bytes()).hexdigest(),
              "test_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (ROOT/"research/v4_stage2/mock_contract.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    sys.exit(0 if report["passed"] and report["protected_sources_unchanged"] and not report["torch_imported"] else 1)
