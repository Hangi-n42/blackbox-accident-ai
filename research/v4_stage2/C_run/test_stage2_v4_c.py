"""CPU-only C state/number contracts; not model accuracy."""
import hashlib
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from PIL import Image
from solution import stage2_v4_c as candidate
from scripts.test_stage2_v4 import MockVLM


class CapturingMock(MockVLM):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.images = []

    def ask(self, images, prompt, max_new_tokens=128):
        self.images.append([image.size for image in images])
        return super().ask(images, prompt, max_new_tokens=max_new_tokens)


def run(numbers=None, model=None, arm="C"):
    numbers = numbers if numbers is not None else list(range(60))
    paths = [Path(f"f_{n}.jpg") for n in numbers]
    scores = np.zeros(len(paths)); scores[max(0, len(paths)-3)] = 9
    model = model or CapturingMock()
    with patch.object(candidate.base, "_sheet", return_value=Image.new("RGB", (64, 32))), \
            patch.object(candidate.base, "_read_rgb", return_value=Image.new("RGB", (80, 40))):
        prediction, d = candidate._predict_file(paths, scores, model) if arm == "C" else candidate.ab._predict_file(paths, scores, model, variant="A")
    return prediction, d, model


class Contracts(unittest.TestCase):
    def test_a_first_three_calls_fields_and_images_preserved(self):
        a, _, am = run(arm="A")
        c, _, cm = run()
        self.assertEqual(am.calls, cm.calls[:3])
        self.assertEqual(am.images, cm.images[:3])
        for key in ("collision_frame", "entry_side", "evasion_space"):
            self.assertEqual(a[key], c[key])
        self.assertEqual(len(cm.calls), 4)
        self.assertEqual(cm.images[-1], [(80, 40), (64, 32)])

    def test_start_accepts_original_number_not_literal_zero(self):
        for first in (0, 301):
            p, d, m = run(list(range(first, first+60)), CapturingMock(fine_status="ALREADY_IN_LANE_AT_START", fine_value=first))
            self.assertEqual(p["entry_frame"], first)
            self.assertEqual(d["entry_refinement"]["accepted_rule"], "full_clip_start")
            self.assertIn(f"numbered {first}", m.calls[-1][0])

    def test_wrong_start_number_string_boolean_float_rejected(self):
        a, _, _ = run(arm="A")
        for value in (1, "0", False, 0.0):
            p, d, _ = run(model=CapturingMock(fine_status="ALREADY_IN_LANE_AT_START", fine_value=value))
            self.assertEqual(p, a)
            self.assertFalse(d["entry_refinement"]["accepted"])

    def test_observed_only_local_candidates(self):
        a, _, _ = run(arm="A")
        p, d, _ = run(model=CapturingMock(fine_status="OBSERVED", fine_value=0))
        self.assertNotIn(0, d["entry_refinement"]["fine_candidates"])
        self.assertEqual(p, a)
        p, d, _ = run()
        self.assertTrue(d["entry_refinement"]["accepted"])
        self.assertIn(p["entry_frame"], d["entry_refinement"]["fine_candidates"])

    def test_before_after_uncertain_never_authorize_first_frame(self):
        a, _, _ = run(arm="A")
        for status in ("BEFORE_WINDOW", "AFTER_WINDOW", "UNCERTAIN", "already_in_lane_at_start"):
            p, _, _ = run(model=CapturingMock(fine_status=status, fine_value=0))
            self.assertEqual(p, a)

    def test_invalid_json_and_call_error_preserve(self):
        a, _, _ = run(arm="A")
        for raw in ("[]", "bad", "{}", 'prefix {"status":"ALREADY_IN_LANE_AT_START","entry_frame":0}'):
            p, _, _ = run(model=CapturingMock(fine_status="RAW", fine_value=raw))
            self.assertEqual(p, a)
        p, d, _ = run(model=CapturingMock(fine_value=RuntimeError("mock")))
        self.assertEqual(p, a)
        self.assertEqual(d["entry_refinement"]["fallback_reason"], "refinement_call_failed")

    def test_single_and_already_complete_skip(self):
        for numbers in ([300], list(range(12))):
            _, d, m = run(numbers)
            self.assertEqual(len(m.calls), 3)
            self.assertEqual(d["entry_refinement"]["skip_reason"], "all_precontact_frames_already_shown")

    def test_invalid_coarse_skip(self):
        _, d, m = run(model=CapturingMock(coarse_bad=True))
        self.assertEqual(len(m.calls), 3)
        self.assertEqual(d["entry_refinement"]["skip_reason"], "invalid_or_unshown_coarse_entry")

    def test_sparse_numbers_shift_and_independence(self):
        numbers = [7+11*n for n in range(60)]
        p, _, _ = run(numbers)
        shifted, _, _ = run([n+300 for n in numbers])
        for key in ("collision_frame", "entry_frame"):
            self.assertEqual(shifted[key]-p[key], 300)
            self.assertIn(p[key], numbers)
        shared = CapturingMock()
        first, _, _ = run(model=shared)
        run(numbers, shared)
        again, _, _ = run(model=shared)
        self.assertEqual(first, again)

    def test_local_start_distinct_from_full_start(self):
        _, d, _ = run()
        lower = d["entry_refinement"]["fine_candidates"][0]
        self.assertGreater(lower, 0)
        p, d, m = run(model=CapturingMock(fine_value=lower))
        self.assertEqual(p["entry_frame"], lower)
        self.assertTrue(d["entry_refinement"]["selected_local_window_start"])
        self.assertIn("first tile may not be the start", m.calls[-1][0])
        self.assertIn("Do not substitute a different vehicle", m.calls[-1][0])


if __name__ == "__main__":
    frozen = json.loads((ROOT/"research/v4_stage2/frozen_C.json").read_text())
    verify = lambda: all(hashlib.sha256((ROOT/name).read_bytes()).hexdigest() == value for name, value in frozen["protected_sha256"].items())
    assert verify()
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Contracts))
    report = {"scope": "CPU mock contracts, not accuracy", "tests": result.testsRun, "passed": result.wasSuccessful(),
              "torch_imported": "torch" in sys.modules, "protected_C_and_AB_unchanged": verify(),
              "candidate_sha256": hashlib.sha256((ROOT/"solution/stage2_v4_c.py").read_bytes()).hexdigest()}
    (ROOT/"research/v4_stage2/mock_C_contract.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    sys.exit(0 if report["passed"] and report["protected_C_and_AB_unchanged"] and not report["torch_imported"] else 1)
