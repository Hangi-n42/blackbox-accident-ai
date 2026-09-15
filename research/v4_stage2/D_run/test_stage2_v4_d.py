"""D CPU mock contracts; no model accuracy evaluation."""
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
from solution import stage2_v4_d as candidate
from scripts.test_stage2_v4_c import CapturingMock


class SideMock(CapturingMock):
    def ask(self, images, prompt, max_new_tokens=128):
        answer = super().ask(images, prompt, max_new_tokens=max_new_tokens)
        if candidate.ORIGINAL in prompt:
            value = json.loads(answer); value["entry_side"] = "LEFT"
            return json.dumps(value)
        return answer


def run(numbers=None, *, model=None, baseline=False):
    numbers = numbers if numbers is not None else list(range(60))
    paths = [Path(f"f_{n}.jpg") for n in numbers]
    scores = np.zeros(len(paths)); scores[max(0, len(paths)-3)] = 9
    model = model or SideMock()
    with patch.object(candidate.base, "_sheet", return_value=Image.new("RGB", (16, 8))):
        p, d = (candidate.baseline._predict_file if baseline else candidate._predict_file)(paths, scores, model)
    return p, d, model


class Contracts(unittest.TestCase):
    def test_exact_first_sentence_replacement_only(self):
        b, _, bm = run(baseline=True)
        d, diagnostic, dm = run()
        self.assertEqual(dm.calls[0][0], bm.calls[0][0].replace(candidate.ORIGINAL, candidate.REPLACEMENT, 1))
        self.assertEqual(dm.calls[0][1], bm.calls[0][1])
        self.assertEqual(dm.calls[1:], bm.calls[1:])
        self.assertEqual(dm.images, bm.images)
        self.assertEqual(len(dm.calls), 4)
        self.assertEqual(diagnostic["first_question_ablation"]["calls_forwarded"], 4)
        for key in ("collision_frame", "entry_frame", "evasion_space"):
            self.assertEqual(b[key], d[key])
        self.assertEqual(b["entry_side"], "LEFT")
        self.assertEqual(d["entry_side"], "RIGHT")

    def test_collision_request_and_json_fields_retained(self):
        _, _, model = run()
        prompt = model.calls[0][0]
        self.assertIn("Which numbered frame first shows physical contact", prompt)
        self.assertIn("Return JSON with collision_frame and entry_side (LEFT or RIGHT)", prompt)
        self.assertIn("not its direction of travel", prompt)
        self.assertNotIn(candidate.ORIGINAL, prompt)

    def test_wrapper_reinitialized_per_file(self):
        shared = SideMock()
        first, _, _ = run(model=shared)
        run([100+7*n for n in range(60)], model=shared)
        again, _, _ = run(model=shared)
        self.assertEqual(first, again)
        self.assertEqual(sum(candidate.REPLACEMENT in prompt for prompt, _ in shared.calls), 3)

    def test_noncontinuous_original_numbers_and_offset(self):
        numbers = [5+13*n for n in range(60)]
        p, _, _ = run(numbers)
        shifted, _, _ = run([n+301 for n in numbers])
        self.assertEqual(p["collision_frame"], numbers[-3])
        for key in ("collision_frame", "entry_frame"):
            self.assertIn(p[key], numbers)
            self.assertEqual(shifted[key]-p[key], 301)

    def test_single_frame_preserves_budget_and_original_number(self):
        p, _, m = run([907])
        self.assertEqual(len(m.calls), 4)
        self.assertEqual(p["collision_frame"], 907)
        self.assertEqual(p["entry_frame"], 907)

    def test_unexpected_first_prompt_rejected(self):
        mock = SideMock()
        with self.assertRaises(ValueError):
            candidate._FirstQuestion(mock).ask([], "unrelated", 64)
        self.assertEqual(mock.calls, [])

    def test_does_not_rewrite_later_ask(self):
        mock = SideMock()
        wrapper = candidate._FirstQuestion(mock)
        prompt = "Available frames: [0, 5]. "+candidate.ORIGINAL
        wrapper.ask([], prompt, 64)
        wrapper.ask([], prompt, 48)
        self.assertIn(candidate.REPLACEMENT, mock.calls[0][0])
        self.assertEqual(mock.calls[1][0], prompt)


if __name__ == "__main__":
    frozen = json.loads((ROOT/"research/v4_stage2/frozen_D.json").read_text())
    verify = lambda: all(hashlib.sha256((ROOT/name).read_bytes()).hexdigest() == value for name, value in frozen["protected_sha256"].items())
    assert verify()
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Contracts))
    report = {"scope": "CPU mock contracts, not accuracy", "tests": result.testsRun, "passed": result.wasSuccessful(),
              "torch_imported": "torch" in sys.modules, "protected_sources_results_unchanged": verify(),
              "candidate_sha256": hashlib.sha256((ROOT/"solution/stage2_v4_d.py").read_bytes()).hexdigest()}
    (ROOT/"research/v4_stage2/mock_D_contract.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    sys.exit(0 if report["passed"] and report["protected_sources_results_unchanged"] and not report["torch_imported"] else 1)
