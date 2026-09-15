"""CPU contracts only: does not measure model accuracy or semantic tracking."""
import hashlib
import json
from pathlib import Path
import re
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import cv2
import numpy as np
from PIL import Image
from solution import stage2_v5 as c


class Mock:
    def __init__(self, final=None, fine_status="OBSERVED"):
        self.calls = []
        self.final = final or {"entry_side": "RIGHT", "evasion_space": 1}
        self.fine_status = fine_status

    def ask(self, images, prompt, max_new_tokens=128):
        self.calls.append({"prompt": prompt, "tokens": max_new_tokens,
                           "image_sha": [hashlib.sha256(x.tobytes()).hexdigest() for x in images],
                           "pixels": sum(x.width*x.height for x in images)})
        if "Identify the OTHER" in prompt:
            return '{"status":"UNCERTAIN"}'
        if "Timeline candidates:" in prompt:
            nums = json.loads(re.search(r"Timeline candidates: (\[.*?\])", prompt)[1])
            a, b = len(nums)//2, min(len(nums)-1, len(nums)//2+1)
            return json.dumps(dict(status="BRACKET", before_frame=nums[a], after_frame=nums[b], best_frame=nums[a]))
        if "Offered frames:" in prompt:
            nums = json.loads(re.search(r"Offered frames: (\[.*?\])", prompt)[1])
            return json.dumps(dict(status=self.fine_status, entry_frame=nums[len(nums)//2]))
        return json.dumps(self.final)


def run(numbers=None, model=None, motion=None):
    numbers = numbers if numbers is not None else list(range(70))
    paths = [Path(f"frame_{n}.png") for n in numbers]
    scores = np.zeros(len(paths)); scores[motion if motion is not None else len(paths)-3 if len(paths)>2 else 0] = 10
    model = model or Mock()
    with patch.object(c.base, "_read_rgb", return_value=Image.new("RGB", (640, 360), (71, 92, 103))):
        prediction, diagnostic = c._predict_file(paths, scores, model)
    return prediction, diagnostic, model


class Contracts(unittest.TestCase):
    def test_four_calls_pixels_and_tokens(self):
        _, d, m = run()
        self.assertEqual(len(m.calls), 4)
        self.assertEqual([x["tokens"] for x in m.calls], [64, 96, 48, 48])
        self.assertTrue(all(x["pixels"] <= 1_200_000 for x in m.calls))
        self.assertTrue(all(len(x["tiles"]) <= 12 for x in d["render"]))

    def test_terminal_classification_cannot_change_time_reference_or_inputs(self):
        a, ad, am = run()
        b, bd, bm = run(model=Mock(final=dict(entry_side="LEFT", evasion_space=0,
                    collision_frame=999, entry_frame=999, reference_frame=999, bbox=[0,0,1,1])))
        self.assertEqual(a["collision_frame"], b["collision_frame"])
        self.assertEqual(a["entry_frame"], b["entry_frame"])
        self.assertEqual(ad["reference"], bd["reference"])
        self.assertEqual(am.calls, bm.calls)
        self.assertEqual(ad["coarse"], bd["coarse"])
        self.assertEqual(ad["fine"], bd["fine"])

    def test_sparse_number_and_offset(self):
        a, _, _ = run([3+13*i for i in range(70)])
        b, _, _ = run([203+13*i for i in range(70)])
        for key in ("collision_frame", "entry_frame"):
            self.assertEqual(b[key]-a[key], 200)
        self.assertEqual(a["collision_frame"], 3+13*67)

    def test_shared_model_file_independence(self):
        model = Mock()
        a, _, _ = run(model=model)
        run([5+17*i for i in range(32)], model=model)
        b, _, _ = run(model=model)
        self.assertEqual(a, b)

    def test_single_frame_skips_fine(self):
        p, d, m = run([904])
        self.assertEqual(p["collision_frame"], 904)
        self.assertEqual(p["entry_frame"], 904)
        self.assertEqual(len(m.calls), 3)
        self.assertEqual(d["fine"]["reason"], "single_frame_skip")

    def test_motion_is_not_entry_upper_bound(self):
        p, d, _ = run(motion=2)
        self.assertGreater(p["entry_frame"], p["collision_frame"])
        self.assertTrue(d["entry_after_motion_collision"])
        self.assertEqual(d["fine"]["reason"], "accepted")
        self.assertIn(69, d["coarse"]["candidates"])

    def test_strict_fine_rejection_and_original_start(self):
        paths = [Path(f"f_{n}.png") for n in (40, 80, 200, 300)]
        for value in (True, 80., "80", 81, None, []):
            entry, _ = c._fine_selection(dict(status="OBSERVED", entry_frame=value), paths, [1,2], 0, 0)
            self.assertEqual(entry, 0)
        for status in ("REENTRY", "NOT_FIRST_ENTRY", "BEFORE_WINDOW", "AFTER_WINDOW", "UNCERTAIN"):
            entry, _ = c._fine_selection(dict(status=status, entry_frame=200), paths, [1,2], 3, 1)
            self.assertEqual(entry, 1)
        self.assertEqual(c._fine_selection(dict(status="ALREADY_AT_START", same_target=True, entry_frame=40), paths, [0,2], 3, 2)[0], 0)
        self.assertEqual(c._fine_selection(dict(status="ALREADY_AT_START", same_target=True, entry_frame=200), paths, [0,2], 3, 1)[0], 1)

    def test_invalid_coarse_fallback_global_and_preserved_center(self):
        paths = [Path(f"f_{n}.png") for n in range(300)]
        offered = c.base._uniform_indices(0, 299, 12)
        best, bracket, _ = c._coarse_selection(dict(status="BRACKET", before_frame=299, after_frame=0, best_frame=0), paths, offered, 2)
        self.assertIsNone(bracket)
        self.assertEqual(c._entry_candidates(300, bracket, best), offered)
        fine = c._entry_candidates(300, (100,200), 123)
        self.assertIn(123, fine)
        self.assertIn(0, fine); self.assertIn(299, fine)
        self.assertLessEqual(len(fine), 12)

    def test_bad_json_contract_fallback(self):
        class Broken(Mock):
            def ask(self, images, prompt, max_new_tokens=128):
                super().ask(images, prompt, max_new_tokens)
                return "not json"
        p, d, m = run(model=Broken())
        self.assertEqual(len(m.calls), 4)
        self.assertEqual(p["entry_frame"], 0)
        self.assertEqual(d["entry_origin"], "invalid_fallback_first")

    def test_bbox_validation(self):
        for bad in ([0,0,0,1], [0,0,2,1], [0,0,float("nan"),1], [False,0,1,1], "box", None):
            self.assertIsNone(c._bbox(bad))
        self.assertEqual(c._bbox([.1,.2,.5,.6]), [.1,.2,.5,.6])

    def test_tracking_translates_instead_of_static_bbox(self):
        rng = np.random.default_rng(314)
        previous = rng.integers(0,256,(180,320),dtype=np.uint8)
        current = cv2.warpAffine(previous, np.float32([[1,0,4],[0,1,2]]), (320,180))
        box, reason = c._advance_roi(previous,current,[.2,.2,.6,.7])
        self.assertEqual(reason,"tracked")
        self.assertAlmostEqual(box[0],.2+4/320,delta=.004)
        self.assertAlmostEqual(box[1],.2+2/180,delta=.004)

    def test_tracking_loss_size_change_and_budget(self):
        zero = np.zeros((180,320),dtype=np.uint8)
        self.assertIsNone(c._advance_roi(zero,zero,[.2,.2,.6,.7])[0])
        self.assertEqual(c._advance_roi(zero,zero[:150],[.2,.2,.6,.7])[1],"resolution_change")
        with patch.object(c,"_gray",side_effect=[(zero,(640,360)),(zero,(1280,720)),(zero,(640,360))]):
            boxes, events = c._track_rois([Path("f_0.png"),Path("f_1.png")],1,[.2,.2,.6,.7])
            self.assertEqual(set(boxes),{1})
            self.assertEqual(events[0]["reason"],"resolution_change")
        with patch.object(c,"_gray",side_effect=AssertionError("budget must skip decode")):
            boxes, events = c._track_rois([None]*302,150,[.2,.2,.6,.7])
            self.assertEqual(set(boxes),{150})
            self.assertEqual(events[0]["reason"],"tracking_budget_full_frame_fallback")

    def test_local_start_not_full_start_prompt(self):
        _, _, m = run()
        self.assertIn("may NOT be the start of the full clip", m.calls[2]["prompt"])
        self.assertNotIn("entry_side", m.calls[0]["prompt"])
        self.assertNotIn("entry_side", m.calls[1]["prompt"])

    def test_long_input_hard_tracking_budget_full_timeline(self):
        class Identified(Mock):
            def ask(self, images, prompt, max_new_tokens=128):
                answer = super().ask(images, prompt, max_new_tokens)
                if "Identify the OTHER" in prompt:
                    offered = json.loads(re.search(r"Shown original frame numbers: (\[.*?\])", prompt)[1])
                    return json.dumps(dict(status="IDENTIFIED", reference_frame=offered[0], bbox=[.1,.2,.4,.7]))
                return answer
        with patch.object(c,"_gray",side_effect=AssertionError("long-video tracking must skip decoding")):
            _, d, m = run(list(range(5000)), model=Identified())
        self.assertEqual(len(m.calls),4)
        self.assertIn(4999,d["coarse"]["candidates"])
        self.assertEqual(d["tracking"]["events"][0]["budget"],300)
        self.assertEqual(d["tracking"]["events"][0]["required_steps"],4999)

    def test_empty_unsorted_duplicate_rejected(self):
        with self.assertRaises(ValueError): c._predict_file([],[],Mock())
        for numbers in ([3,2],[2,2]):
            with self.assertRaises(ValueError): run(numbers)
        with self.assertRaises(FileNotFoundError): c.predict_stage2(ROOT/"not_existing_v5_test_dir",ROOT)


if __name__ == "__main__":
    cv2.setNumThreads(2)
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(Contracts)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    frozen = json.loads((ROOT/"research/v5_stage2/preimplementation_freeze.json").read_text())
    protected = {p:h for p,h in frozen["sha256"].items() if not p.startswith("research/")}
    unchanged = all(hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h for p,h in protected.items())
    report = {"passed": result.wasSuccessful() and unchanged, "tests":result.testsRun,
              "cpu_contract_only":True,"gpu_used":False,"protected_sources_unchanged":unchanged,
              "candidate_sha256":hashlib.sha256((ROOT/"solution/stage2_v5.py").read_bytes()).hexdigest()}
    (ROOT/"research/v5_stage2/cpu_contract.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    raise SystemExit(0 if report["passed"] else 1)
