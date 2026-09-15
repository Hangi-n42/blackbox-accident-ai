"""CPU mocked contracts only; no learned-model accuracy or GPU evaluation."""
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from PIL import Image
from solution import stage2_motion_collision as candidate


class ScriptedVLM:
    def __init__(self, *args, **kwargs):
        self.calls = []

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def ask(self, images, prompt, max_new_tokens=128):
        self.calls.append((prompt, max_new_tokens))
        # Out-of-range answers exercise the existing base snapping policy.
        return ('{"collision_frame": 999999, "entry_side": "RIGHT"}',
                '{"collision_frame": 999999}', '{"entry_frame": 999999}',
                '{"evasion_space": 1}')[(len(self.calls)-1) % 4]


def run(numbers, scores):
    paths = [Path(f"frame_{number}.jpg") for number in numbers]
    baseline, model = ScriptedVLM(), ScriptedVLM()
    with patch.object(candidate.base, "_sheet", return_value=Image.new("RGB", (8, 8))):
        expected, _ = candidate.base._predict_file(paths, np.asarray(scores), baseline)
        actual, diagnostics = candidate._predict_file(paths, np.asarray(scores), model)
    return expected, actual, diagnostics, baseline, model


class MotionCollisionContracts(unittest.TestCase):
    def test_only_collision_changes_and_four_calls_identical(self):
        expected, actual, diagnostic, baseline, model = run([10, 23, 85, 140], [0, 9, 2, 1])
        self.assertEqual(actual["collision_frame"], 23)
        self.assertNotEqual(expected["collision_frame"], actual["collision_frame"])
        for key in ("entry_frame", "entry_side", "evasion_space"):
            self.assertEqual(actual[key], expected[key])
        self.assertEqual(len(model.calls), 4)
        self.assertEqual(model.calls, baseline.calls)
        self.assertEqual(diagnostic["collision_replacement"]["base_collision_frame"], expected["collision_frame"])
        # No entry clamp is introduced when motion replacement precedes entry.
        self.assertGreater(actual["entry_frame"], actual["collision_frame"])

    def test_original_number_offset_equivariance(self):
        _, first, _, _, _ = run([7, 18, 43, 88], [0, 3, 7, 1])
        _, shifted, _, _, _ = run([1007, 1018, 1043, 1088], [0, 3, 7, 1])
        for key in ("collision_frame", "entry_frame"):
            self.assertEqual(shifted[key]-first[key], 1000)
        self.assertEqual(first["collision_frame"], 43)

    def test_tie_uses_existing_numpy_first_argmax(self):
        _, actual, _, _, _ = run([5, 16, 60], [0, 4, 4])
        self.assertEqual(actual["collision_frame"], 16)

    def test_single_frame(self):
        expected, actual, _, _, model = run([901], [0])
        self.assertEqual(actual, expected)
        self.assertEqual(actual["collision_frame"], 901)
        self.assertEqual(len(model.calls), 4)

    def test_separate_inputs_have_no_answer_state(self):
        _, before, _, _, first_model = run([1, 8, 30], [0, 9, 1])
        run([400, 800, 900], [0, 1, 9])
        _, after, _, _, second_model = run([1, 8, 30], [0, 9, 1])
        self.assertEqual(before, after)
        self.assertEqual(first_model.calls, second_model.calls)

    def test_folder_sort_filter_valid_path_alignment_and_independence(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for name in ("B", "A"):
                folder = root/"images"/name
                folder.mkdir(parents=True)
                for number in (80, 10, 32):
                    (folder/f"frame_{number}.jpg").touch()
                (folder/"notes.txt").touch()
            scanned = []
            def scan(paths):
                scanned.append([candidate.base._frame_number(path) for path in paths])
                # Simulate unreadable middle image, preserving returned path alignment.
                kept = [paths[0], paths[2]]
                return kept, np.array([0, 7] if paths[0].parent.name == "A" else [7, 0]), "LEFT"
            model = ScriptedVLM()
            with patch.object(candidate, "CandidateVLM", return_value=model) as factory, \
                    patch.object(candidate.base, "_motion_scan", side_effect=scan), \
                    patch.object(candidate.base, "_sheet", return_value=Image.new("RGB", (8, 8))):
                result = candidate.predict_stage2(root, root/"model")
            factory.assert_called_once_with(root/"model"/"vlm", precision="nf4")
            self.assertEqual(scanned, [[10, 32, 80], [10, 32, 80]])
            self.assertEqual(result.ID.tolist(), ["A", "B"])
            self.assertEqual(result.collision_frame.tolist(), [80, 10])
            self.assertEqual(result.entry_frame.tolist(), [80, 80])
            self.assertEqual(result.entry_side.tolist(), ["RIGHT", "RIGHT"])
            self.assertEqual(result.evasion_space.tolist(), [1, 1])
            self.assertEqual(len(model.calls), 8)
            self.assertEqual(result.columns.tolist(), candidate.COLUMNS)

    def test_missing_images_fails_before_model_load(self):
        with tempfile.TemporaryDirectory() as temporary, patch.object(candidate, "CandidateVLM") as factory:
            with self.assertRaises(FileNotFoundError):
                candidate.predict_stage2(temporary, temporary)
            factory.assert_not_called()

    def test_empty_image_root_has_existing_empty_table_behavior(self):
        with tempfile.TemporaryDirectory() as temporary:
            (Path(temporary)/"images").mkdir()
            with patch.object(candidate, "CandidateVLM", return_value=ScriptedVLM()):
                result = candidate.predict_stage2(temporary, temporary)
            self.assertTrue(result.empty)
            self.assertEqual(result.columns.tolist(), candidate.COLUMNS)

    def test_empty_or_unreadable_folder_raises_existing_error(self):
        for unreadable in (False, True):
            with self.subTest(unreadable=unreadable), tempfile.TemporaryDirectory() as temporary:
                folder = Path(temporary)/"images"/"A"
                folder.mkdir(parents=True)
                if unreadable:
                    (folder/"frame_10.jpg").write_text("not an image")
                with patch.object(candidate, "CandidateVLM", return_value=ScriptedVLM()):
                    with self.assertRaisesRegex(ValueError, "no decodable numbered images"):
                        candidate.predict_stage2(temporary, temporary)

    def test_duplicate_numbers_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)/"images"/"A"
            folder.mkdir(parents=True)
            for name in ("frame_10.jpg", "other_10.png"):
                (folder/name).touch()
            with patch.object(candidate, "CandidateVLM", return_value=ScriptedVLM()), \
                    patch.object(candidate.base, "_motion_scan") as scan:
                with self.assertRaisesRegex(ValueError, "Duplicate Stage2 frame number"):
                    candidate.predict_stage2(temporary, temporary)
                scan.assert_not_called()


if __name__ == "__main__":
    frozen = [ROOT/"solution"/name for name in ("stage2.py", "stage2_v2.py", "stage2_nf4.py", "vlm_candidate.py", "vlm.py")]
    hash_files = lambda: {path.relative_to(ROOT).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest() for path in frozen}
    before = hash_files()
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(MotionCollisionContracts))
    report = {"kind": "CPU mocked contract checks; not accuracy evaluation", "tests_run": result.testsRun,
              "passed": result.wasSuccessful(), "torch_imported": "torch" in sys.modules,
              "protected_sources_before": before, "protected_sources_after": hash_files()}
    report["protected_sources_unchanged"] = before == report["protected_sources_after"]
    report["source_sha256"] = {path.relative_to(ROOT).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
                              for path in (ROOT/"solution/stage2_motion_collision.py", Path(__file__))}
    (ROOT/"research/stage2_motion_collision_contract.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    sys.exit(0 if result.wasSuccessful() and report["protected_sources_unchanged"] and not report["torch_imported"] else 1)
