"""CPU-only generated-image contracts; no human labels or model execution."""
import hashlib
import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT/'artifacts/submissions/verify_v6/model/stage2/code'))
spec = importlib.util.spec_from_file_location('space_anchor_candidate', ROOT/'research/v7/solution/stage2_space_anchor_v7.py')
candidate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(candidate)


def signature(images):
    return [(im.size, hashlib.sha256(im.convert('RGB').tobytes()).hexdigest()) for im in images]


class Recorder:
    def __init__(self, answers):
        self.answers = iter(answers)
        self.calls = []

    def ask(self, images, prompt, max_new_tokens=128):
        self.calls.append((signature(images), prompt, max_new_tokens))
        return next(self.answers)


class Contracts(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='space_anchor_synthetic_')
        self.addCleanup(self.tmp.cleanup)

    def paths(self, numbers):
        paths = []
        for index, number in enumerate(numbers):
            path = Path(self.tmp.name)/f'frame_{number:06}.png'
            Image.new('RGB', (96, 64), (index*23 % 256, number % 256, 70)).save(path)
            paths.append(path)
        return paths

    def test_actual_frozen_four_calls_and_nonconsecutive_numbers(self):
        paths = self.paths([11, 20, 31, 49, 61, 77, 98])
        base = np.array([0, 1, 9, 2, 1, 0, 0], np.float32)
        new = np.array([0, 1, 9, 2, 1, 0, 40], np.float32)
        answers = ['{"collision_frame":31,"entry_side":"RIGHT"}',
                   '{"collision_frame":31}', '{"entry_frame":11}', '{"evasion_space":0}']
        old_vlm = Recorder(answers)
        old, _ = candidate.frozen._predict_file(paths, base, new, old_vlm)
        new_vlm = Recorder(answers[:-1]+['{"evasion_space":1}'])
        result, diagnostics = candidate._predict_file(paths, base, new, new_vlm)
        self.assertEqual(old_vlm.calls[:3], new_vlm.calls[:3])
        self.assertEqual(old_vlm.calls[3][1:], new_vlm.calls[3][1:])
        self.assertNotEqual(old_vlm.calls[3][0], new_vlm.calls[3][0])
        self.assertEqual(new_vlm.calls[3][0], signature([candidate.base._sheet(paths, [4,6], columns=3)]))
        self.assertEqual(diagnostics['space_anchor']['context_frames'], [61,98])
        self.assertEqual(len(new_vlm.calls), 4)
        self.assertEqual(result['evasion_space'], 1)
        self.assertEqual({k:v for k,v in old.items() if k!='evasion_space'},
                         {k:v for k,v in result.items() if k!='evasion_space'})
        helper_vlm = Recorder(['{"evasion_space":1}'])
        value, detail = candidate.predict_space_only(paths, new, helper_vlm, old_vlm.calls[3][1])
        self.assertEqual(value, 1)
        self.assertEqual(helper_vlm.calls[0], new_vlm.calls[3])
        self.assertEqual(detail['context_frames'], [61,98])

    def test_single_frame_and_fallback(self):
        paths = self.paths([37])
        answers = ['{}','{}','{}','{"evasion_space":true}']
        result, diagnostics = candidate._predict_file(paths, np.zeros(1), np.zeros(1), Recorder(answers))
        self.assertEqual(result, dict(collision_frame=37, entry_frame=37, entry_side='LEFT', evasion_space=0))
        self.assertEqual(diagnostics['space_anchor']['context_frames'], [37])
        self.assertTrue(diagnostics['space_anchor']['fallback'])

    def test_original_parser_acceptance_and_invalid_fallback(self):
        paths = self.paths([10,90,200])
        for raw, expected, fallback in [('{"evasion_space":"1"}',1,False),
                                        ('{"evasion_space":1.0}',1,False),
                                        ('{"evasion_space":2}',0,True), ('not JSON',0,True)]:
            with self.subTest(raw=raw):
                value, detail = candidate.predict_space_only(paths,[9,0,0],Recorder([raw]),'cached fixture prompt')
                self.assertEqual((value,detail['fallback']),(expected,fallback))
                self.assertEqual(detail['context_frames'],[10,200])

    def test_proxy_identity_forwarding_and_budget(self):
        paths = self.paths([4])
        class IdentityRecorder:
            def __init__(self): self.images=[]
            def ask(self, images, prompt, max_new_tokens=128):
                self.images.append(images)
                return '{}'
        recorder = IdentityRecorder()
        proxy = candidate._FourthCallProxy(paths,[0],recorder)
        images = [Image.new('RGB',(2,2))]
        for _ in range(3): proxy.ask(images,'p',48)
        self.assertTrue(all(value is images for value in recorder.images))
        proxy.ask(images,'p',40)
        self.assertIsNot(recorder.images[3],images)
        with self.assertRaises(RuntimeError): proxy.ask(images,'p',40)

    def test_invalid_cap_or_scores_rejected_before_call(self):
        paths = self.paths([9])
        for scores, cap in [([0],41),([float('nan')],40),([],40)]:
            recorder = Recorder([])
            with self.assertRaises(ValueError):
                candidate.predict_space_only(paths,scores,recorder,'cached',cap)
            self.assertEqual(recorder.calls,[])


if __name__ == '__main__':
    unittest.main()
