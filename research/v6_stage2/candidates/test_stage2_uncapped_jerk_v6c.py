"""CPU-only numeric/image contracts. Synthetic fixtures are not evaluation footage."""
import copy
import importlib.util
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='2'
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
spec=importlib.util.spec_from_file_location('solution.stage2_uncapped_jerk_v6c',Path(__file__).with_name('stage2_uncapped_jerk_v6c.py'))
candidate=importlib.util.module_from_spec(spec);spec.loader.exec_module(candidate)
import numpy as np
from PIL import Image
candidate.primitives.cv2.setNumThreads(2)


class FakeVLM:
    def __init__(self):self.calls=[]
    def ask(self,images,prompt,max_new_tokens):
        self.calls.append((prompt,max_new_tokens))
        return ['{"collision_frame":11,"entry_side":"RIGHT"}', '{"collision_frame":11}',
                '{"entry_frame":11}', '{"evasion_space":1}'][len(self.calls)-1]


class Contracts(unittest.TestCase):
    def setUp(self):
        temp=tempfile.TemporaryDirectory();self.addCleanup(temp.cleanup)
        self.folder=Path(temp.name);self.paths=[]
        rng=np.random.default_rng(19);base=rng.integers(0,256,(96,160,3),dtype=np.uint8)
        for i,shift in enumerate([0,1,2,5,4,8,8,9,13,13,15,18]):
            p=self.folder/f'frame_{11+i*7:06d}.png'
            Image.fromarray(np.roll(base,shift,axis=1)).save(p);self.paths.append(p)

    def test_real_synthetic_images_base_scores_bitwise_original(self):
        old_paths,old_scores,_=candidate.primitives._motion_scan(self.paths)
        valid,base,new,features=candidate._dual_motion_scan(self.paths)
        self.assertEqual(valid,old_paths)
        self.assertEqual(base.dtype,old_scores.dtype)
        self.assertEqual(base.tobytes(),old_scores.tobytes())
        self.assertEqual(features.shape,(len(valid),3))
        self.assertEqual(base[0],0);self.assertEqual(new[0],0)

    def test_numeric_upper_cap_tie_removed_only_for_jerk(self):
        # Median/MAD zero -> denominator .001; jerk peaks .02/.04 both cap at10.
        features=np.zeros((9,3),dtype=np.float32)
        features[6,0]=.02;features[7,0]=.04
        base,new=candidate._scores_from_features(features)
        self.assertEqual(base[6],base[7]);self.assertEqual(base[6],10)
        self.assertEqual(int(np.argmax(base)),6);self.assertEqual(int(np.argmax(new)),7)
        self.assertGreater(new[7],new[6])
        features[6,1:]=[100,100];features[7,1:]=[200,200]
        base2,new2=candidate._scores_from_features(features)
        np.testing.assert_array_equal(base2-base,new2-new)
        np.testing.assert_allclose(base2[6]-base[6],8.5,rtol=0,atol=1e-6)

    def test_single_frame_and_bad_image_match_original(self):
        bad=self.folder/'frame_000019.png';bad.write_bytes(b'not an image')
        paths=[self.paths[0],bad,self.paths[1]]
        a,b,_=candidate.primitives._motion_scan(paths)
        valid,base,_,_=candidate._dual_motion_scan(paths)
        self.assertEqual(a,valid);self.assertEqual(b.tobytes(),base.tobytes())
        valid,base,new,_=candidate._dual_motion_scan(self.paths[:1])
        self.assertEqual(base.tolist(),[0]);self.assertEqual(new.tolist(),[0])
        with self.assertRaises(ValueError):candidate._dual_motion_scan([bad])

    def test_no_upper_cap_active_scores_equal(self):
        f=np.array([[0,0,0],[1,3,4],[2,4,3],[3,6,5],[4,2,2]],dtype=np.float32)
        base,new=candidate._scores_from_features(f)
        self.assertEqual(base.tobytes(),new.tobytes())

    def test_cached_helper_original_numbers_hashes_and_other_fields(self):
        pred={'collision_frame':11,'entry_frame':18,'entry_side':'LEFT','evasion_space':0}
        diag={'original':{'value':7}};before=copy.deepcopy((pred,diag))
        base=np.array([4,3,2],dtype=np.float32);new=np.array([4,3,8],dtype=np.float32)
        result,out=candidate.apply_collision(self.paths[:3],pred,diag,new,base)
        self.assertEqual(result['collision_frame'],25)
        self.assertEqual((pred,diag),before)
        self.assertTrue(out['uncapped_jerk']['changed_target_only'])
        self.assertEqual(out['uncapped_jerk']['base_score_sha256'],candidate._score_hash(base))
        self.assertEqual(out['uncapped_jerk']['new_score_sha256'],candidate._score_hash(new))
        for field in ('entry_frame','entry_side','evasion_space'):self.assertEqual(result[field],pred[field])

    def test_actual_baseline_four_fake_calls_same_other_outputs(self):
        base=np.zeros(len(self.paths),dtype=np.float32);base[4]=10
        new=base.copy();new[8]=50
        oldvlm,newvlm=FakeVLM(),FakeVLM()
        with patch.object(candidate.baseline.base,'_sheet',return_value='CPU_FAKE_SHEET'):
            original,diag=candidate.baseline._predict_file(self.paths,base,oldvlm)
            result,out=candidate._predict_file(self.paths,base,new,newvlm)
        self.assertEqual(len(newvlm.calls),4)
        self.assertEqual(oldvlm.calls,newvlm.calls)
        self.assertEqual(result['collision_frame'],67)
        for field in ('entry_frame','entry_side','evasion_space'):self.assertEqual(result[field],original[field])
        self.assertEqual({k:v for k,v in out.items() if k!='uncapped_jerk'},diag)


if __name__=='__main__':unittest.main()
