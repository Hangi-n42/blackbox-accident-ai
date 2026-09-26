"""Independent NumPy reference checks for visual feature injection."""
import unittest
import numpy as np
import mlx.core as mx
from deepstack_fix import deepstack_process
class DeepstackContract(unittest.TestCase):
    def test_distinct_visual_positions_only(self):
        for mask in [np.array([[False,True,True,False]]),np.array([[False,True,False,True],[True,False,False,True]])]:
            original=np.arange(mask.size*3,dtype=np.float32).reshape(*mask.shape,3)
            visual=np.arange(mask.sum()*3,dtype=np.float32).reshape(-1,3)+100
            expected=original.copy();expected[mask]+=visual
            result=deepstack_process(None,mx.array(original),mx.array(mask),mx.array(visual))
            np.testing.assert_array_equal(np.asarray(result),expected)
    def test_mismatched_feature_count_rejected(self):
        with self.assertRaises(ValueError):
            deepstack_process(None,mx.zeros((1,4,3)),mx.array([[False,True,False,False]]),mx.zeros((2,3)))
if __name__=='__main__':unittest.main()
