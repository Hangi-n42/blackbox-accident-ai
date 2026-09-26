"""CPU contract tests; no model weights or ground-truth labels needed."""
import sys
import unittest
from pathlib import Path
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'artifacts/submissions/verify_v6/model/stage2/code'))
from temporal_entry import refine_entry
class Answers:
    def __init__(self, answers): self.answers=iter(answers)
    def ask(self,*args,**kwargs): return next(self.answers)
class EntryContract(unittest.TestCase):
    def setUp(self):
        self.paths=[Path(f'frame_{i}.jpg') for i in range(1200)]
        self.pred=dict(collision_frame=1000,entry_frame=500,entry_side='LEFT',evasion_space=0)
    @patch('temporal_entry.v2._sheet',return_value=None)
    def test_zero_and_narrowing(self,_):
        out,detail=refine_entry(self.paths,self.pred,{},Answers(['{"entry_frame":0}']*3))
        self.assertEqual(out,{**self.pred,'entry_frame':0})
        self.assertEqual(detail['calls'],3)
        for a,b in zip(detail['rounds'],detail['rounds'][1:]):
            self.assertLess(b['candidates'][-1]-b['candidates'][0],a['candidates'][-1]-a['candidates'][0])
    @patch('temporal_entry.v2._sheet',return_value=None)
    def test_unoffered_frame_retains_prediction(self,_):
        out,detail=refine_entry(self.paths,self.pred,{},Answers(['{"entry_frame":999999}']))
        self.assertEqual(out,self.pred)
        self.assertFalse(detail['rounds'][0]['accepted'])
if __name__=='__main__':unittest.main()
