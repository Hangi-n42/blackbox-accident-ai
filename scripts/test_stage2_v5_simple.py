"""CPU contracts for the simple candidate; no model accuracy claim."""
import hashlib
import json
from pathlib import Path
import re
import sys
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
from PIL import Image
from solution import stage2_v5_simple as c
from solution import stage2_motion_collision as baseline


class Mock:
    def __init__(self,collision=11,final=None):
        self.calls=[];self.collision=collision;self.final=final or dict(evasion_space=1)
    def ask(self,images,prompt,max_new_tokens=128):
        self.calls.append((prompt,max_new_tokens,[hashlib.sha256(i.tobytes()).hexdigest() for i in images]))
        if "collision_frame and entry_side" in prompt:return json.dumps(dict(collision_frame=self.collision,entry_side="RIGHT"))
        if "entry_frame only" in prompt:
            candidates=json.loads(re.search(r"Lane entry candidates: (\[.*?\])",prompt)[1])
            return json.dumps(dict(entry_frame=candidates[len(candidates)//2]))
        return json.dumps(self.final)


def run(numbers=None,model=None,motion=None):
    numbers=numbers if numbers is not None else list(range(70))
    paths=[Path(f"frame_{n}.png") for n in numbers]
    scores=np.zeros(len(paths));scores[motion if motion is not None else max(0,len(paths)-3)]=5
    model=model or Mock()
    with patch.object(c.base,"_read_rgb",return_value=Image.new("RGB",(640,360))):
        p,d=c._predict_file(paths,scores,model)
    return p,d,model


class Contracts(unittest.TestCase):
    def test_v3_first_call_exact(self):
        _,_,new=run()
        old=Mock()
        paths=[Path(f"frame_{n}.png") for n in range(70)]
        with patch.object(c.base,"_read_rgb",return_value=Image.new("RGB",(640,360))):
            baseline._predict_file(paths,np.arange(70),old)
        self.assertEqual(new.calls[0],old.calls[0])

    def test_first_collision_does_not_propagate(self):
        p,d,m=run(model=Mock(collision=0))
        q,e,n=run(model=Mock(collision=999999))
        self.assertEqual(p,q)
        self.assertEqual(m.calls,n.calls)
        for k in ("coarse_candidates","fine_candidates","space_context"):
            self.assertEqual(d[k],e[k])

    def test_last_call_cannot_modify_times(self):
        a,d,m=run()
        b,e,n=run(model=Mock(final=dict(evasion_space=0,entry_frame=900,collision_frame=800,entry_side="LEFT")))
        for k in ("collision_frame","entry_frame","entry_side"):self.assertEqual(a[k],b[k])
        self.assertEqual(m.calls,n.calls)

    def test_call_token_budget_and_entire_timeline(self):
        p,d,m=run(motion=2)
        self.assertEqual(len(m.calls),4)
        self.assertEqual(sum(x[1] for x in m.calls),192)
        self.assertGreater(p["entry_frame"],p["collision_frame"])
        self.assertEqual(d["coarse_candidates"][-1],69)
        self.assertEqual(d["space_context"],[0,2,4])

    def test_sparse_and_offset(self):
        numbers=[5+17*i for i in range(70)]
        a,_,_=run(numbers);b,_,_=run([i+200 for i in numbers])
        for key in ("entry_frame","collision_frame"):
            self.assertIn(a[key],numbers);self.assertEqual(b[key]-a[key],200)

    def test_file_independence_single_and_long(self):
        m=Mock();a,_,_=run(model=m);run([2+19*i for i in range(20)],model=m);b,_,_=run(model=m)
        self.assertEqual(a,b)
        p,d,m=run([904]);self.assertEqual(p["entry_frame"],904);self.assertEqual(len(m.calls),4)
        _,d,_=run(list(range(5000)));self.assertIn(4999,d["fine_candidates"]);self.assertLessEqual(len(d["fine_candidates"]),12)

    def test_fine_preserves_coarse_and_anchors(self):
        offered=c.base._uniform_indices(0,999,12)
        coarse=offered[6]
        fine=c._fine_indices(1000,offered,coarse,True)
        self.assertTrue({0,999,coarse}.issubset(fine));self.assertLessEqual(len(fine),12)
        self.assertEqual(c._fine_indices(1000,offered,0,False),offered)

    def test_invalid_response_keeps_valid_coarse(self):
        paths=[Path(f"f_{i}.png") for i in (3,9,24)]
        for bad in (True,"9",9.,99,None,[]):
            self.assertEqual(c._select(dict(entry_frame=bad),paths,[0,1,2],1),(1,False))
        self.assertEqual(c._select(dict(entry_frame=9),paths,[0,1,2],0),(1,True))

    def test_all_invalid_distinct_from_valid_first(self):
        class Broken(Mock):
            def ask(self,images,prompt,max_new_tokens=128):
                super().ask(images,prompt,max_new_tokens)
                return "not JSON"
        p,d,_=run(model=Broken());self.assertEqual(p["entry_frame"],0)
        self.assertEqual(d["entry_origin"],"invalid_fallback_first")
        self.assertFalse(d["valid_model_selected_first"])
        class First(Mock):
            def ask(self,images,prompt,max_new_tokens=128):
                answer=super().ask(images,prompt,max_new_tokens)
                return '{"entry_frame":0}' if "entry_frame only" in prompt else answer
        _,d,_=run(model=First());self.assertTrue(d["valid_model_selected_first"])
        self.assertEqual(d["entry_origin"],"fine_valid_selected")

    def test_local_first_is_not_clip_start_and_invalid_paths(self):
        _,_,m=run()
        self.assertIn("may not be the start of the full clip",m.calls[2][0])
        self.assertNotIn("bbox",m.calls[2][0]);self.assertNotIn("status",m.calls[2][0])
        with self.assertRaises(ValueError):c._predict_file([],[],Mock())
        for n in ([2,1],[1,1]):
            with self.assertRaises(ValueError):run(n)
        with self.assertRaises(FileNotFoundError):c.predict_stage2(ROOT/"not_existing_simple",ROOT)


if __name__=="__main__":
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Contracts))
    report=dict(passed=result.wasSuccessful(),tests=result.testsRun,cpu_contract_only=True,GPU_used=False,
                candidate_sha256=hashlib.sha256((ROOT/"solution/stage2_v5_simple.py").read_bytes()).hexdigest())
    (ROOT/"research/v5_stage2/cpu_simple_contract.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    raise SystemExit(0 if result.wasSuccessful() else 1)
