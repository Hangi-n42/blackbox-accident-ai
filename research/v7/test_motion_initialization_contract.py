"""CPU-only synthetic contracts. No real images, labels, VLM, or accuracy test."""
import os
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[name]='2'
os.environ['CUDA_VISIBLE_DEVICES']=''
import sys
sys.dont_write_bytecode=True
import hashlib,importlib.util,json,types,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parent
SOURCE=OUT/'solution/stage2_motion_init_v7.py'
FROZEN=ROOT/'artifacts/submissions/verify_v6/model/stage2/code/solution'
package=types.ModuleType('motion_init_contract_v6');package.__path__=[str(FROZEN)]
sys.modules[package.__name__]=package
spec=importlib.util.spec_from_file_location(package.__name__+'.candidate',SOURCE)
candidate=importlib.util.module_from_spec(spec);sys.modules[spec.name]=candidate;spec.loader.exec_module(candidate)
v6=candidate.v6

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def bits(a,b):return a.dtype==b.dtype and a.shape==b.shape and a.tobytes()==b.tobytes()
BEFORE={str(p.relative_to(ROOT)):sha(p) for p in [SOURCE,FROZEN/'stage2_uncapped_jerk_v6c.py',FROZEN/'stage2.py']}

class Contracts(unittest.TestCase):
    def check_features(self,f):
        f=np.array(f,dtype=np.float32);saved=f.copy();base,score=v6._scores_from_features(f)
        oldbase=base.copy();oldscore=score.copy();new=candidate.corrected_scores(f,score)
        self.assertTrue(bits(f,saved));self.assertTrue(bits(base,oldbase));self.assertTrue(bits(score,oldscore))
        self.assertFalse(np.shares_memory(new,score));self.assertEqual(new.shape,score.shape);self.assertEqual(new.dtype,score.dtype)
        if len(f)>1:
            keep=np.arange(len(f))!=1;self.assertTrue(bits(new[keep],score[keep]));self.assertLessEqual(new[1],score[1])
        else:self.assertTrue(bits(new,score))
        if np.argmax(score)!=1:self.assertEqual(int(np.argmax(new)),int(np.argmax(score)))
        return base,score,new

    def test_one_valid_frame_unchanged(self):
        _,old,new=self.check_features([[0,0,0]])
        self.assertEqual(new.tolist(),[0]);self.assertEqual(np.argmax(new),0)

    def test_two_frames_pure_jerk_becomes_existing_tie(self):
        _,old,new=self.check_features([[0,0,0],[20,0,0]])
        self.assertEqual(np.argmax(old),1);self.assertEqual(new.tolist(),[0,0]);self.assertEqual(np.argmax(new),0)

    def test_two_frames_residual_appearance_still_can_win(self):
        _,old,new=self.check_features([[0,0,0],[20,7,9]])
        self.assertGreater(new[1],0);self.assertEqual(np.argmax(new),1)

    def test_large_boundary_residual_and_appearance_are_not_excluded(self):
        _,old,new=self.check_features([[0,0,0],[100,100,100],[1,0,0],[1,0,0],[1,0,0]])
        self.assertEqual(np.argmax(new),1);self.assertAlmostEqual(float(new[1]),8.5,places=6)

    def test_existing_later_winner_preserved(self):
        _,old,new=self.check_features([[0,0,0],[2,1,1],[200,4,4],[1,1,1],[1,1,1]])
        self.assertEqual(np.argmax(old),2);self.assertEqual(np.argmax(new),2)

    def test_only_boundary_correction_can_change_argmax(self):
        _,old,new=self.check_features([[0,0,0],[100,0,0],[3,0,0],[1,0,0],[1,0,0]])
        self.assertEqual(np.argmax(old),1);self.assertEqual(np.argmax(new),2)

    def test_large_uncapped_value_does_not_cancel_small_appearance(self):
        _,old,new=self.check_features([[0,0,0],[1e10,0,.001],[0,0,0],[0,0,0],[0,0,0]])
        self.assertGreater(old[1],1e12);self.assertEqual(float(new[1]),.25)

    def test_fixed_random_finite_feature_properties(self):
        rng=np.random.default_rng(742)
        for n in (1,2,3,5,31,128):
            for repeat in range(30):
                f=rng.lognormal(0,3,(n,3)).astype(np.float32);f[0]=0
                with self.subTest(n=n,repeat=repeat):self.check_features(f)

    def test_reject_empty_and_nonfinite_features(self):
        for f,s in [(np.empty((0,3),np.float32),np.empty(0,np.float32)),([[0,0,0],[1,float('nan'),0]],[0,1]),([[0,0,0],[1,1,0]],[0,float('inf')])]:
            with self.subTest(f=f),self.assertRaises(ValueError):candidate.corrected_scores(f,s)

    def run_mocked_pipeline(self,features,numbers,invalid_leading=False):
        # Exercise actual candidate folder sorting / final original-number mapping.
        # Stub only data extraction and the existing model operation; neither runs.
        f=np.asarray(features,np.float32);base,score=v6._scores_from_features(f)
        expected=candidate.corrected_scores(f,score);called=[];original_prediction={}
        class NoGPU:
            def __init__(self,*a,**kw):called.append(('constructor',kw));self.tag=1
            def __enter__(self):return self
            def __exit__(self,*args):return False
        with tempfile.TemporaryDirectory(prefix='motion_contract_',dir=OUT) as tmp:
            data=Path(tmp);folder=data/'images'/'arbitrary_file';folder.mkdir(parents=True)
            paths=[folder/f'frame_{n:06}.png' for n in numbers]
            for p in reversed(paths):p.touch()
            invalid=folder/'frame_000002.png'
            if invalid_leading:invalid.touch()
            def scan(p):
                self.assertEqual([v6.primitives._frame_number(x) for x in p],sorted(numbers+([2] if invalid_leading else [])))
                return paths,base,score,f
            def predict(p,b,s,model):
                self.assertEqual(p,paths);self.assertTrue(bits(b,base));self.assertTrue(bits(s,score));self.assertEqual(model.tag,1)
                called.append(('frozen_predict',None))
                prediction={'collision_frame':numbers[int(np.argmax(score))],'entry_frame':numbers[0],'entry_side':'RIGHT','evasion_space':1}
                original_prediction.update(prediction)
                return prediction,{'synthetic_contract_only':True}
            with patch.object(v6.baseline,'CandidateVLM',NoGPU),patch.object(v6,'_dual_motion_scan',scan),patch.object(v6,'_predict_file',predict):
                result=candidate.predict_stage2(data,data/'unused_model')
            row=result.iloc[0].to_dict();self.assertEqual(row['ID'],'arbitrary_file')
            self.assertEqual(row['collision_frame'],numbers[int(np.argmax(expected))])
            for key in ('entry_frame','entry_side','evasion_space'):self.assertEqual(row[key],original_prediction[key])
            self.assertEqual(sum(x[0]=='constructor' for x in called),1);self.assertEqual(sum(x[0]=='frozen_predict' for x in called),1)
            self.assertTrue(bits(base,v6._scores_from_features(f)[0]));self.assertTrue(bits(score,v6._scores_from_features(f)[1]))
            return row

    def test_pipeline_one_valid_original_frame_40(self):
        row=self.run_mocked_pipeline([[0,0,0]],[40],invalid_leading=True)
        self.assertEqual(row['collision_frame'],40)

    def test_pipeline_second_valid_number_100_not_number_1(self):
        row=self.run_mocked_pipeline([[0,0,0],[200,0,0]],[40,100],invalid_leading=True)
        self.assertEqual(row['collision_frame'],40)

    def test_pipeline_second_valid_residual_contact_retained(self):
        row=self.run_mocked_pipeline([[0,0,0],[200,30,40]],[40,100],invalid_leading=True)
        self.assertEqual(row['collision_frame'],100)

if __name__=='__main__':
    suite=unittest.defaultTestLoader.loadTestsFromTestCase(Contracts)
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    unchanged=all(sha(ROOT/p)==v for p,v in BEFORE.items())
    report={'status':'passed' if result.wasSuccessful() and unchanged else 'failed','tests_run':result.testsRun,
        'failures':[(str(t),e) for t,e in result.failures],'errors':[(str(t),e) for t,e in result.errors],
        'source_sha256s':BEFORE,'test_sha256':sha(Path(__file__)),'protected_unchanged':unchanged,
        'random_property_cases':180,'gpu_used':False,'real_frames_or_labels_read':False,'accuracy_evaluation':False,
        'limitation':'Model call stub verifies candidate forwards unchanged baseline inputs; does not rerun four actual VLM calls or certify GPU nondeterminism. Synthetic finite features only.'}
    with (OUT/'motion_initialization_contract_results.json').open('x',encoding='utf-8') as f:json.dump(report,f,ensure_ascii=False,indent=2)
    raise SystemExit(0 if report['status']=='passed' else 1)
