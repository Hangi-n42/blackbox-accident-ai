"""One frozen CPU candidate on existing exposed development traces."""
import sys,json,hashlib,importlib.util,os
from pathlib import Path
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='2'
ROOT=Path(__file__).resolve().parents[2];HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'artifacts/submissions/verify_v6/model/stage2/code'))
import numpy as np
spec=importlib.util.spec_from_file_location('solution.stage2_motion_init_v7',HERE/'solution/stage2_motion_init_v7.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
spec=importlib.util.spec_from_file_location('v7_runner',HERE/'run_stage2.py');runner=importlib.util.module_from_spec(spec);spec.loader.exec_module(runner)
out=HERE/'motion_init_dev';assert sys.flags.isolated and not out.exists();out.mkdir()
rows,bindings=runner.corpus()
bindings.update({str(p):runner.sha(p) for p in [HERE/'motion_initialization_protocol.json',HERE/'solution/stage2_motion_init_v7.py',Path(__file__)]})
runner.put(out/'freeze.json',dict(files=bindings,role='exposed_development',no_model_calls=True))
results=[]
for row in rows:
    rec,t=row['record'],row['trace'];features=np.asarray(t['features'],dtype=np.float32)
    base,scores=m.v6._scores_from_features(features)
    assert np.array_equal(scores,np.asarray(t['new_scores'],dtype=np.float32))
    fixed=m.corrected_scores(features,scores)
    assert np.array_equal(scores[2:],fixed[2:]) and scores[0]==fixed[0]
    pred={**t['candidate'],'collision_frame':t['frame_numbers'][int(np.argmax(fixed))]}
    results.append(dict(ID=row['ID'],baseline=t['candidate'],candidate=pred,frame_pts=rec['input']['source_frame_pts'],source_sha256=rec['input']['source_sha256'],changed_score_indices=np.flatnonzero(fixed!=scores).tolist(),old_initial_score=float(scores[1]),new_initial_score=float(fixed[1])))
assert all(runner.sha(p)==h for p,h in bindings.items())
runner.put(out/'report.json',dict(status='complete',videos=results,source_role='exposed_development',model_calls=0))
print(json.dumps([dict(ID=r['ID'],old=r['baseline']['collision_frame'],new=r['candidate']['collision_frame']) for r in results]))
