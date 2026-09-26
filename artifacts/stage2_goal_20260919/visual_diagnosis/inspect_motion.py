"""Read cached motion components and verify first two measured shifts; no candidate evaluation."""
from pathlib import Path
import json,sys,hashlib
import numpy as np
from PIL import Image
ROOT=Path(__file__).resolve().parents[3];OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'artifacts/submissions/verify_v6/model/stage2/code'))
from solution import stage2_uncapped_jerk_v6c as policy
cv2=policy.primitives.cv2;cv2.setNumThreads(2)
BASE=ROOT/'artifacts/stage2_goal_20260919/current_baseline';inp={r['ID']:r for r in json.loads((BASE/'inputs.json').read_text())};rows=[]
for sid in ['00007','00010']:
 a=np.load(BASE/'run/stage2/human_dev'/sid/'motion.npz');features=a['features'];base,new=policy._scores_from_features(features)
 assert np.array_equal(base,a['base_scores']) and np.array_equal(new,a['new_scores'])
 grays=[cv2.cvtColor(np.asarray(Image.open(ROOT/r['path']).convert('RGB').resize((160,96))),cv2.COLOR_RGB2GRAY) for r in inp[sid]['images'][:3]]
 shifts=[]
 for prev,now in zip(grays,grays[1:]):
  flow=cv2.calcOpticalFlowFarneback(prev,now,None,.5,2,11,2,5,1.1,0);shifts.append(np.median(flow[8:76,8:152].reshape(-1,2),axis=0))
 measured=np.array([np.linalg.norm(shifts[0]),np.linalg.norm(shifts[1]-shifts[0])]);assert np.array_equal(measured.astype('float32'),features[1:3,0])
 jerk=features[:,0];median=np.median(jerk);scale=max(float(np.median(np.abs(jerk-median))*1.4826),1e-3);uncapped=np.maximum((jerk-median)/scale,0);res=.6*policy.primitives._robust_scale(features[:,1]);app=.25*policy.primitives._robust_scale(features[:,2])
 idx=[0,1,2]+[int(i) for i in np.argsort(new)[-3:][::-1]]
 rows.append({'ID':sid,'raw_shift_0_to_1':shifts[0].tolist(),'raw_shift_1_to_2':shifts[1].tolist(),'first_jerk_equals_norm_first_observed_shift':True,'second_jerk_equals_change_of_two_observed_shifts':True,'full_cached_scores_match_production_formula':True,'selected':int(np.argmax(new)),'selected_components':[{'index':i,'pts_seconds':inp[sid]['images'][i]['pts_seconds'],'raw_features':features[i].tolist(),'normalized_uncapped_jerk_contribution':float(uncapped[i]),'weighted_residual_contribution':float(res[i]),'weighted_appearance_contribution':float(app[i]),'final_score':float(new[i])} for i in sorted(set(idx))],'candidate_executed':False})
(OUT/'motion_components.json').write_text(json.dumps(rows,indent=2));print(json.dumps(rows,indent=2))
