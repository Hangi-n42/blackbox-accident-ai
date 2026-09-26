"""Select four further existing routes without reading predictions; freeze proxy intervals."""
import json,hashlib
from pathlib import Path
import numpy as np
O=Path(__file__).resolve().parent;R=O.parents[1];C=R/'artifacts/data_curation_20260917/comma';D=R/'artifacts/pipeline_diagnosis_20260917'
def read(p):return json.loads(p.read_text())
def write(p,d):p.write_text(json.dumps(d,ensure_ascii=False,indent=2))
def sha(p):return hashlib.file_digest(p.open('rb'),'sha256').hexdigest()
assert not (O/'freeze.json').exists()
old=read(D/'comma_cases.json');used={r['route'] for r in old};allrows=read(C/'split_manifest.json');chosen=[]
for vehicle in sorted({r['vehicle_group'] for r in allrows}):
 candidates=[r for r in allrows if r['vehicle_group']==vehicle and r['route'] not in used]
 chosen+=sorted(candidates,key=lambda r:hashlib.sha256(('smooth-followup-v1:'+r['segment']).encode()).hexdigest())[:2]
cases=[]
for j,r in enumerate(chosen):
 d=dict(np.load(C/r['path']));v=d['speed_smoothed'];a=d['acceleration_proxy'];al=d['accel_candidate']
 m=d['accel_use_mask']&(((al==3)&(v<.1))|((v>1)&(((al==0)&(a>.5))|((al==1)&(a<-.5))|((al==2)&(abs(a)<.1)))))
 oldmask=m.copy();m[:]=False
 for k in range(5,len(m)-5):m[k]=oldmask[k-5:k+6].all() and (al[k-5:k+6]==al[k]).all()
 id=f'extra_{j:02d}';np.savez_compressed(O/(id+'_labels.npz'),**d,diagnostic_accel_mask=m)
 chunk,route,seg=r['segment'].split('/');cases.append(dict(id=id,segment=r['segment'],route=r['route'],vehicle=r['vehicle_group'],raw_path=f'external_data/comma2k19/{chunk}/{route.replace("|","_")}/{seg}/video.hevc',input_path=str((O/'comma_input/videos'/(id+'.mkv')).relative_to(R)),labels_npz=id+'_labels.npz',truth_source=str((C/r['path']).relative_to(R)),truth_sha256=sha(C/r['path']),stable_samples=int(m.sum()),previously_in_smoothing_experiment=False,current_model_training_exposed=True,license='MIT comma.ai 2018',source_url='https://github.com/commaai/comma2k19'))
write(O/'comma_cases.json',cases)
events=[]
for row,base in [(r,O) for r in cases]+[(r,D) for r in old]:
 d=np.load(base/row['labels_npz']);y=d['accel_candidate'];starts=np.r_[0,np.flatnonzero(y[1:]!=y[:-1])+1];ends=np.r_[starts[1:],len(y)]
 runs=[(int(a),int(b-1),int(y[a])) for a,b in zip(starts,ends) if y[a]>=0 and b-a>=5]
 for prev,nxt in zip(runs,runs[1:]):
  left=prev[1];right=nxt[0]
  if prev[2]==nxt[2] or right-left>20 or not d['speed_valid'][left:right+1].all():continue
  events.append(dict(id=row['id'],old_class=prev[2],new_class=nxt[2],left_index=left,right_index=right,interval_relative_s=[left/10,right/10],interval_boot_s=[float(d['time'][left]),float(d['time'][right])],truth='sensor-proxy transition bracket, NOT official transition time',prior_group='extra' if base==O else 'previous'))
write(O/'transition_manifest.json',events)
write(O/'freeze.json',dict(selection='2 routes per vehicle, smallest SHA256(smooth-followup-v1:segment); excludes all previous five routes; selection never uses prediction',candidate_source='artifacts/stage3_smoothing_20260917/run.py:smooth',candidate_sha256=sha(R/'artifacts/stage3_smoothing_20260917/run.py'),model_sha256=sha(R/'artifacts/submissions/verify_v6/model/stage3/motion_model.joblib'),stable_rule='identical to previous evaluation mask',transition_rule='>=5 consecutive known proxy labels on each side, gap <=2s; no missing speed. Bracket only. Within +/-2s search first new label sustained 3 samples after any old label sustained 3 samples; compare distance outside bracket and missing crossings, not official timing accuracy.',promotion_rule='Further-route macro F1 must not decrease; report per-vehicle and opposite errors; matched transition interval-distance mean must not worsen and missing transitions must not increase. If evidence missing keep provisional.',cases_sha256=sha(O/'comma_cases.json'),transition_sha256=sha(O/'transition_manifest.json'),script_sha256=sha(Path(__file__))))
assert len(cases)==4 and not {r['route'] for r in cases}&used
print([(r['id'],r['segment'],r['stable_samples']) for r in cases]);print('transition brackets',len(events))
