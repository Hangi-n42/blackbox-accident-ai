import json,hashlib
from pathlib import Path
import numpy as np
O=Path(__file__).resolve().parent;R=O.parents[1];C=R/'artifacts/data_curation_20260917/comma';D=R/'artifacts/pipeline_diagnosis_20260917';S=R/'artifacts/stage3_smoothing_check_20260917'
def read(p):return json.loads(p.read_text())
def write(p,d):p.write_text(json.dumps(d,ensure_ascii=False,indent=2))
assert not (O/'cases.json').exists()
rows=[]
for base in [D,S]:
 for r in read(base/'comma_cases.json'):rows.append(dict(r,labels_path=str((base/r['labels_npz']).relative_to(R)),role='train' if r['vehicle'].startswith('b0') else 'comparison'))
used={r['segment'] for r in rows};opts=[r for r in read(C/'split_manifest.json') if r['vehicle_group'].startswith('b0') and r['segment'] not in used]
r=max(opts,key=lambda r:int((np.load(C/r['path'])['accel_candidate']==3).sum()));d=dict(np.load(C/r['path']));v=d['speed_smoothed'];a=d['acceleration_proxy'];al=d['accel_candidate'];m=d['accel_use_mask']&(((al==3)&(v<.1))|((v>1)&(((al==0)&(a>.5))|((al==1)&(a<-.5))|((al==2)&(abs(a)<.1)))))
old=m.copy();m[:]=False
for k in range(5,len(m)-5):m[k]=old[k-5:k+6].all() and (al[k-5:k+6]==al[k]).all()
id='train_stop';np.savez_compressed(O/(id+'_labels.npz'),**d,diagnostic_accel_mask=m);chunk,route,seg=r['segment'].split('/');new=dict(id=id,segment=r['segment'],route=r['route'],vehicle=r['vehicle_group'],raw_path=f'external_data/comma2k19/{chunk}/{route.replace("|","_")}/{seg}/video.hevc',input_path=str((O/'comma_input/videos'/(id+'.mkv')).relative_to(R)),labels_npz=id+'_labels.npz',labels_path=str((O/(id+'_labels.npz')).relative_to(R)),role='train',selection='unused training-vehicle segment with most reviewed STOPPED labels; no prediction used',license='MIT comma.ai 2018')
rows.append(new);write(O/'comma_cases.json',[new])
for row in rows:
 d=np.load(R/row['labels_path']);idx=np.flatnonzero(d['diagnostic_accel_mask']);idx=idx[idx%5==0] if row['role']=='train' else idx
 row['sample_indices']=idx.tolist();row['class_counts']={str(k):int((d['accel_candidate'][idx]==k).sum()) for k in range(4)};row['existing_model_training_exposed']=True;row['independent_test']=False
assert len(rows)==10;assert {r['vehicle'] for r in rows if r['role']=='train'}.isdisjoint({r['vehicle'] for r in rows if r['role']=='comparison'})
write(O/'cases.json',rows);print([(r['id'],r['role'],len(r['sample_indices']),r['class_counts']) for r in rows])
