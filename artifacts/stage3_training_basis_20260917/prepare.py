"""Expand existing23 sensor-aligned segments; keep thresholds and vehicle split unchanged."""
import json,hashlib
from pathlib import Path
import numpy as np
O=Path(__file__).resolve().parent;R=O.parents[1];C=R/'artifacts/data_curation_20260917/comma';P=R/'artifacts/stage3_lower_roi_ablation_20260917'
def read(p):return json.loads(p.read_text())
def write(p,d):p.write_text(json.dumps(d,ensure_ascii=False,indent=2))
def sha(p):return hashlib.file_digest(p.open('rb'),'sha256').hexdigest()
assert not (O/'cases.json').exists()
known={x['segment']:x for x in read(P/'cases.json')};rows=[];events=[];labels=read(C/'split_manifest.json')
for no,r in enumerate(labels):
 d=dict(np.load(C/r['path']));v=d['speed_smoothed'];a=d['acceleration_proxy'];al=d['accel_candidate'];mask=d['accel_use_mask']&(((al==3)&(v<.1))|((v>1)&(((al==0)&(a>.5))|((al==1)&(a<-.5))|((al==2)&(abs(a)<.1)))))
 initial=mask.copy();mask[:]=False
 for k in range(5,len(mask)-5):mask[k]=initial[k-5:k+6].all() and (al[k-5:k+6]==al[k]).all()
 old=known.get(r['segment']);id=old['id'] if old else f'expanded_{no:02d}';feature=None
 if old:
  p=P/(id+'_features.npy');p2=R/'artifacts/stage3_error_diagnosis_20260917'/(id+'_features.npz');feature=str((p if p.exists() else p2).relative_to(R));oldlabels=np.load(R/old['labels_path']);assert np.array_equal(mask,oldlabels['diagnostic_accel_mask'])
 chunk,route,seg=r['segment'].split('/');raw=f'external_data/comma2k19/{chunk}/{route.replace("|","_")}/{seg}/video.hevc';np.savez_compressed(O/(id+'_labels.npz'),**d,diagnostic_accel_mask=mask)
 role='train' if r['vehicle_group'].startswith('b0') else 'comparison';ix=np.flatnonzero(mask);trainix=ix[ix%5==0]
 row=dict(id=id,segment=r['segment'],route=r['route'],vehicle=r['vehicle_group'],role=role,raw_path=raw,truth_source=str((C/r['path']).relative_to(R)),truth_sha256=sha(C/r['path']),labels_npz=id+'_labels.npz',feature_cache=feature,prior_small_experiment=old is not None,training_indices=trainix.tolist(),evaluation_indices=ix.tolist(),stable_class_counts={str(k):int((al[ix]==k).sum()) for k in range(4)},training_class_counts={str(k):int((al[trainix]==k).sum()) for k in range(4)},total_timestamps=len(al),invalid_speed=int((~d['speed_valid']).sum()),excluded_from_strict=len(al)-len(ix),truth_type='sensor_proxy_trial_thresholds_not_official',existing_model_training_exposed=True,independent_test=False,source_url='https://github.com/commaai/comma2k19',license='MIT comma.ai2018',license_file='artifacts/data_curation_20260917/comma/SOURCE_LICENSE')
 rows.append(row)
 starts=np.r_[0,np.flatnonzero(al[1:]!=al[:-1])+1];ends=np.r_[starts[1:],len(al)];runs=[(int(b),int(e-1),int(al[b])) for b,e in zip(starts,ends) if al[b]>=0 and e-b>=5]
 for p,n in zip(runs,runs[1:]):
  if p[2]==n[2] or n[0]-p[1]>20 or not d['speed_valid'][p[1]:n[0]+1].all():continue
  events.append(dict(id=id,role=role,old_class=p[2],new_class=n[2],old_run_start=p[0],left_index=p[1],right_index=n[0],new_run_end=n[1],interval_relative_s=[p[1]/10,n[0]/10],truth='proxy transition bracket; uncertain point time'))
assert len(rows)==23;assert {r['vehicle'] for r in rows if r['role']=='train'}.isdisjoint({r['vehicle'] for r in rows if r['role']=='comparison'})
write(O/'cases.json',rows);write(O/'transition_manifest.json',events)
counts={}
for role in ['train','comparison']:
 part=[r for r in rows if r['role']==role];counts[role]={'segments':len(part),'classes_at2Hz':{str(k):sum(r['training_class_counts'][str(k)] for r in part) for k in range(4)},'classes_at10Hz':{str(k):sum(r['stable_class_counts'][str(k)] for r in part) for k in range(4)},'total_timestamps':sum(r['total_timestamps'] for r in part),'strict_excluded':sum(r['excluded_from_strict'] for r in part),'transitions':sum(e['role']==role for e in events),'routes_by_class':{str(k):sum(r['training_class_counts'][str(k)]>0 for r in part) for k in range(4)}}
write(O/'inventory.json',counts);print(json.dumps(counts,indent=2))
