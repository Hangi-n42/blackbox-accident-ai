"""Paired retraining: 864 features vs 576 excluding lower 4 ROIs; fixed data and settings."""
import json,importlib.util,time,warnings
from pathlib import Path
import numpy as np,pandas as pd,cv2,joblib
from sklearn.base import clone
from sklearn.exceptions import ConvergenceWarning
from threadpoolctl import threadpool_limits
O=Path(__file__).resolve().parent;R=O.parents[1];D=R/'artifacts/pipeline_diagnosis_20260917'
spec=importlib.util.spec_from_file_location('existing',R/'artifacts/stage3_smoothing_20260917/run.py');old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
read=old.read;write=old.write;sha=old.sha;ACCEL=old.ACCEL

def crossing(pred,e):
 lo=max(e['old_run_start'],e['left_index']-20);hi=min(e['new_run_end']-2,e['right_index']+20,len(pred)-3);seen=False
 for k in range(lo,hi+1):
  if np.all(pred[k:k+3]==e['old_class']):seen=True
  if seen and np.all(pred[k:k+3]==e['new_class']):return max(e['left_index']-k,0,k-e['right_index'])/10
 return None

def main():
 assert not (O/'freeze.json').exists(),'fresh output required'
 production=read(D/'freeze.json')['files'];assert all(sha(R/p)==h for p,h in production.items())
 checkpoint=joblib.load(R/'artifacts/submissions/verify_v6/model/stage3/motion_model.joblib');template=checkpoint['accel'];cases=read(O/'cases.json');mask=np.arange(864).reshape(6,12,4,3)[:,:8,:,:].ravel();assert len(mask)==576 and len(set(mask))==576
 write(O/'freeze.json',{'model_sha256':sha(R/'artifacts/submissions/verify_v6/model/stage3/motion_model.joblib'),'cases_sha256':sha(O/'cases.json'),'script_sha256':sha(Path(__file__)),'experiment':'same 218 training timestamps, scaler fit separately on training only, identical LogisticRegression parameters; only drop ROI8..11 in all six temporal feature blocks','parameters':{k:str(v) for k,v in template.get_params().items()},'candidate_indices':mask.tolist(),'training_sampling':'fixed valid samples every5 indices (2Hz)','evaluation':'Civic5 routes at10Hz valid mask, public50 labels not fitted; transitions use previous corrected proxy brackets','gates':'Civic macro F1 and CONSTANT F1 strictly improve, opposite accel errors do not increase; public macro F1 no regression vs paired retrained control; matched transition distance and missing count no worse. No matching transitions means insufficient evidence, do not promote.','not_independent':'Previously exposed data. New models train on RAV4 only; development compared on Civic. Old production trained on both and is reference only.','no_smoothing':True,'production_changed':False})
 cv2.setNumThreads(2);features={};inputs=[];xx=[];yy=[]
 for row in cases:
  id=row['id'];cache=R/'artifacts/stage3_error_diagnosis_20260917'/(id+'_features.npz');start=time.perf_counter()
  if cache.exists():x=np.load(cache)['features'];source=str(cache.relative_to(R))
  else:x=old.extract_motion(R/row['input_path'],source_fps=10.);source='current extractor on prepared10Hz';np.save(O/(id+'_features.npy'),x)
  d=np.load(R/row['labels_path']);assert len(x)==len(d['time']) and x.shape[1]==864 and np.isfinite(x).all();features[id]=x
  if row['role']=='train':ix=np.array(row['sample_indices']);xx.append(x[ix]);yy.append(d['accel_candidate'][ix])
  inputs.append(dict(id=id,source=source,frames=len(x),seconds=time.perf_counter()-start));print('features',id,flush=True)
 X=np.concatenate(xx);Y=np.concatenate(yy);assert set(Y)=={0,1,2,3};models={};fit=[]
 for variant,indices in [('control',np.arange(864)),('drop_lower',mask)]:
  model=clone(template);start=time.perf_counter()
  with warnings.catch_warnings(record=True) as seen,threadpool_limits(limits=2):
   warnings.simplefilter('always');model.fit(X[:,indices],Y)
  assert not any(issubclass(w.category,ConvergenceWarning) for w in seen),'Unconverged model'
  models[variant]=(model,indices);joblib.dump({'accel':model,'feature_indices':indices,'feature_version':'existing864_select_columns','not_production_package':True},O/(variant+'.joblib'));fit.append(dict(variant=variant,samples=len(Y),features=len(indices),seconds=time.perf_counter()-start,warnings=[str(w.message) for w in seen],iterations=model.named_steps['logisticregression'].n_iter_.tolist()));print('fit',variant,flush=True)
 rows=[];predrows=[];byid={}
 for row in cases:
  id=row['id'];x=features[id];d=np.load(R/row['labels_path']);pred={v:m.predict(x[:,ix]).astype(int) for v,(m,ix) in models.items()};byid[id]=pred
  for i in range(len(x)):predrows.append(dict(id=id,sample_index=i,control=str(ACCEL[pred['control'][i]]),drop_lower=str(ACCEL[pred['drop_lower'][i]])))
  for i in row['sample_indices']:rows.append(dict(id=id,scope=row['role'],sample_index=i,truth=str(ACCEL[d['accel_candidate'][i]]),control=str(ACCEL[pred['control'][i]]),drop_lower=str(ACCEL[pred['drop_lower'][i]])))
 public=pd.read_csv(R/'Baseline/data/stage3/labels.csv')
 for id,g in public.groupby('ID'):
  x=np.load(R/'artifacts/data_pilot_20260916/comma_experiment/mac_public_features'/(id+'.npy'));ix=g.sample_index.to_numpy();pred={v:m.predict(x[:,cols]).astype(int) for v,(m,cols) in models.items()}
  for i,t in zip(ix,g.accel_label):rows.append(dict(id=id,scope='public',sample_index=int(i),truth=t,control=str(ACCEL[pred['control'][i]]),drop_lower=str(ACCEL[pred['drop_lower'][i]])))
 s=pd.DataFrame(rows);results={}
 for name,sub in list(s.groupby('scope'))+list(s.groupby('id')):results[name]={v:old.metrics(sub.truth,sub[v]) for v in ['control','drop_lower']}
 eval_ids={r['id'] for r in cases if r['role']=='comparison'};events=[]
 for e in read(R/'artifacts/stage3_smoothing_check_20260917/transitions_corrected.json'):
  if e['id'] not in eval_ids:continue
  ds={v:crossing(byid[e['id']][v],e) for v in models};events.append(dict(id=e['id'],old_class=e['old_class'],new_class=e['new_class'],interval=e['interval_relative_s'],**ds))
 both=[e for e in events if e['control'] is not None and e['drop_lower'] is not None];ts={'events':len(events),'both_detected':len(both),'control_missing':sum(e['control'] is None for e in events),'candidate_missing':sum(e['drop_lower'] is None for e in events),'control_distance_s':float(np.mean([e['control'] for e in both])) if both else None,'candidate_distance_s':float(np.mean([e['drop_lower'] for e in both])) if both else None}
 b=results['comparison']['control'];c=results['comparison']['drop_lower'];p=results['public'];gates={'comparison_macro_improved':c['macro_f1']>b['macro_f1'],'constant_f1_improved':c['per_class_f1']['CONSTANT']>b['per_class_f1']['CONSTANT'],'opposite_errors_no_increase':c['opposite_accel_errors']<=b['opposite_accel_errors'],'public_no_regression':p['drop_lower']['macro_f1']>=p['control']['macro_f1'],'transitions_no_worse':bool(both) and ts['candidate_distance_s']<=ts['control_distance_s'] and ts['candidate_missing']<=ts['control_missing']}
 s.to_csv(O/'scored_rows.csv',index=False);pd.DataFrame(predrows).to_csv(O/'predictions.csv',index=False);changes=s[s.control!=s.drop_lower].copy();changes['effect']=np.where(changes.drop_lower==changes.truth,'fixed',np.where(changes.control==changes.truth,'regressed','wrong_to_wrong'));changes.to_csv(O/'changed_rows.csv',index=False)
 write(O/'transitions.json',events);write(O/'runtime.json',{'input':inputs,'fit':fit});write(O/'report.json',{'results':results,'training_class_counts':dict(zip(map(str,ACCEL),[int((Y==i).sum()) for i in range(4)])),'transitions':ts,'gates':gates,'decision':'retain_candidate_for_review' if all(gates.values()) else 'reject_keep_production','production_unchanged':all(sha(R/p)==h for p,h in production.items()),'no_steering_model_change':True,'converged':True,'exit_status':0})
 print(json.dumps({'results':{k:results[k] for k in ['train','comparison','public']},'transitions':ts,'gates':gates},indent=2))
if __name__=='__main__':main()
