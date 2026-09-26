"""One masked-body statistics candidate; reuse exact paired control and split."""
import json,importlib.util,time,warnings,sys
from pathlib import Path
import numpy as np,pandas as pd,cv2,joblib
from sklearn.base import clone
from sklearn.exceptions import ConvergenceWarning
from threadpoolctl import threadpool_limits
O=Path(__file__).resolve().parent;R=O.parents[1];P=R/'artifacts/stage3_lower_roi_ablation_20260917';D=R/'artifacts/pipeline_diagnosis_20260917'
spec=importlib.util.spec_from_file_location('prior_ablation',P/'run.py');prior=importlib.util.module_from_spec(spec);spec.loader.exec_module(prior)
old=prior.old;read=old.read;write=old.write;sha=old.sha;ACCEL=old.ACCEL
import solution.stage3_v5_compatible as engine
from masked_features import BodyFeatureComputer,Base,POLYGON_X,POLYGON_Y

def main():
 assert not (O/'freeze.json').exists()
 production=read(D/'freeze.json')['files'];assert all(sha(R/p)==h for p,h in production.items())
 cases=read(P/'cases.json');control=joblib.load(P/'control.joblib')['accel'];template=clone(control)
 write(O/'boundary_review.json',{'reviewer':'AI visual inspection, not human pixel annotation','videos':15,'frames':45,'sample_images':[f'review_{i}.jpg' for i in range(5)],'finding':'Lower ROI contains road and vehicle body. Interior below conservative piecewise boundary lies within visible hood/dashboard in reviewed samples; night borders/reflections uncertain. Not full body segmentation.','normalized_boundary':{'x':POLYGON_X,'y':POLYGON_Y},'exclude':'y >= interpolated boundary(x)','selection':'same fixed geometry for all15 videos, no IDs or labels; polygon fixed before candidate results','limitations':'Unreviewed frames and other camera mountings not certified; upper hood remnants remain; flow still computed on unmasked full frame','source':'comma2k19 MIT comma.ai2018; public examples original competition provenance retained'})
 write(O/'freeze.json',{'cases_sha256':sha(P/'cases.json'),'control_sha256':sha(P/'control.joblib'),'review_sha256':sha(O/'boundary_review.json'),'script_sha256':sha(Path(__file__)),'feature_script_sha256':sha(O/'masked_features.py'),'candidate':'exclude reviewed conservative body-interior pixels only from ROI summary; preserve864 dims, flow input, geometry, temporal features and model settings','train':'same218 RAV4 labels, scaler fitted on training only','comparison':'same1897 Civic plus50 public, prior37 transition intervals','gates':'same prior ablation: Civic macroF1 and constantF1 improve, opposite errors not increase, public F1 no regression, transition distance/missing no worse','no_smoothing':True,'no_production_changes':True})
 # Meaningful contracts: unchanged upper rows, fixed dimensionality, no empty ROI.
 rng=np.random.default_rng(42);flow=rng.normal(size=(144,256,2)).astype(np.float32);a=Base()(flow);b=BodyFeatureComputer()(flow);assert np.array_equal(a[:96],b[:96]);assert a.shape==b.shape==(144,) and np.isfinite(b).all();assert not np.array_equal(a[96:],b[96:])
 engine.FrameFeatureComputer=BodyFeatureComputer;cv2.setNumThreads(2);features={};runtime=[]
 public=pd.read_csv(R/'Baseline/data/stage3/labels.csv');jobs=cases+[dict(id=id,input_path=f'artifacts/public_eval_10hz/stage3/videos/{id}.mp4',role='public') for id in sorted(public.ID.unique())]
 for r in jobs:
  start=time.perf_counter();x=engine.extract_motion(R/r['input_path'],source_fps=10.)
  if r['role']=='public':base=np.load(R/'artifacts/data_pilot_20260916/comma_experiment/mac_public_features'/(r['id']+'.npy'))
  else:
   cache=P/(r['id']+'_features.npy');base=np.load(cache) if cache.exists() else np.load(R/'artifacts/stage3_error_diagnosis_20260917'/(r['id']+'_features.npz'))['features']
  upper=np.arange(864).reshape(6,12,4,3)[:,:8].ravel();assert x.shape==base.shape and np.array_equal(x[:,upper],base[:,upper]),'Unmasked upper features changed'
  np.save(O/(r['id']+'_features.npy'),x);features[r['id']]=x;runtime.append(dict(id=r['id'],frames=len(x),seconds=time.perf_counter()-start,upper_features_identical=True));print('features',r['id'],flush=True)
 X=[];Y=[]
 for r in cases:
  if r['role']!='train':continue
  ix=np.array(r['sample_indices']);d=np.load(R/r['labels_path']);X.append(features[r['id']][ix]);Y.append(d['accel_candidate'][ix])
 X=np.concatenate(X);Y=np.concatenate(Y);assert len(Y)==218
 with warnings.catch_warnings(record=True) as ws,threadpool_limits(limits=2):warnings.simplefilter('always');template.fit(X,Y)
 assert not any(issubclass(w.category,ConvergenceWarning) for w in ws)
 joblib.dump({'accel':template,'feature_version':'body_interior_mask864','boundary_x':POLYGON_X,'boundary_y':POLYGON_Y,'not_production_package':True},O/'body_mask.joblib')
 # Baseline reuse is exact same input, sample indices, features, model and parameters.
 baseline=pd.read_csv(P/'scored_rows.csv');bp=pd.read_csv(P/'predictions.csv');rows=[];predrows=[];byid={}
 for r in jobs:
  id=r['id'];pred=template.predict(features[id]).astype(int);byid[id]=pred
  for i,p in enumerate(pred):predrows.append(dict(id=id,sample_index=i,body_mask=str(ACCEL[p])))
  sub=baseline[baseline.id==id]
  for z in sub.itertuples():rows.append(dict(id=id,scope=z.scope,sample_index=z.sample_index,truth=z.truth,control=z.control,body_mask=str(ACCEL[pred[z.sample_index]])))
 s=pd.DataFrame(rows);results={}
 for name,sub in list(s.groupby('scope'))+list(s.groupby('id')):results[name]={v:old.metrics(sub.truth,sub[v]) for v in ['control','body_mask']}
 events=[];eval_ids={r['id'] for r in cases if r['role']=='comparison'}
 for e in read(R/'artifacts/stage3_smoothing_check_20260917/transitions_corrected.json'):
  if e['id'] not in eval_ids:continue
  b=bp[bp.id==e['id']].sort_values('sample_index').control.map({str(v):i for i,v in enumerate(ACCEL)}).to_numpy();events.append(dict(id=e['id'],interval=e['interval_relative_s'],control=prior.crossing(b,e),body_mask=prior.crossing(byid[e['id']],e)))
 both=[e for e in events if e['control'] is not None and e['body_mask'] is not None];ts=dict(events=len(events),both_detected=len(both),control_missing=sum(e['control'] is None for e in events),candidate_missing=sum(e['body_mask'] is None for e in events),control_distance_s=float(np.mean([e['control'] for e in both])) if both else None,candidate_distance_s=float(np.mean([e['body_mask'] for e in both])) if both else None)
 b=results['comparison']['control'];c=results['comparison']['body_mask'];p=results['public'];gates={'comparison_macro_improved':c['macro_f1']>b['macro_f1'],'constant_f1_improved':c['per_class_f1']['CONSTANT']>b['per_class_f1']['CONSTANT'],'opposite_errors_no_increase':c['opposite_accel_errors']<=b['opposite_accel_errors'],'public_no_regression':p['body_mask']['macro_f1']>=p['control']['macro_f1'],'transitions_no_worse':bool(both) and ts['candidate_distance_s']<=ts['control_distance_s'] and ts['candidate_missing']<=ts['control_missing']}
 s.to_csv(O/'scored_rows.csv',index=False);pd.DataFrame(predrows).to_csv(O/'predictions.csv',index=False);ch=s[s.control!=s.body_mask].copy();ch['effect']=np.where(ch.body_mask==ch.truth,'fixed',np.where(ch.control==ch.truth,'regressed','wrong_to_wrong'));ch.to_csv(O/'changed_rows.csv',index=False)
 write(O/'transitions.json',events);write(O/'runtime.json',runtime);write(O/'report.json',{'results':results,'transitions':ts,'gates':gates,'decision':'retain_candidate_for_review' if all(gates.values()) else 'reject_keep_production','convergence_iterations':template.named_steps['logisticregression'].n_iter_.tolist(),'warnings':[str(w.message) for w in ws],'upper_features_identical_all15':True,'production_unchanged':all(sha(R/p)==h for p,h in production.items()),'control_reused_verified':sha(P/'control.joblib')==read(O/'freeze.json')['control_sha256'],'no_steering_change':True,'exit_status':0})
 print(json.dumps({'metrics':{k:results[k] for k in ['comparison','public']},'transitions':ts,'gates':gates},indent=2))
if __name__=='__main__':main()
