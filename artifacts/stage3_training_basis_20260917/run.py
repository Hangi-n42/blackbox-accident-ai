"""All acquired23 aligned segments; original features, fixed labels, two vehicle-separated baselines."""
import importlib.util,json,time,warnings,sys
from pathlib import Path
import numpy as np,pandas as pd,cv2,av,joblib
from sklearn.base import clone
from sklearn.exceptions import ConvergenceWarning
from threadpoolctl import threadpool_limits
O=Path(__file__).resolve().parent;R=O.parents[1];P=R/'artifacts/stage3_lower_roi_ablation_20260917';D=R/'artifacts/pipeline_diagnosis_20260917'
spec=importlib.util.spec_from_file_location('prior',P/'run.py');prior=importlib.util.module_from_spec(spec);spec.loader.exec_module(prior);old=prior.old;read=old.read;write=old.write;sha=old.sha;ACCEL=old.ACCEL
import solution.stage3_v5_compatible as engine

class SelectedNativeFrames:
 def __init__(self,path,indices):
  self.container=av.open(str(path));self.frames=enumerate(self.container.decode(video=0));self.indices=iter(map(int,indices))
 def read(self):
  target=next(self.indices,None)
  if target is None:return False,None
  for i,frame in self.frames:
   if i==target:return True,frame.to_ndarray(format='bgr24')
  raise ValueError('Missing selected native frame')
 def release(self):self.container.close()

def extract_direct(row):
 d=np.load(O/row['labels_npz']);indices=d['frame_index'];assert np.all(np.diff(indices)>0)
 saved=engine.cv2.VideoCapture
 engine.cv2.VideoCapture=lambda ignored:SelectedNativeFrames(R/row['raw_path'],indices)
 try:return engine.extract_motion(R/row['raw_path'],source_fps=10.)
 finally:engine.cv2.VideoCapture=saved

def loadcache(path):
 a=np.load(path);return a['features'] if isinstance(a,np.lib.npyio.NpzFile) else a

def main():
 assert not (O/'freeze.json').exists()
 cases=read(O/'cases.json');production=read(D/'freeze.json')['files'];assert all(sha(R/p)==h for p,h in production.items());small=joblib.load(P/'control.joblib')['accel']
 write(O/'freeze.json',{'cases_sha256':sha(O/'cases.json'),'transitions_sha256':sha(O/'transition_manifest.json'),'script_sha256':sha(Path(__file__)),'small_control_sha256':sha(P/'control.joblib'),'source':'all23 acquired aligned comma segments only','features':'unchanged original864; no ROI exclusion, no smoothing','primary':'train RAV4all11 at2Hz, evaluate Civicall12 at10Hz valid mask; public50 never fitted','reverse_audit':'train Civicall12 at2Hz, evaluate RAV4all11 at10Hz; no tuning from this audit','labels':'same previous strict thresholds/window and missing masks, no new class definition','parameters':{k:str(v) for k,v in small.get_params().items()},'decision':'Foundation checkpoints only; no production replacement from exposed proxy results','selection_before_model_results':True})
 cv2.setNumThreads(2);features={};runtime=[]
 # Prove streaming selection matches the previously lossless prepared video path before reuse.
 test=next(r for r in cases if r['feature_cache']);x=extract_direct(test);cached=loadcache(R/test['feature_cache']);assert np.array_equal(x,cached),'Selected native stream differs from canonical preparation'
 for row in cases:
  start=time.perf_counter();id=row['id']
  if row['feature_cache']:xx=loadcache(R/row['feature_cache']);source='existing original864 cache'
  else:xx=extract_direct(row);np.save(O/(id+'_features.npy'),xx);source='same PyAV selected native frames streamed to unchanged extractor'
  assert xx.shape==(row['total_timestamps'],864) and np.isfinite(xx).all();features[id]=xx;runtime.append(dict(id=id,source=source,seconds=time.perf_counter()-start));print('features',id,source,flush=True)
 models={};fitlog=[]
 for name,role in [('expanded_rav4','train'),('reverse_civic','comparison')]:
  xx=[];yy=[]
  for row in cases:
   if row['role']!=role:continue
   ix=np.array(row['training_indices']);labels=np.load(O/row['labels_npz']);xx.append(features[row['id']][ix]);yy.append(labels['accel_candidate'][ix])
  X=np.concatenate(xx);Y=np.concatenate(yy);assert set(Y)=={0,1,2,3};model=clone(small);start=time.perf_counter()
  with warnings.catch_warnings(record=True) as ws,threadpool_limits(limits=2):warnings.simplefilter('always');model.fit(X,Y)
  assert not any(issubclass(w.category,ConvergenceWarning) for w in ws)
  models[name]=model;joblib.dump({'accel':model,'feature_version':'existing864','training_vehicle_role':role,'not_production_package':True},O/(name+'.joblib'));fitlog.append(dict(name=name,n=len(Y),class_counts={str(ACCEL[i]):int((Y==i).sum()) for i in range(4)},seconds=time.perf_counter()-start,iterations=model.named_steps['logisticregression'].n_iter_.tolist(),warnings=[str(w.message) for w in ws]));print('fit',name,len(Y),flush=True)
 rows=[];full=[];lookup={}
 for r in cases:
  id=r['id'];x=features[id];model=models['expanded_rav4'] if r['role']=='comparison' else models['reverse_civic'];label=np.load(O/r['labels_npz']);p=model.predict(x).astype(int);b=small.predict(x).astype(int) if r['role']=='comparison' else None;lookup[id]=dict(expanded=p,small=b)
  for i in range(len(x)):full.append(dict(id=id,role=r['role'],sample_index=i,vehicle_held_out_prediction=str(ACCEL[p[i]]),small_prediction=str(ACCEL[b[i]]) if b is not None else None))
  for i in r['evaluation_indices']:rows.append(dict(id=id,role=r['role'],prior_small_case=r['prior_small_experiment'],sample_index=i,truth=str(ACCEL[label['accel_candidate'][i]]),expanded=str(ACCEL[p[i]]),small=str(ACCEL[b[i]]) if b is not None else None))
 s=pd.DataFrame(rows);results={}
 for name,sub in [('primary_civic_all',s[s.role=='comparison']),('primary_civic_previous5',s[(s.role=='comparison')&s.prior_small_case]),('primary_civic_added7',s[(s.role=='comparison')&~s.prior_small_case])]:results[name]={k:old.metrics(sub.truth,sub[k]) for k in ['small','expanded']}
 assert abs(results['primary_civic_previous5']['small']['macro_f1']-read(P/'report.json')['results']['comparison']['control']['macro_f1'])<1e-12
 sub=s[s.role=='train'];results['reverse_rav4_all']=old.metrics(sub.truth,sub.expanded)
 public=pd.read_csv(R/'Baseline/data/stage3/labels.csv');pub=[]
 for id,g in public.groupby('ID'):
  x=np.load(R/'artifacts/data_pilot_20260916/comma_experiment/mac_public_features'/(id+'.npy'));pred={name:model.predict(x).astype(int) for name,model in dict(small=small,**models).items()}
  for z in g.itertuples():pub.append(dict(id=id,sample_index=int(z.sample_index),truth=z.accel_label,**{k:str(ACCEL[v[z.sample_index]]) for k,v in pred.items()}))
 pub=pd.DataFrame(pub);results['public']={k:old.metrics(pub.truth,pub[k]) for k in ['small','expanded_rav4','reverse_civic']}
 events=[]
 for e in read(O/'transition_manifest.json'):
  values=lookup[e['id']];a=prior.crossing(values['expanded'],e);b=prior.crossing(values['small'],e) if e['role']=='comparison' else None;events.append(dict(e,expanded_distance=a,small_distance=b))
 primary=[e for e in events if e['role']=='comparison'];both=[e for e in primary if e['expanded_distance'] is not None and e['small_distance'] is not None];tr={'n':len(primary),'both_detected':len(both),'small_missing':sum(e['small_distance'] is None for e in primary),'expanded_missing':sum(e['expanded_distance'] is None for e in primary),'small_matched_distance_s':float(np.mean([e['small_distance'] for e in both])) if both else None,'expanded_matched_distance_s':float(np.mean([e['expanded_distance'] for e in both])) if both else None}
 s.to_csv(O/'scored_rows.csv',index=False);pd.DataFrame(full).to_csv(O/'cross_vehicle_predictions.csv',index=False);pub.to_csv(O/'public_predictions.csv',index=False);write(O/'transitions.json',events);write(O/'runtime.json',dict(features=runtime,fit=fitlog));write(O/'report.json',{'metrics':results,'transitions_primary':tr,'checks':{'direct_native_matches_prepared_cache':True,'prior_small_control_reproduced':True,'both_models_converged':True,'no_vehicle_overlap':True,'production_unchanged':all(sha(R/p)==h for p,h in production.items()),'original_labels_unchanged':all(sha(R/r['truth_source'])==r['truth_sha256'] for r in cases)},'decision':'training_foundation_ready_for_controlled_experiments_not_production_adoption','scope':'23segments,2vehicles,known development exposure; proxy labels; no new dataset download; original features unchanged','exit_status':0})
 print(json.dumps({'metrics':results,'transitions_primary':tr},indent=2))
if __name__=='__main__':main()
