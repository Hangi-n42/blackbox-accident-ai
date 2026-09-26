"""Fixed candidate, four further routes, and proxy transition brackets; no tuning."""
import importlib.util,json,sys,time,hashlib
from pathlib import Path
import numpy as np,pandas as pd,cv2,joblib
from threadpoolctl import threadpool_limits
O=Path(__file__).resolve().parent;R=O.parents[1];OLD=R/'artifacts/stage3_smoothing_20260917'
spec=importlib.util.spec_from_file_location('previous_experiment',OLD/'run.py');old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
read=old.read;write=old.write;sha=old.sha;ACCEL=old.ACCEL

def transition(pred,event):
 left,right=event['left_index'],event['right_index'];lo=max(0,left-20);hi=min(len(pred)-3,right+20);seen_old=False
 for k in range(lo,hi+1):
  if np.all(pred[k:k+3]==event['old_class']):seen_old=True
  if seen_old and np.all(pred[k:k+3]==event['new_class']):return dict(index=k,relative_s=k/10,distance_outside_interval_s=max(left-k,0,k-right)/10)
 return None

def main():
 assert not (O/'report.json').exists()
 f=read(O/'freeze.json');assert sha(OLD/'run.py')==f['candidate_sha256'];assert sha(O/'transition_manifest.json')==f['transition_sha256'];assert sha(O/'comma_cases.json')==f['cases_sha256']
 production=read(R/'artifacts/pipeline_diagnosis_20260917/freeze.json')['files'];assert all(sha(R/p)==h for p,h in production.items())
 model=joblib.load(R/'artifacts/submissions/verify_v6/model/stage3/motion_model.joblib');classes=np.asarray(model['accel'].classes_,int);cv2.setNumThreads(2)
 # Known crossing, delayed crossing, and absent crossing checks.
 e={'left_index':9,'right_index':11,'old_class':0,'new_class':1};assert transition(np.r_[np.zeros(10,int),np.ones(20,int)],e)['distance_outside_interval_s']==0;assert transition(np.zeros(30,int),e) is None
 rows=[];preds=[];timing=[];casepred={}
 for r in read(O/'comma_cases.json'):
  start=time.perf_counter();x=old.extract_motion(R/r['input_path'],source_fps=10.)
  if 'motion_regressor' in model:x=old.transfer_features(x,model['motion_regressor'])
  if 'accel_motion_regressor' in model:x=old.transfer_features(x,model['accel_motion_regressor'])
  with threadpool_limits(limits=2):p=model['accel'].predict_proba(x);base=model['accel'].predict(x).astype(int)
  assert np.array_equal(classes[p.argmax(1)],base)
  sp=old.smooth(p);cand=classes[sp.argmax(1)];d=np.load(O/r['labels_npz']);assert len(base)==len(d['time']);idx=np.flatnonzero(d['diagnostic_accel_mask'])
  np.savez_compressed(O/(r['id']+'_probabilities.npz'),classes=classes,probabilities=p,smoothed_probabilities=sp)
  casepred[r['id']]=(base,cand)
  for i in range(len(base)):preds.append(dict(id=r['id'],sample_index=i,baseline=str(ACCEL[base[i]]),candidate=str(ACCEL[cand[i]])))
  for i in idx:rows.append(dict(id=r['id'],vehicle=r['vehicle'],sample_index=int(i),truth=str(ACCEL[d['accel_candidate'][i]]),baseline=str(ACCEL[base[i]]),candidate=str(ACCEL[cand[i]])))
  timing.append(dict(id=r['id'],frames=len(base),scored=len(idx),seconds=time.perf_counter()-start));print(r['id'],len(idx),flush=True)
 # Reuse previous 5 probabilities without inference or label changes.
 for path in OLD.glob('comma_*_probabilities.npz'):
  z=np.load(path);id=path.name.replace('_probabilities.npz','');casepred[id]=(z['classes'][z['probabilities'].argmax(1)],z['classes'][z['smoothed_probabilities'].argmax(1)])
 transitions=[]
 for e in read(O/'transition_manifest.json'):
  base,cand=casepred[e['id']];a=transition(base,e);b=transition(cand,e)
  transitions.append(dict(e,baseline=a,candidate=b,candidate_minus_baseline_s=(b['index']-a['index'])/10 if a and b else None))
 scored=pd.DataFrame(rows);metrics={}
 for group,subset in [('all_extra',scored)]+list(scored.groupby('vehicle'))+list(scored.groupby('id')):
  metrics[group]={name:old.metrics(subset.truth,subset[name]) for name in ['baseline','candidate']}
 summaries={}
 for group in ['all','extra','previous']:
  ts=transitions if group=='all' else [t for t in transitions if t['prior_group']==group];both=[t for t in ts if t['baseline'] and t['candidate']]
  summaries[group]={'n':len(ts),'both_detected':len(both),'baseline_missing':sum(t['baseline'] is None for t in ts),'candidate_missing':sum(t['candidate'] is None for t in ts),'baseline_matched_mean_distance_s':float(np.mean([t['baseline']['distance_outside_interval_s'] for t in both])) if both else None,'candidate_matched_mean_distance_s':float(np.mean([t['candidate']['distance_outside_interval_s'] for t in both])) if both else None,'candidate_later_count':sum(t['candidate_minus_baseline_s']>0 for t in both),'candidate_earlier_count':sum(t['candidate_minus_baseline_s']<0 for t in both),'mean_signed_crossing_shift_s':float(np.mean([t['candidate_minus_baseline_s'] for t in both])) if both else None}
 s=summaries['all'];m=metrics['all_extra'];gates={'extra_macro_no_regression':m['candidate']['macro_f1']>=m['baseline']['macro_f1'],'matched_transition_distance_no_worse':s['both_detected']>0 and s['candidate_matched_mean_distance_s']<=s['baseline_matched_mean_distance_s'],'missing_crossings_no_increase':s['candidate_missing']<=s['baseline_missing']}
 pd.DataFrame(preds).to_csv(O/'predictions.csv',index=False);scored.to_csv(O/'scored_rows.csv',index=False);changes=scored[scored.baseline!=scored.candidate].copy();changes['effect']=np.where(changes.candidate==changes.truth,'fixed',np.where(changes.baseline==changes.truth,'regressed','wrong_to_wrong'));changes.to_csv(O/'changed_rows.csv',index=False)
 write(O/'transitions.json',transitions);write(O/'runtime.json',timing);write(O/'report.json',{'metrics':metrics,'transitions':summaries,'gates':gates,'decision':'keep_provisional_for_independent_review' if all(gates.values()) else 'do_not_promote_keep_production_baseline','production_unchanged':all(sha(R/p)==h for p,h in production.items()),'scope':'four routes new to smoothing experiment, NOT new to current model training; transition brackets are sensor-proxy sensitivity diagnostics not official timing truth','execution_exit_status':0})
 print(json.dumps({'metrics':metrics['all_extra'],'transition_summary':summaries,'gates':gates},indent=2))
if __name__=='__main__':main()
