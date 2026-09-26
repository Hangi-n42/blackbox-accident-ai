"""One frozen 5-sample probability mean candidate; no fitting or production edits."""
import json,hashlib,sys,time
from pathlib import Path
import numpy as np,pandas as pd,joblib,cv2
from scipy.ndimage import uniform_filter1d
from sklearn.metrics import f1_score,confusion_matrix
from threadpoolctl import threadpool_limits
O=Path(__file__).resolve().parent;R=O.parents[1];D=R/'artifacts/pipeline_diagnosis_20260917';RELEASE=R/'artifacts/submissions/verify_v6'
sys.path.insert(0,str(RELEASE/'model/stage2/code'))
from solution.stage3_v5_compatible import extract_motion,ACCEL,transfer_features

def read(p):return json.loads(p.read_text())
def write(p,d):p.write_text(json.dumps(d,ensure_ascii=False,indent=2))
def sha(p):return hashlib.file_digest(p.open('rb'),'sha256').hexdigest()
def smooth(p):return uniform_filter1d(p,size=5,axis=0,mode='nearest')
def metrics(a,b):
 a=np.asarray(a);b=np.asarray(b);cm=confusion_matrix(a,b,labels=list(ACCEL));n=len(a)
 return {'n':n,'correct':int((a==b).sum()),'macro_f1':float(f1_score(a,b,labels=list(ACCEL),average='macro',zero_division=0)), 'per_class_f1':dict(zip(map(str,ACCEL),map(float,f1_score(a,b,labels=list(ACCEL),average=None,zero_division=0)))),'confusion':cm.tolist(),'constant_errors':int(((a=='CONSTANT')&(a!=b)).sum()),'opposite_accel_errors':int((((a=='ACCELERATING')&(b=='DECELERATING'))|((a=='DECELERATING')&(b=='ACCELERATING'))).sum()),'stopped_recall':float((b[a=='STOPPED']=='STOPPED').mean()) if (a=='STOPPED').any() else None}

def main():
 assert not (O/'freeze.json').exists(),'Use a fresh output folder for another experiment'
 # Check unchanged production and truth before scoring. Do not rehash videos.
 previous=read(D/'freeze.json')
 for p,h in previous['files'].items():assert sha(R/p)==h,p
 model_path=RELEASE/'model/stage3/motion_model.joblib'
 config={'candidate':'centered arithmetic mean of 5 probability rows at 10Hz (nominal 0.5s, first-to-last center span 0.4s)','boundary':'nearest edge repeat within each video; no cross-video mixing','lookahead_s':.2,'selection':'one candidate only, no threshold fitting; RAV4 development then Civic comparison then public','acceptance':'Civic macro F1 and CONSTANT F1 strictly improve, CONSTANT errors decrease; public macro F1 does not decrease; report opposite errors and STOPPED recall','model_sha256':sha(model_path),'truth_manifest_sha256':sha(D/'evaluation_manifest.json'),'script_sha256':sha(Path(__file__)),'production_changed':False,'training':False}
 write(O/'freeze.json',config)
 # Minimal numerical checks: constant signal, normalization and isolated impulse average.
 p=np.tile([.1,.2,.3,.4],(9,1));assert np.allclose(smooth(p),p)
 z=np.zeros((9,1));z[4]=1;assert np.allclose(smooth(z)[2:7],.2)
 assert np.allclose(smooth(p).sum(1),1)
 model=joblib.load(model_path);classes=np.asarray(model['accel'].classes_,int);assert sorted(classes.tolist())==[0,1,2,3]
 cv2.setNumThreads(2);rows=[];timing=[];saved=[]
 cases=read(D/'comma_cases.json');cases.sort(key=lambda r:0 if r['split']=='train_candidate' else 1)
 public_truth=pd.read_csv(R/'Baseline/data/stage3/labels.csv')
 jobs=[dict(r,kind='comma') for r in cases]+[dict(id=id,input_path=f'artifacts/public_eval_10hz/stage3/videos/{id}.mp4',kind='public') for id in sorted(public_truth.ID.unique())]
 for r in jobs:
  id=r['id'];start=time.perf_counter();cache=R/'artifacts/data_pilot_20260916/comma_experiment/mac_public_features'/(id+'.npy')
  if r['kind']=='public':x=np.load(cache);source='existing public feature cache'
  else:x=extract_motion(R/r['input_path'],source_fps=10.);source='current production feature extraction'
  if 'motion_regressor' in model:x=transfer_features(x,model['motion_regressor'])
  ax=transfer_features(x,model['accel_motion_regressor']) if 'accel_motion_regressor' in model else x
  with threadpool_limits(limits=2):
   base=model['accel'].predict(ax).astype(int);p=model['accel'].predict_proba(ax)
  assert np.array_equal(classes[p.argmax(axis=1)],base),'Probability argmax differs from deployed predict'
  elapsed=time.perf_counter()-start;t=time.perf_counter();sp=smooth(p);candidate=classes[sp.argmax(axis=1)];overhead=time.perf_counter()-t
  oldpath=D/'comma_predictions.csv' if r['kind']=='comma' else R/'artifacts/mac_experiments/baseline_20260916/stage3.csv'
  old=pd.read_csv(oldpath);old=old[old.ID==id].sort_values('sample_index');assert len(old)==len(base)
  assert np.array_equal(ACCEL[base],old.accel_label.to_numpy()),f'Baseline reproduction failure {id}'
  np.savez_compressed(O/(id+'_probabilities.npz'),classes=classes,probabilities=p,smoothed_probabilities=sp)
  frame=pd.DataFrame({'ID':id,'sample_index':np.arange(len(base)),'baseline_accel':ACCEL[base],'candidate_accel':ACCEL[candidate],'baseline_steer':old.steer_label.to_numpy(),'candidate_steer':old.steer_label.to_numpy()});saved.append(frame)
  if r['kind']=='comma':
   labels=np.load(D/r['labels_npz']);idx=np.flatnonzero(labels['diagnostic_accel_mask']);truth=ACCEL[labels['accel_candidate'][idx]];scope='rav4_development' if r['split']=='train_candidate' else 'civic_comparison'
  else:
   g=public_truth[public_truth.ID==id];idx=g.sample_index.to_numpy();truth=g.accel_label.to_numpy();scope='public'
  rows.extend({'ID':id,'scope':scope,'sample_index':int(i),'truth':str(a),'baseline':str(ACCEL[b]),'candidate':str(ACCEL[c])} for i,a,b,c in zip(idx,truth,base[idx],candidate[idx]))
  timing.append({'id':id,'frames':len(base),'feature_source':source,'feature_and_prediction_s':elapsed,'postprocess_s':overhead,'all_frame_baseline_matches':True,'changed_all_frames':int((base!=candidate).sum())});print(id,scope,'scored',len(idx),'changed',int((base[idx]!=candidate[idx]).sum()),flush=True)
 scored=pd.DataFrame(rows);pred=pd.concat(saved,ignore_index=True);assert (pred.baseline_steer==pred.candidate_steer).all()
 scored.to_csv(O/'scored_rows.csv',index=False);pred.to_csv(O/'predictions.csv',index=False);write(O/'runtime.json',timing)
 results={}
 for scope in ['rav4_development','civic_comparison','comma_all','public']:
  s=scored[scored.scope!='public'] if scope=='comma_all' else scored[scored.scope==scope]
  results[scope]={k:metrics(s.truth,s[k]) for k in ['baseline','candidate']}
 for id,s in scored.groupby('ID'):results[id]={k:metrics(s.truth,s[k]) for k in ['baseline','candidate']}
 b=results['civic_comparison']['baseline'];c=results['civic_comparison']['candidate'];pub=results['public'];gates={'civic_macro_improved':c['macro_f1']>b['macro_f1'],'civic_constant_f1_improved':c['per_class_f1']['CONSTANT']>b['per_class_f1']['CONSTANT'],'civic_constant_errors_reduced':c['constant_errors']<b['constant_errors'],'public_no_regression':pub['candidate']['macro_f1']>=pub['baseline']['macro_f1']}
 changes=scored[scored.baseline!=scored.candidate].copy();changes['effect']=np.where(changes.candidate==changes.truth,'fixed',np.where(changes.baseline==changes.truth,'regressed','wrong_to_wrong'));changes.to_csv(O/'changed_scored_rows.csv',index=False)
 report={'results':results,'gates':gates,'decision':'eligible_for_further_review_not_independent_validation' if all(gates.values()) else 'reject_keep_baseline','baseline_reproduction_rows':len(pred),'steering_unchanged':True,'numerical_checks_passed':True,'production_files_unchanged':all(sha(R/p)==h for p,h in previous['files'].items()),'scope':'All datasets development-exposed, comma used in existing model training. Proxy timestamps not independent events. No independent or official combined score.'}
 write(O/'report.json',report);print(json.dumps({'scopes':{k:results[k] for k in ['rav4_development','civic_comparison','comma_all','public']},'gates':gates,'decision':report['decision']},indent=2))
if __name__=='__main__':main()
