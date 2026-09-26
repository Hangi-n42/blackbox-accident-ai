"""Independent saved-prediction/metric/mass/protection audit; no fitting."""
import run
import numpy as np,pandas as pd,joblib
O,R,read,write=run.O,run.R,run.read,run.write
f=read(O/'freeze.json');assert all(run.sha(R/p)==h for p,h in f['inputs'].items())
df=pd.read_csv(O/'predictions.csv',dtype={'id':str});assert not df.duplicated(['scope','id','sample_index']).any()
assert len(df)==3862
baseline=pd.read_csv(run.B/'comparison_predictions.csv',dtype={'id':str})
joined=df.merge(baseline,on=['scope','id','sample_index'],suffixes=('_new','_old'),validate='one_to_one')
for key in ['truth','baseline','group','sensor_accel','sensor_speed','steer_truth','steer_prediction']:
 a=joined[key+'_new'];b=joined[key+'_old'];assert ((a==b)|(a.isna()&b.isna())).all(),key
for scope,part in df.groupby('scope'):
 for sid,g in part.groupby('id'):
  ctx='zod' if scope=='zod_heldout' else sid if scope=='public_oof' else scope
  ix=g.sample_index.to_numpy(int);prob=np.load(O/'models'/(ctx+'_'+sid+'.npz'))['prob'][ix]
  assert np.array_equal(prob.argmax(1),g.candidate.to_numpy())
  assert np.allclose(prob,g[[f'candidate_p{k}' for k in range(4)]].to_numpy(),atol=1e-14,rtol=0)
metrics=read(O/'metrics.json')
for row in metrics:
 g=df[df.scope==row['scope']]
 if row['group']!='all':g=g[g.group==row['group']]
 cm=np.zeros((4,4),int)
 for y,p in zip(g.truth,g[row['variant']]):cm[int(y),int(p)]+=1
 assert cm.tolist()==row['confusion']
 den=cm.sum(0)+cm.sum(1);f1=np.divide(2*cm.diagonal(),den,out=np.zeros(4,float),where=den>0)
 assert np.allclose(f1,row['class_f1'],atol=1e-14) and abs(f1.mean()-row['macro_f1'])<1e-14
 if row['scope']=='public_oof':
  q=g[g.truth!=3];scm=np.zeros((3,3),int)
  for y,p in zip(q.steer_truth,q.steer_prediction):scm[int(y),int(p)]+=1
  den=scm.sum(0)+scm.sum(1);sf=np.divide(2*scm.diagonal(),den,out=np.zeros(3,float),where=den>0).mean()
  assert abs(.7*f1.mean()+.3*sf-row['S3'])<1e-14
logs=read(O/'execution.json')
for c,log in zip(f['contexts'],logs):
 assert c['name']==log['name']
 b=joblib.load(R/c['base']);m=joblib.load(O/'models'/(c['name']+'.joblib'))
 assert b[-1].get_params()==m[-1].get_params()
 for a in ['mean_','scale_','var_','n_samples_seen_']:assert np.array_equal(getattr(b[0],a),getattr(m[0],a))
 n=log['n_original'];assert n==len(c['selection']) and log['n_augmented']==13*n
 assert np.isclose(log['sample_weight_sum'],n) and np.allclose(log['class_effective_masses'],n/4)
 assert not log['warnings']
checks={'fits':len(logs),'rows':len(df),'metric_rows_recomputed':len(metrics),'probability_files_match_table':True,'baseline_labels_groups_steering_unchanged':True,'sample_and_class_loss_mass_unchanged':True,'model_params_and_scaler_unchanged':True,'protected_hashes_unchanged':True,'convergence_warnings':0,'new_downloads':0,'operational_changes':0,'independent_performance_claim':False}
write(O/'final_checks.json',checks);print(__import__('json').dumps(checks,indent=2))
