"""One data-only comparison; preserve all previous results and production models."""
from qa import O,R,read,write
import importlib.util,time,hashlib,sys,warnings
import numpy as np,pandas as pd,joblib
from sklearn.base import clone
from sklearn.metrics import f1_score,confusion_matrix
from sklearn.exceptions import ConvergenceWarning
from threadpoolctl import threadpool_limits
P=R/'artifacts/stage3_state_learning_20260919'
spec=importlib.util.spec_from_file_location('prior',P/'run.py');prior=importlib.util.module_from_spec(spec);spec.loader.exec_module(prior)
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def metric(y,p):
 y=np.asarray(y);p=np.asarray(p);cm=confusion_matrix(y,p,labels=range(4));den=int(np.isin(y,[0,1]).sum());opp=int(cm[0,1]+cm[1,0])
 return {'n':len(y),'class_support':np.bincount(y,minlength=4).tolist(),'macro_f1':float(f1_score(y,p,labels=range(4),average='macro',zero_division=0)),
  'observed_class_macro_f1':float(f1_score(y,p,labels=np.unique(y),average='macro',zero_division=0)),
  'class_f1':f1_score(y,p,labels=range(4),average=None,zero_division=0).tolist(),'confusion':cm.tolist(),
  'opposite':opp,'opposite_denominator':den,'opposite_rate':opp/den if den else None}
def opposite(y,p):return ((y==0)&(p==1))|((y==1)&(p==0))
def main():
 assert not (O/'comparison_metrics.json').exists(),'Preserve completed experiment'
 frozen=read(O/'freeze.json');cs,_=prior.data('dis');splits=prior.make_splits(cs);zrows=read(O/'split_manifest.json')
 zcs={}
 for r in zrows:
  if r['role']=='excluded':continue
  d=dict(np.load(O/'labels'/(r['id']+'.npz')));x=np.load(O/'features'/(r['id']+'.npz'))['base']
  assert len(x)==len(d['time']) and np.isfinite(x).all()
  zcs[r['id']]={**r,**d,'x':x}
 zsel=[(r['id'],i) for r in zrows if r['role']=='train' for i in range(0,len(zcs[r['id']]['x']),5) if zcs[r['id']]['accel_candidate'][i]>=0]
 zx=np.stack([zcs[s]['x'][i] for s,i in zsel]);zy=np.array([zcs[s]['accel_candidate'][i] for s,i in zsel]);assert len(zx)>0
 ext_sel=[(id,i) for id,c in cs.items() if not c['public'] for i in range(0,c['n'],5) if c['y'][i]>=0];assert len(ext_sel)==2395
 trainwrite={'comma_selection':ext_sel,'zod_selection':zsel,'zod_class_counts':np.bincount(zy,minlength=4).tolist(),'zod_train_videos':25,'zod_heldout_videos':8}
 write(O/'training_manifest.json',trainwrite)
 inputs=[R/'Baseline/data/stage3/labels.csv',R/'research/stage3_oof_external.csv',prior.B/'cases.json',prior.B/'expanded_rav4.joblib']
 inputs += [prior.F/(id+'.npz') for id in cs]
 inputs += [prior.B/c['labels_npz'] for c in cs.values() if not c['public']]
 write(O/'comparison_inputs.json',{str(p.relative_to(R)):sha(p) for p in inputs})
 template=joblib.load(prior.B/'expanded_rav4.joblib')['accel'];models=O/'models';models.mkdir(exist_ok=True)
 log=[];records=[];base_source=P/'coverage_control/curated_usable'
 def fit(sel,add,name):
  x=np.stack([cs[s]['x'][i] for s,i in sel]);y=np.array([cs[s]['y'][i] for s,i in sel]);m=clone(template)
  if add:x=np.concatenate([x,zx]);y=np.concatenate([y,zy])
  st=time.monotonic()
  with warnings.catch_warnings(record=True) as ws:
   warnings.simplefilter('always');m.fit(x,y)
  assert not any(issubclass(w.category,ConvergenceWarning) for w in ws),name
  joblib.dump(m,models/(name+'.joblib'));log.append({'name':name,'n':len(y),'counts':np.bincount(y,minlength=4).tolist(),'seconds':time.monotonic()-st,'warnings':[str(w.message) for w in ws]})
  print('fit',name,len(y),round(time.monotonic()-st,2),flush=True);return m
 base=fit(ext_sel,False,'comma_only');candidate=fit(ext_sel,True,'comma_plus_zod')
 prod=joblib.load(R/'releases/v7/source/model/stage3/motion_model.joblib')['accel']
 for sid in frozen['heldout_ids']:
  c=zcs[sid];bp=base.predict_proba(c['x']);cp=candidate.predict_proba(c['x']);pp=prod.predict_proba(c['x'])
  np.savez_compressed(models/(sid+'_heldout.npz'),baseline=bp,candidate=cp,production=pp)
  for i in np.flatnonzero(c['diagnostic_accel_mask']):
   row={'scope':'zod_heldout','id':sid,'group':c['group'],'sample_index':int(i),'time_s':i/10,'truth':int(c['accel_candidate'][i]),'baseline':int(bp[i].argmax()),'candidate':int(cp[i].argmax()),'production':int(pp[i].argmax()),'sensor_accel':float(c['acceleration_proxy'][i]),'sensor_speed':float(c['speed_smoothed'][i])}
   for name,p in [('baseline',bp),('candidate',cp)]:row.update({f'{name}_p{k}':float(p[i,k]) for k in range(4)})
   records.append(row)
 # Public and old date diagnostics keep their existing exact split and public-label exposure policy.
 hist=pd.read_csv(R/'research/stage3_oof_external.csv');steer={(r.ID,r.sample_index):prior.STEER.index(r.steer_forest) for r in hist.itertuples()}
 for sp in splits:
  src=base_source/sp['name'];sel=read(src/'training_manifest.json')['selection'];base=joblib.load(src/'model.joblib');candidate=fit(sel,True,sp['name']+'_plus_zod')
  write(models/(sp['name']+'_selection.json'),{'existing':sel,'new_zod':zsel})
  for sid in sp['held']:
   c=cs[sid];bp=base.predict_proba(c['x']);assert np.array_equal(bp,np.load(src/(sid+'.npz'))['prob']),(sp['name'],sid)
   cp=candidate.predict_proba(c['x']);np.savez_compressed(models/(sp['name']+'_'+sid+'.npz'),baseline=bp,candidate=cp)
   for i in c['evaluation_indices']:
    row={'scope':'public_oof' if c['public'] else sp['name'],'id':sid,'group':c['vehicle'],'sample_index':int(i),'time_s':i/10,'truth':int(c['y'][i]),'baseline':int(bp[i].argmax()),'candidate':int(cp[i].argmax()),'sensor_accel':float(c['sensor'][i]),'sensor_speed':float(c.get('speed',np.full(c['n'],np.nan))[i])}
    for name,p in [('baseline',bp),('candidate',cp)]:row.update({f'{name}_p{k}':float(p[i,k]) for k in range(4)})
    if c['public']:row.update(steer_truth=int(c['steer'][i]),steer_prediction=steer[sid,i])
    records.append(row)
 df=pd.DataFrame(records);bo=opposite(df.truth,df.baseline);co=opposite(df.truth,df.candidate)
 df['new_opposite']=co&~bo;df['fixed_opposite']=bo&~co;df['corrected']=(df.baseline!=df.truth)&(df.candidate==df.truth);df['new_wrong']=(df.baseline==df.truth)&(df.candidate!=df.truth)
 df.to_csv(O/'comparison_predictions.csv',index=False);scores=[];changes=[]
 for scope,part in df.groupby('scope',sort=False):
  for group,g in [('all',part),*list(part.groupby('group'))] if scope!='public_oof' else [('all',part)]:
   for name in ['baseline','candidate']:
    m=metric(g.truth,g[name]);r={'scope':scope,'group':group,'variant':name,**m}
    if scope=='public_oof':
     keep=g.truth!=3;sf=float(f1_score(g.loc[keep,'steer_truth'],g.loc[keep,'steer_prediction'],labels=range(3),average='macro',zero_division=0));r.update(steer_f1=sf,S3=.7*m['macro_f1']+.3*sf)
    scores.append(r)
   changes.append({'scope':scope,'group':group,**{k:int(g[k].sum()) for k in ['new_opposite','fixed_opposite','corrected','new_wrong']}})
 write(O/'comparison_metrics.json',scores);write(O/'comparison_changes.json',changes);write(O/'execution.json',log)
 f=read(O/'freeze.json');assert all(sha(R/p)==h for p,h in {**f['protected'],**f['label_files'],**read(O/'comparison_inputs.json')}.items())
 write(O/'comparison_checks.json',{'completed':True,'protected_inputs_labels_unchanged':True,'baseline_cached_probabilities_bit_exact':True,'convergence_warning_count':0,'data_only_recipe':True,'zod_eval_rows':int((df.scope=='zod_heldout').sum()),'public_rows':int((df.scope=='public_oof').sum())})
 print(pd.DataFrame(scores).query("group=='all'")[['scope','variant','n','macro_f1','opposite','S3']].to_string(index=False),flush=True)
if __name__=='__main__':
 with threadpool_limits(limits=2):main()
