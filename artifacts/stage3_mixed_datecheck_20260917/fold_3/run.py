"""Date-group holdout; identical features/model, full and size/class-matched mixed controls."""
import json,hashlib,warnings,time
from pathlib import Path
import numpy as np,pandas as pd,joblib
from sklearn.base import clone
from sklearn.metrics import f1_score,confusion_matrix
from sklearn.exceptions import ConvergenceWarning
from threadpoolctl import threadpool_limits
O=Path(__file__).resolve().parent;R=O.parents[2];B=R/'artifacts/stage3_training_basis_20260917'
read=lambda p:json.loads(p.read_text());write=lambda p,x:p.write_text(json.dumps(x,ensure_ascii=False,indent=2));sha=lambda p:hashlib.file_digest(p.open('rb'),'sha256').hexdigest()
N=['ACCELERATING','DECELERATING','CONSTANT','STOPPED']
def metrics(y,p):
 cm=confusion_matrix(y,p,labels=range(4));counts=cm.sum(1)
 return {'n':len(y),'macro_f1':float(f1_score(y,p,labels=range(4),average='macro',zero_division=0)),'support':dict(zip(N,counts.tolist())),'confusion':cm.tolist(),'recall':{k:float(cm[i,i]/counts[i]) if counts[i] else None for i,k in enumerate(N)},'constant_errors':int(cm[2].sum()-cm[2,2]),'opposite_errors':int(cm[0,1]+cm[1,0])}
def main():
 assert not (O/'freeze.json').exists()
 cases=read(B/'cases.json');production=read(R/'artifacts/pipeline_diagnosis_20260917/freeze.json')['files'];assert all(sha(R/p)==h for p,h in production.items())
 for c in cases:c['date']=c['route'].split('|')[1][:10];c['vehicle_name']='rav4' if c['role']=='train' else 'civic'
 # Only labels and date, no model results: earliest date supporting three moving classes,
 # while the remaining training dates retain all four classes.
 hold={'rav4': '2018-08-02', 'civic': '2018-05-05'}
 for c in cases:c['experiment_role']='heldout' if c['date']==hold[c['vehicle_name']] else 'train'
 write(O/'split_manifest.json',cases)
 write(O/'freeze.json',{'selection':'predeclared alternative dates from parent freeze; same label-support rule','held_dates':hold,'vehicle_and_date_grouped':True,'route_overlap':False,'road_geographic_overlap':'not verified; do not claim independent roads or unseen vehicles','models':['rav4','civic','mixed_full','mixed_budget_rav4','mixed_budget_civic'],'feature_count':864,'seed':42,'labels':'unchanged strict proxy;2Hztrain/10Hzeval','gates':'development support only: mixed full improves heldout pooled and each vehicle F1 vs BOTH singles, publicF1 not below better single; class recall/opposites tradeoffs reported. Matched-budget models separate amount and composition; no automatic production promotion','source_models':{x:sha(B/x) for x in ['expanded_rav4.joblib','cases.json']},'production':production,'script_sha256':sha(Path(__file__))})
 X=[];Y=[];V=[];keys=[];test=[];cache={};labels={}
 for c in cases:
  p=R/c['feature_cache'] if c['feature_cache'] else B/(c['id']+'_features.npy');a=np.load(p);x=a['features'] if isinstance(a,np.lib.npyio.NpzFile) else a;d=np.load(B/c['labels_npz']);y=d['accel_candidate'];cache[c['id']]=x;labels[c['id']]=y
  if c['experiment_role']=='train':
   for i in c['training_indices']:X.append(x[i]);Y.append(y[i]);V.append(c['vehicle_name']);keys.append({'id':c['id'],'sample_index':i,'vehicle':c['vehicle_name'],'truth':int(y[i]),'route':c['route']})
  else:
   for i in c['evaluation_indices']:test.append({'id':c['id'],'sample_index':i,'vehicle':c['vehicle_name'],'truth':int(y[i])})
 X=np.array(X);Y=np.array(Y);V=np.array(V);indices={v:np.flatnonzero(V==v) for v in ['rav4','civic']};indices['mixed_full']=np.arange(len(Y))
 rng=np.random.default_rng(42)
 for vehicle in ['rav4','civic']:
  selected=[]
  for k in range(4):
   n=int(np.sum(Y[indices[vehicle]]==k));a=np.flatnonzero((Y==k)&(V=='rav4'));b=np.flatnonzero((Y==k)&(V=='civic'));na=min(len(a),n//2);nb=min(len(b),n-na);na=n-nb;assert na<=len(a)
   selected.extend(rng.choice(a,na,replace=False));selected.extend(rng.choice(b,nb,replace=False))
  indices['mixed_budget_'+vehicle]=np.array(sorted(selected));assert np.array_equal(np.bincount(Y[indices[vehicle]],minlength=4),np.bincount(Y[indices['mixed_budget_'+vehicle]],minlength=4))
 template=joblib.load(B/'expanded_rav4.joblib')['accel'];models={};fits=[];selections={}
 for name,ix in indices.items():
  assert len(ix)==len(set(ix)) and set(Y[ix])=={0,1,2,3};m=clone(template);start=time.perf_counter()
  with warnings.catch_warnings(record=True) as ws,threadpool_limits(limits=2):
   warnings.simplefilter('always');m.fit(X[ix],Y[ix])
  assert not any(issubclass(w.category,ConvergenceWarning) for w in ws)
  models[name]=m;joblib.dump({'accel':m,'experimental_only':True},O/(name+'.joblib'));fits.append({'model':name,'n':len(ix),'class_counts':np.bincount(Y[ix],minlength=4).tolist(),'vehicle_counts':{v:int((V[ix]==v).sum()) for v in ['rav4','civic']},'seconds':time.perf_counter()-start,'iterations':m.named_steps['logisticregression'].n_iter_.tolist()});selections[name]=[keys[i] for i in ix];print('fit',name,len(ix),flush=True)
 write(O/'training_selection.json',selections);write(O/'fit.json',fits)
 s=pd.DataFrame(test)
 for name,m in models.items():
  pred={id:m.predict(cache[id]).astype(int) for id in s.id.unique()};s[name]=[int(pred[z.id][z.sample_index]) for z in s.itertuples()]
 results={scope:{name:metrics(g.truth,g[name]) for name in models} for scope,g in [('pooled',s),*list(s.groupby('vehicle'))]}
 routes={id:{name:metrics(g.truth,g[name]) for name in models} for id,g in s.groupby('id')}
 public=pd.read_csv(R/'Baseline/data/stage3/labels.csv');pub=[]
 for id,g in public.groupby('ID'):
  x=np.load(R/'artifacts/data_pilot_20260916/comma_experiment/mac_public_features'/(id+'.npy'));pred={n:m.predict(x).astype(int) for n,m in models.items()}
  for z in g.itertuples():pub.append({'id':id,'sample_index':int(z.sample_index),'truth':N.index(z.accel_label),**{n:int(p[z.sample_index]) for n,p in pred.items()}})
 p=pd.DataFrame(pub);results['public']={n:metrics(p.truth,p[n]) for n in models}
 gates={scope:results[scope]['mixed_full']['macro_f1']>max(results[scope][v]['macro_f1'] for v in ['rav4','civic']) for scope in ['pooled','rav4','civic']};gates['public_no_regression']=results['public']['mixed_full']['macro_f1']>=max(results['public'][v]['macro_f1'] for v in ['rav4','civic'])
 train_routes={c['route'] for c in cases if c['experiment_role']=='train'};test_routes={c['route'] for c in cases if c['experiment_role']=='heldout'};assert train_routes.isdisjoint(test_routes)
 assert all(sha(R/p)==h for p,h in production.items())
 s.to_csv(O/'heldout_predictions.csv',index=False);p.to_csv(O/'public_predictions.csv',index=False);write(O/'route_results.json',routes);write(O/'report.json',{'results':results,'gates':gates,'held_dates':hold,'heldout_routes':len(test_routes),'train_routes':len(train_routes),'production_unchanged':True,'exit_status':0,'decision':'development_candidate_only' if all(gates.values()) else 'mixed_tradeoffs_no_adoption'})
 print(json.dumps({'results':results,'gates':gates},indent=2))
if __name__=='__main__':main()
