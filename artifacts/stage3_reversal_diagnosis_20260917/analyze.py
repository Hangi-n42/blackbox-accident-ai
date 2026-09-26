"""74 frozen new reversals; probabilities and common-reference linear margin decomposition."""
from pathlib import Path
import json,hashlib
import numpy as np,pandas as pd,joblib
from threadpoolctl import threadpool_limits
O=Path(__file__).resolve().parent;R=O.parents[1];B=R/'artifacts/stage3_training_basis_20260917';F=R/'artifacts/stage3_mixed_datecheck_20260917/fold_1'
write=lambda n,x:(O/n).write_text(json.dumps(x,ensure_ascii=False,indent=2));sha=lambda p:hashlib.file_digest(p.open('rb'),'sha256').hexdigest()
N=['ACCELERATING','DECELERATING','CONSTANT','STOPPED'];T=['raw','mean5','mean15','mean31','diff5','diff15'];C=['horizontal','vertical','magnitude','radial']

def main():
 assert not (O/'summary.json').exists()
 cases=json.load(open(B/'cases.json'));models={n:joblib.load(F/(n+'.joblib'))['accel'] for n in ['rav4','mixed_budget_rav4']}
 selected=pd.read_csv(F.parent/'opposite_changes.csv');selected=selected[(selected.fold==1)&(selected.effect=='new_opposite')&selected.id.isin(['extra_01','expanded_14'])]
 assert len(selected)==74 and (selected.truth==0).all() and (selected.mixed_budget_rav4==1).all()
 cache={};sensor={};inputs=[F/(n+'.joblib') for n in models]+[F/'training_selection.json',F.parent/'opposite_changes.csv']
 for c in cases:
  p=R/c['feature_cache'] if c['feature_cache'] else B/(c['id']+'_features.npy');a=np.load(p);cache[c['id']]=a['features'] if isinstance(a,np.lib.npyio.NpzFile) else a;inputs.append(p)
  if c['id'] in ['extra_01','expanded_14']:sensor[c['id']]=np.load(B/c['labels_npz']);inputs.append(B/c['labels_npz'])
 training=json.load(open(F/'training_selection.json'));refs=np.array([cache[z['id']][z['sample_index']] for z in training['rav4']]);reference=refs.mean(0)
 write('freeze.json',{'selected_count':74,'margin':'DECELERATING logit minus ACCELERATING logit','reference':'mean of baseline RAV4 training features; same reference for both models','decomposition':'coef/scaler.scale on identical raw features centered at common reference; includes reference margin, not causal attribution','files':{str(p.relative_to(R)):sha(p) for p in inputs},'models_unchanged':True,'no_fit':True,'script_sha256':sha(Path(__file__))})
 params={}
 for name,m in models.items():
  scale=m.named_steps['standardscaler'];lr=m.named_steps['logisticregression'];assert list(lr.classes_)==[0,1,2,3]
  w=lr.coef_/scale.scale_;b=lr.intercept_-w@scale.mean_;params[name]=(w[1]-w[0],float((w@reference+b)[1]-(w@reference+b)[0]))
 rows=[];contribs=[];full=[];controls=[]
 for id,d in sensor.items():
  x=cache[id];prob={n:m.predict_proba(x) for n,m in models.items()};pred={n:m.predict(x).astype(int) for n,m in models.items()};scores={n:m.decision_function(x) for n,m in models.items()};sel=selected[selected.id==id].sample_index.to_numpy()
  for i in range(len(x)):
   a={'id':id,'index':i,'relative_s':i/10,'speed_mps':float(d['speed_smoothed'][i]),'accel_proxy':float(d['acceleration_proxy'][i]),'proxy_truth':int(d['accel_candidate'][i]),'strict_mask':bool(d['diagnostic_accel_mask'][i]),'selected_new_reversal':bool(i in sel)}
   for n in models:
    a[n+'_prediction']=int(pred[n][i]);a.update({n+'_'+k:float(prob[n][i,j]) for j,k in enumerate(N)})
   full.append(a)
  for i in sel:
   assert pred['rav4'][i]==int(selected[(selected.id==id)&(selected.sample_index==i)].rav4.iloc[0]);assert pred['mixed_budget_rav4'][i]==1
   a=dict(full[-len(x)+i]);lo=max(0,i-23);hi=min(len(x),i+24);a['context_same_accel_fraction']=float(np.mean(d['accel_candidate'][lo:hi]==0));a['context_speed_valid']=bool(d['speed_valid'][lo:hi].all())
   cc={}
   for n,(w,bias) in params.items():
    vals=w*(x[i]-reference);margin=float(scores[n][i,1]-scores[n][i,0]);err=float(abs(vals.sum()+bias-margin));assert err<1e-5
    cc[n]=vals.reshape(6,12,4,3);a[n+'_margin']=margin;a[n+'_reference_margin']=bias;a[n+'_reconstruction_error']=err
   delta=cc['mixed_budget_rav4']-cc['rav4'];a['margin_shift']=a['mixed_budget_rav4_margin']-a['rav4_margin'];rows.append(a)
   contribs.append({'id':id,'index':int(i),'models':{n:{'temporal':dict(zip(T,v.sum((1,2,3)).tolist())),'roi':v.sum((0,2,3)).tolist(),'channel':dict(zip(C,v.sum((0,1,3)).tolist()))} for n,v in cc.items()},'delta_temporal':dict(zip(T,delta.sum((1,2,3)).tolist())),'delta_roi':delta.sum((0,2,3)).tolist(),'delta_channel':dict(zip(C,delta.sum((0,1,3)).tolist()))})
  # Same-video, same-proxy-label diagnostic controls; existing cases, not independent samples.
  valid=np.flatnonzero(d['diagnostic_accel_mask']&(d['accel_candidate']==0)&(pred['rav4']==0)&(pred['mixed_budget_rav4']==0))
  controls.append({'id':id,'both_correct_accel_count':len(valid),'indices':valid.tolist()})
 s=pd.DataFrame(rows);pd.DataFrame(full).to_csv(O/'full_context.csv',index=False);s.to_csv(O/'selected_74.csv',index=False);write('contributions.json',contribs);write('controls.json',controls)
 out={}
 for id,g in s.groupby('id'):
  cs=[c for c in contribs if c['id']==id];entry={'n':len(g),'original_predictions':g.rav4_prediction.value_counts().to_dict(),'time_ranges':[],'speed_range':[float(g.speed_mps.min()),float(g.speed_mps.max())],'acceleration_proxy_range':[float(g.accel_proxy.min()),float(g.accel_proxy.max())],'context_all_accel_count':int((g.context_same_accel_fraction==1).sum()),'context_valid_count':int(g.context_speed_valid.sum()),'prob_mean':{n:{k:float(g[n+'_'+k].mean()) for k in N} for n in models},'margins':{n:float(g[n+'_margin'].mean()) for n in models},'delta_reference_margin':params['mixed_budget_rav4'][1]-params['rav4'][1],'mean_delta_temporal':{k:float(np.mean([c['delta_temporal'][k] for c in cs])) for k in T},'mean_delta_roi':np.mean([c['delta_roi'] for c in cs],axis=0).tolist(),'mean_delta_channel':{k:float(np.mean([c['delta_channel'][k] for c in cs])) for k in C}}
  ix=np.sort(g['index'].to_numpy())
  for b in np.split(ix,np.flatnonzero(np.diff(ix)>1)+1):entry['time_ranges'].append([int(b[0])/10,int(b[-1])/10])
  out[id]=entry
 write('summary.json',out)
 assert all(sha(R/p)==h for p,h in json.load(open(O/'freeze.json'))['files'].items())
 print(json.dumps(out,ensure_ascii=False,indent=2))
if __name__=='__main__':
 with threadpool_limits(limits=2):main()
