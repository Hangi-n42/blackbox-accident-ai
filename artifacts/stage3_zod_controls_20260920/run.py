"""Two preregistered controls of the completed ZOD mixing experiment."""
from pathlib import Path
import sys,json,hashlib,copy,time,warnings
from datetime import datetime,timezone
import numpy as np,pandas as pd,joblib
from sklearn.base import clone
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import f1_score
from sklearn.exceptions import ConvergenceWarning
from threadpoolctl import threadpool_limits
O=Path(__file__).resolve().parent;R=O.parents[1];B=R/'artifacts/stage3_zod_20260920'
sys.path.insert(0,str(B));import compare as old
read=old.read;write=old.write;sha=old.sha
ALPHA=.20
VARIANTS=['comma','original_mix','scaler_fixed','mass_capped']
def load():
 cs,_=old.prior.data('dis');manifest=read(B/'training_manifest.json');z={}
 for r in read(B/'split_manifest.json'):
  if r['role']!='excluded':
   d=dict(np.load(B/'labels'/(r['id']+'.npz')));z[r['id']]={**d,'x':np.load(B/'features'/(r['id']+'.npz'))['base']}
 contexts=[{'name':'zod','selection':manifest['comma_selection'],'base':B/'models/comma_only.joblib','mix':B/'models/comma_plus_zod.joblib'}]
 for sp in old.prior.make_splits(cs):
  p=old.P/'coverage_control/curated_usable'/sp['name']
  contexts.append({'name':sp['name'],'selection':read(p/'training_manifest.json')['selection'],'base':p/'model.joblib','mix':B/'models'/(sp['name']+'_plus_zod.joblib')})
 return cs,z,contexts,manifest['zod_selection']
def matrices(cs,z,ctx,zsel):
 sel=ctx['selection'];x=np.stack([cs[s]['x'][i] for s,i in sel]);y=np.array([cs[s]['y'][i] for s,i in sel]);pub=np.array([cs[s]['public'] for s,i in sel])
 zx=np.stack([z[s]['x'][i] for s,i in zsel]);zy=np.array([z[s]['accel_candidate'][i] for s,i in zsel]);return np.concatenate([x,zx]),np.r_[y,zy],pub,len(y)
def weights(y,pub,n0):
 # sklearn1.5.2 multiplies sample_weight by unweighted class_weight='balanced'.
 # Divide by that known multiplier so the final effective loss is the specified q.
 basew=compute_sample_weight('balanced',y[:n0]);balance=compute_sample_weight('balanced',y)
 q=np.zeros(len(y));q[:n0]=basew;rows=[]
 for k in range(4):
  ci=np.flatnonzero((y[:n0]==k)&~pub);pi=np.flatnonzero((y[:n0]==k)&pub);zi=np.flatnonzero(y[n0:]==k)+n0
  mc=float(basew[ci].sum());mp=float(basew[pi].sum());a=ALPHA if len(zi) else 0.
  q[ci]*=1-a
  if len(zi):q[zi]=a*mc/len(zi)
  assert np.isclose(q[y==k].sum(),n0/4) and np.allclose(q[pi],basew[pi])
  rows.append({'class':k,'comma_n':len(ci),'public_n':len(pi),'zod_n':len(zi),'baseline_class_mass':n0/4,'comma_mass':float(q[ci].sum()),'public_mass':float(q[pi].sum()),'zod_mass':float(q[zi].sum()),'final_class_mass':float(q[y==k].sum()),'zod_fraction_of_comma_mass':a,'zod_fraction_of_class_mass':float(q[zi].sum()/(n0/4))})
 sw=q/balance;assert np.allclose(sw*balance,q) and np.isclose(q.sum(),n0)
 return sw,rows
def freeze():
 assert not (O/'freeze.json').exists(),'Preserve frozen experiment'
 cs,z,contexts,zsel=load();files={**read(B/'freeze.json')['protected'],**read(B/'freeze.json')['label_files'],**read(B/'comparison_inputs.json')}
 files.update({str(p.relative_to(R)):sha(p) for p in B.glob('features/*.npz')})
 for p in [B/'freeze.json',B/'split_manifest.json',B/'training_manifest.json',B/'comparison_predictions.csv',B/'comparison_metrics.json',B/'comparison_changes.json',B/'compare.py',B/'qa.py',Path(__file__)]:files[str(p.relative_to(R))]=sha(p)
 manifests=[]
 for c in contexts:
  for p in [c['base'],c['mix']]:files[str(p.relative_to(R))]=sha(p)
  x,y,pub,n0=matrices(cs,z,c,zsel);sw,masses=weights(y,pub,n0)
  manifests.append({'name':c['name'],'selection':c['selection'],'baseline_model':str(c['base'].relative_to(R)),'original_mix_model':str(c['mix'].relative_to(R)),'n_baseline':n0,'n_mixed':len(y),'class_masses':masses})
 write(O/'freeze.json',{'utc':datetime.now(timezone.utc).isoformat(),'variants':VARIANTS,
 'step1':'Same mixed rows,balanced weights,C and solver; only replace refit mixed scaler by that fold baseline comma scaler. Fit classifier only; never call Pipeline.fit.',
 'step2':'Same frozen scaler and same mixed rows. Effective baseline balanced weight=N0/(4*n0_class). Within each class reserve20% of comma-only baseline mass for ZOD if present,retain80% for comma. Public40 labels if present keep original effective weights. Without ZOD class retain100% baseline. Thus total perclass=N0/4 and total=N0. No search.',
 'alpha':ALPHA,'sklearn':'1.5.2','class_weight_handling':'Keep classifier class_weight=balanced; sample_weight=desired_effective_weight/(Nmixed/(4*nmixed_class)). Verified installed1.5.2 multiplies these exactly.',
 'inference':'DIS864 and existing10Hz caches only; all features,labels,indices,splits and fixed steering unchanged.',
 'exposure':'All ZOD evaluation clips now development diagnostic, not independent test. Public and comma date folds also development exposed. No production adoption or independent performance claim.',
 'interpretation':'Step1 isolates scaler effect conditional on original mixing. Step2 jointly changes loss mass and within-class source share; cannot distinguish those two causes or prove domain/label correctness. No additional ablation/grid.',
 'metrics':'Public S3 and perclass F1; each comma date/vehicle; ZOD and perdate/clip. New opposites,corrected,newwrong,resolved opposite->correct vs->CONSTANT vs otherwrong against comma,original_mix and previous step. Opposite->CONSTANT stays wrong.',
 'zod_selection':zsel,'contexts':manifests,'inputs':files})
 print('frozen two controls; alpha=.20;9 contexts; no predictions computed',flush=True)
def run(variant):
 assert variant in ['scaler_fixed','mass_capped']
 if variant=='mass_capped':assert (O/'scaler_fixed/checks.json').exists(),'Run first control before second'
 dest=O/variant;dest.mkdir();cs,z,contexts,zsel=load();frozen=read(O/'freeze.json');assert frozen['alpha']==ALPHA
 assert all(sha(R/p)==h for p,h in frozen['inputs'].items())
 df=pd.read_csv(B/'comparison_predictions.csv',dtype={'id':str});rows=[];logs=[]
 for c in contexts:
  name=c['name'];base=joblib.load(c['base']);mix=joblib.load(c['mix']);x,y,pub,n0=matrices(cs,z,c,zsel)
  m=copy.deepcopy(base);m.steps[-1]=(m.steps[-1][0],clone(base[-1]));sw=None;mass=[]
  if variant=='mass_capped':sw,mass=weights(y,pub,n0)
  start=time.monotonic()
  with warnings.catch_warnings(record=True) as ws:
   warnings.simplefilter('always');m[-1].fit(m[0].transform(x),y,sample_weight=sw)
  assert not any(issubclass(w.category,ConvergenceWarning) for w in ws)
  assert m[-1].get_params()==base[-1].get_params()
  for a in ['mean_','scale_','var_','n_samples_seen_']:assert np.array_equal(getattr(m[0],a),getattr(base[0],a))
  joblib.dump(m,dest/(name+'.joblib'));logs.append({'name':name,'n':len(y),'seconds':time.monotonic()-start,'warnings':[str(w.message) for w in ws],'iterations':m[-1].n_iter_.tolist(),'class_masses':mass})
  part=df[df.scope=='zod_heldout'] if name=='zod' else df[df.id==name] if name.startswith('OPEN') else df[df.scope==name]
  for sid,g in part.groupby('id',sort=False):
   xx=z[sid]['x'] if name=='zod' else cs[sid]['x'];bp=base.predict_proba(xx);mp=mix.predict_proba(xx);npred=m.predict_proba(xx)
   cache=np.load(B/'models'/(sid+'_heldout.npz' if name=='zod' else name+'_'+sid+'.npz'))
   assert np.array_equal(bp,cache['baseline']) and np.array_equal(mp,cache['candidate']),(name,sid)
   ix=g.sample_index.to_numpy(int);assert np.array_equal(bp[ix].argmax(1),g.baseline) and np.array_equal(mp[ix].argmax(1),g.candidate)
   assert np.isfinite(npred).all() and np.allclose(npred.sum(1),1)
   np.savez_compressed(dest/(name+'_'+sid+'.npz'),prob=npred)
   gg=g.copy();gg['control_prediction']=npred[ix].argmax(1)
   for j in range(4):gg[f'control_p{j}']=npred[ix,j]
   rows.append(gg)
  print(variant,name,'done',flush=True)
 pd.concat(rows,ignore_index=True).to_csv(dest/'predictions.csv',index=False);write(dest/'execution.json',logs)
 assert all(sha(R/p)==h for p,h in frozen['inputs'].items())
 write(dest/'checks.json',{'completed':True,'baseline_and_original_mix_probability_replay_bit_exact':True,'scaler_exact_baseline':True,'classifier_params_unchanged':True,'inputs_production_old_results_unchanged':True,'convergence_warnings':0,'class_mass_invariants_pass':variant=='mass_capped'})
def evaluate():
 base=pd.read_csv(B/'comparison_predictions.csv',dtype={'id':str});keys=['scope','id','sample_index'];df=base.rename(columns={'baseline':'comma','candidate':'original_mix'})
 for v in ['scaler_fixed','mass_capped']:
  p=pd.read_csv(O/v/'predictions.csv',dtype={'id':str});q=p[keys+['control_prediction']+[f'control_p{j}' for j in range(4)]].rename(columns={'control_prediction':v,**{f'control_p{j}':f'{v}_p{j}' for j in range(4)}})
  df=df.merge(q,on=keys,validate='one_to_one')
 assert len(df)==len(base)==3862
 scores=[];changes=[];scopes=list(df.groupby('scope',sort=False));geo=read(B/'geographic_stratum_pre_results.json');ids={r['heldout'] for r in geo['rows'] if r['geographically_separated']};scopes.append(('zod_geographic_diagnostic',df[(df.scope=='zod_heldout')&df.id.isin(ids)]))
 for scope,part in scopes:
  for group,g in [('all',part),*list(part.groupby('group'))] if scope!='public_oof' else [('all',part)]:
   for v in VARIANTS:
    m=old.metric(g.truth,g[v]);r={'scope':scope,'group':group,'variant':v,**m}
    if scope=='public_oof':
     keep=g.truth!=3;sf=float(f1_score(g.loc[keep,'steer_truth'],g.loc[keep,'steer_prediction'],labels=range(3),average='macro',zero_division=0));r.update(steer_f1=sf,S3=.7*m['macro_f1']+.3*sf)
    # Arithmetic check independent of sklearn's F1 implementation.
    cm=np.array(m['confusion']);den=cm.sum(0)+cm.sum(1);mf=np.divide(2*cm.diagonal(),den,out=np.zeros(4,float),where=den>0).mean();assert abs(mf-m['macro_f1'])<1e-12
    scores.append(r)
   for v,ref in [('original_mix','comma'),('scaler_fixed','comma'),('scaler_fixed','original_mix'),('mass_capped','comma'),('mass_capped','original_mix'),('mass_capped','scaler_fixed')]:
    before=old.opposite(g.truth,g[ref]);after=old.opposite(g.truth,g[v]);resolved=before&~after
    changes.append({'scope':scope,'group':group,'variant':v,'reference':ref,'new_opposite':int((after&~before).sum()),'resolved_opposite':int(resolved.sum()),'resolved_to_correct':int((resolved&(g[v]==g.truth)).sum()),'resolved_to_constant_wrong':int((resolved&(g[v]==2)&(g[v]!=g.truth)).sum()),'resolved_to_other_wrong':int((resolved&(g[v]!=2)&(g[v]!=g.truth)).sum()),'corrected':int(((g[ref]!=g.truth)&(g[v]==g.truth)).sum()),'new_wrong':int(((g[ref]==g.truth)&(g[v]!=g.truth)).sum()),'ad_correct':int((g.truth.isin([0,1])&(g[v]==g.truth)).sum())})
 # Completed metrics must reproduce; never overwrite their artifacts.
 for oldrow in read(B/'comparison_metrics.json'):
  new=next(s for s in scores if (s['scope'],s['group'],s['variant'])==(oldrow['scope'],oldrow['group'],'comma' if oldrow['variant']=='baseline' else 'original_mix'))
  assert new['confusion']==oldrow['confusion'] and abs(new['macro_f1']-oldrow['macro_f1'])<1e-12
 df.to_csv(O/'predictions.csv',index=False);write(O/'metrics.json',scores);write(O/'changes.json',changes)
 pd.DataFrame(scores).to_csv(O/'metrics.csv',index=False);write(O/'checks.json',{'completed':True,'rows':len(df),'metrics_arithmetic_checks':len(scores),'old_results_reproduced':True,'protected_inputs_unchanged':all(sha(R/p)==h for p,h in read(O/'freeze.json')['inputs'].items()),'zod_exposure':'development_diagnostic','adopted':False})
 assert read(O/'checks.json')['protected_inputs_unchanged']
 print(pd.DataFrame(scores).query("group=='all'")[['scope','variant','macro_f1','opposite','S3']].to_string(index=False))
if __name__=='__main__':
 with threadpool_limits(limits=2):
  if sys.argv[1]=='freeze':freeze()
  elif sys.argv[1]=='evaluate':evaluate()
  else:run(sys.argv[1])
