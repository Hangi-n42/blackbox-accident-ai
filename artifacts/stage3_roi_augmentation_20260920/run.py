"""One frozen training-only ROI augmentation; cached baseline and input reuse."""
from pathlib import Path
import importlib.util,json,copy,time,warnings,sys
import numpy as np,pandas as pd,joblib
from sklearn.base import clone
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import log_loss,f1_score
from sklearn.exceptions import ConvergenceWarning
from threadpoolctl import threadpool_limits
O=Path(__file__).resolve().parent;R=O.parents[1]
spec=importlib.util.spec_from_file_location('controls',R/'artifacts/stage3_zod_controls_20260920/run.py');ctrl=importlib.util.module_from_spec(spec);spec.loader.exec_module(ctrl)
old=ctrl.old;read,write,sha=ctrl.read,ctrl.write,ctrl.sha;B=ctrl.B

def freeze():
 assert not (O/'freeze.json').exists()
 cs,z,contexts,_=ctrl.load();files=read(R/'artifacts/stage3_zod_controls_20260920/freeze.json')['inputs']
 for p in [Path(__file__),R/'artifacts/stage3_zod_context_20260920/CANDIDATE.md',R/'artifacts/stage3_zod_context_20260920/matched_manifest.json']:
  files[str(p.relative_to(R))]=sha(p)
 write(O/'freeze.json',{'candidate':'Original50percent +12singleROI masks each1/24 of original sample weight. All6blocks/4channels/3stats for a ROI standardized to0. No mask or ensemble at inference.',
  'fixed':'Existing data/labels/splits/DIS864/perfold comma scaler/logistic params/steering; fit classifier coefficients only. No ZOD training.',
  'weight_invariants':'All13copies per example; balanced class multipliers unchanged; original sample total1, class effective loss N/4; no additional weight or parameter search.',
  'success_gate':'Public S3 improves; no comma-date F1 regression; no new opposite in any diagnostic scope. Report score gain separately from gate. No independent/private improvement claim.',
  'analysis':'Compare weighted clean/masked training loss and coefficient contributions after fit. These explain learned objective and score changes, not physical causality. Full6s matched diagnostic remains development exposed.',
  'contexts':[{'name':c['name'],'selection':c['selection'],'base':str(c['base'].relative_to(R))} for c in contexts],
  'inputs':files})
 print('frozen9contexts, one setting; no outcome inspection',flush=True)

def train():
 f=read(O/'freeze.json');assert all(sha(R/p)==h for p,h in f['inputs'].items())
 cs,z,contexts,_=ctrl.load();df=pd.read_csv(B/'comparison_predictions.csv',dtype={'id':str});rows=[];logs=[]
 dest=O/'models';dest.mkdir()
 for c in contexts:
  name=c['name'];base=joblib.load(c['base']);sel=c['selection'];x=np.stack([cs[s]['x'][i] for s,i in sel]);y=np.array([cs[s]['y'][i] for s,i in sel]);n=len(y)
  zz=base[0].transform(x);xx=np.tile(zz,(13,1));yy=np.tile(y,13)
  for roi in range(12):
   ix=[b*144+roi*12+j for b in range(6) for j in range(12)];assert len(ix)==72
   block=xx[(roi+1)*n:(roi+2)*n];block[:,ix]=0
   keep=np.ones(864,bool);keep[ix]=False;assert np.array_equal(block[:,keep],zz[:,keep])
  sw=np.repeat([.5]+[1/24]*12,n);bw=compute_sample_weight('balanced',y);aw=compute_sample_weight('balanced',yy);eff=sw*aw
  assert np.allclose(aw,np.tile(bw,13)) and np.allclose(sw.reshape(13,n).sum(0),1)
  assert np.allclose(eff.reshape(13,n).sum(0),bw)
  masses=[float(eff[yy==k].sum()) for k in range(4)];assert np.allclose(masses,n/4)
  m=copy.deepcopy(base);m.steps[-1]=(m.steps[-1][0],clone(base[-1]));start=time.monotonic()
  with warnings.catch_warnings(record=True) as ws:
   warnings.simplefilter('always');m[-1].fit(xx,yy,sample_weight=sw)
  assert not any(issubclass(w.category,ConvergenceWarning) for w in ws)
  assert m[-1].get_params()==base[-1].get_params()
  for attr in ['mean_','scale_','var_','n_samples_seen_']:assert np.array_equal(getattr(m[0],attr),getattr(base[0],attr))
  objectives={}
  for mn,model in [('baseline',base),('candidate',m)]:
   pp=model[-1].predict_proba(xx);loss=log_loss(yy,pp,sample_weight=eff,labels=range(4));penalty=float((model[-1].coef_**2).sum()/(2*model[-1].C*n))
   objectives[mn]={'clean_balanced_logloss':float(log_loss(y,pp[:n],sample_weight=bw,labels=range(4))),
    'masked_balanced_logloss':float(log_loss(yy[n:],pp[n:],sample_weight=aw[n:],labels=range(4))),
    'augmented_logloss':float(loss),'l2_penalty':penalty,'augmented_objective':loss+penalty,
    'coef_l2':float(np.linalg.norm(model[-1].coef_)),
    'roi_coef_l2':np.sqrt((model[-1].coef_.reshape(4,6,12,4,3)**2).sum((0,1,3,4))).tolist()}
  assert objectives['candidate']['augmented_objective']<=objectives['baseline']['augmented_objective']+1e-5
  joblib.dump(m,dest/(name+'.joblib'))
  logs.append({'name':name,'n_original':n,'n_augmented':len(yy),'sample_weight_sum':float(sw.sum()),'class_effective_masses':masses,'seconds':time.monotonic()-start,'n_iter':m[-1].n_iter_.tolist(),'warnings':[str(w.message) for w in ws],'training_diagnostics':objectives})
  part=df[df.scope=='zod_heldout'] if name=='zod' else df[df.id==name] if name.startswith('OPEN') else df[df.scope==name]
  for sid,g in part.groupby('id',sort=False):
   data=z[sid]['x'] if name=='zod' else cs[sid]['x'];bp=base.predict_proba(data);cp=m.predict_proba(data)
   cache=np.load(B/'models'/(sid+'_heldout.npz' if name=='zod' else name+'_'+sid+'.npz'))
   assert np.array_equal(bp,cache['baseline']),(name,sid)
   ix=g.sample_index.to_numpy(int);assert np.array_equal(bp[ix].argmax(1),g.baseline)
   assert np.isfinite(cp).all() and np.allclose(cp.sum(1),1)
   np.savez_compressed(dest/(name+'_'+sid+'.npz'),prob=cp)
   gg=g.drop(columns=['candidate']+[f'candidate_p{k}' for k in range(4)]+['new_opposite','fixed_opposite','corrected','new_wrong']).copy();gg['candidate']=cp[ix].argmax(1)
   for k in range(4):gg[f'candidate_p{k}']=cp[ix,k]
   rows.append(gg)
  write(O/'execution.json',logs);print(name,'completed',round(logs[-1]['seconds'],2),'sec',flush=True)
 pd.concat(rows,ignore_index=True).to_csv(O/'predictions.csv',index=False)
 assert all(sha(R/p)==h for p,h in f['inputs'].items())
 write(O/'train_checks.json',{'fits':len(logs),'baseline_probability_replay_bit_exact':True,'class_total_and_per_sample_mass_preserved':True,'classifier_params_and_scaler_unchanged':True,'warnings':[w for row in logs for w in row['warnings']],'protected_inputs_unchanged':True})

def evaluate():
 df=pd.read_csv(O/'predictions.csv',dtype={'id':str});assert len(df)==3862
 metrics=[];changes=[]
 for scope,part in df.groupby('scope',sort=False):
  for group,g in [('all',part),*list(part.groupby('group'))] if scope!='public_oof' else [('all',part)]:
   for name in ['baseline','candidate']:
    met=old.metric(g.truth,g[name]);r={'scope':scope,'group':group,'variant':name,**met}
    if scope=='public_oof':
     keep=g.truth!=3;sf=f1_score(g.loc[keep,'steer_truth'],g.loc[keep,'steer_prediction'],labels=range(3),average='macro',zero_division=0);r.update(steer_f1=float(sf),S3=.7*met['macro_f1']+.3*sf)
    cm=np.array(met['confusion']);den=cm.sum(0)+cm.sum(1);assert abs(np.divide(2*cm.diagonal(),den,out=np.zeros(4,float),where=den>0).mean()-met['macro_f1'])<1e-12
    metrics.append(r)
   before=old.opposite(g.truth,g.baseline);after=old.opposite(g.truth,g.candidate);resolved=before&~after
   changes.append({'scope':scope,'group':group,'new_opposite':int((after&~before).sum()),'resolved_to_correct':int((resolved&(g.candidate==g.truth)).sum()),'resolved_to_constant_wrong':int((resolved&(g.candidate==2)&(g.candidate!=g.truth)).sum()),'resolved_to_other_wrong':int((resolved&(g.candidate!=2)&(g.candidate!=g.truth)).sum()),'corrected':int(((g.baseline!=g.truth)&(g.candidate==g.truth)).sum()),'new_wrong':int(((g.baseline==g.truth)&(g.candidate!=g.truth)).sum()),'constant_correct':int(((g.truth==2)&(g.candidate==2)).sum()),'constant_to_AD':int(((g.truth==2)&g.candidate.isin([0,1])).sum()),'constant_to_STOP':int(((g.truth==2)&(g.candidate==3)).sum())})
 for row in read(B/'comparison_metrics.json'):
  if row['variant']=='baseline':
   new=next(r for r in metrics if (r['scope'],r['group'],r['variant'])==(row['scope'],row['group'],'baseline'));assert new['confusion']==row['confusion']
 write(O/'metrics.json',metrics);write(O/'changes.json',changes);pd.DataFrame(metrics).to_csv(O/'metrics.csv',index=False)
 df['new_opposite']=old.opposite(df.truth,df.candidate)&~old.opposite(df.truth,df.baseline);df['corrected']=(df.baseline!=df.truth)&(df.candidate==df.truth);df['new_wrong']=(df.baseline==df.truth)&(df.candidate!=df.truth)
 df[df.baseline!=df.candidate].to_csv(O/'changed_predictions.csv',index=False)
 allm={(r['scope'],r['variant']):r for r in metrics if r['group']=='all'}
 gate={'public_S3_improved':allm['public_oof','candidate']['S3']>allm['public_oof','baseline']['S3'],
  'no_date_F1_regression':all(allm[s,'candidate']['macro_f1']>=allm[s,'baseline']['macro_f1'] for s in df.scope.unique() if s not in ['zod_heldout','public_oof']),
  'no_new_opposite_any_scope':all(r['new_opposite']==0 for r in changes if r['group']=='all')}
 write(O/'decision.json',{'gates':gate,'all_pass':all(gate.values()),'independent_performance_claim':False,'production_replaced':False})
 print(pd.DataFrame(metrics).query("group=='all'")[['scope','variant','macro_f1','opposite','S3']].to_string(index=False));print(json.dumps([r for r in changes if r['group']=='all'],indent=2))

if __name__=='__main__':
 with threadpool_limits(limits=2):
  {'freeze':freeze,'train':train,'evaluate':evaluate}[sys.argv[1]]()
