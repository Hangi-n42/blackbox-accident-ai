"""Exact coefficient-change attribution and the previous matched-pair probe."""
import run
import numpy as np,pandas as pd,joblib
from threadpoolctl import threadpool_limits
O,R,read,write=run.O,run.R,run.read,run.write
with threadpool_limits(limits=2):
 cs,z,contexts,_=run.ctrl.load();models={c['name']:(joblib.load(c['base']),joblib.load(O/'models'/(c['name']+'.joblib'))) for c in contexts}
 df=pd.read_csv(O/'changed_predictions.csv',dtype={'id':str});out=[]
 for row in df.to_dict('records'):
  name='zod' if row['scope']=='zod_heldout' else row['id'] if row['scope']=='public_oof' else row['scope'];b,c=models[name]
  x=(z if name=='zod' else cs)[row['id']]['x'][int(row['sample_index'])];zz=b[0].transform(x[None])[0]
  truth=int(row['truth']);rival=int(row['candidate'] if row['candidate']!=truth else row['baseline'])
  weight_delta=(c[-1].coef_[truth]-c[-1].coef_[rival])-(b[-1].coef_[truth]-b[-1].coef_[rival]);bias_delta=(c[-1].intercept_[truth]-c[-1].intercept_[rival])-(b[-1].intercept_[truth]-b[-1].intercept_[rival])
  values=zz*weight_delta;logits=[m.decision_function(x[None])[0] for m in [b,c]];margins=[float(s[truth]-s[rival]) for s in logits]
  assert np.isclose(values.sum()+bias_delta,margins[1]-margins[0],atol=1e-6)
  t=values.reshape(6,12,4,3)
  out.append({**{k:row[k] for k in ['scope','id','sample_index','truth','baseline','candidate','new_opposite','corrected','new_wrong']},'rival':rival,'true_minus_rival_margins':margins,'bias_change':float(bias_delta),'delta_by_block':t.sum((1,2,3)).tolist(),'delta_by_roi':t.sum((0,2,3)).tolist(),'sum_checked':True})
 write(O/'prediction_change_attribution.json',out)
 # Original four pairs remain development diagnosis; comma members are training exposed.
 pairs=read(R/'artifacts/stage3_zod_context_20260920/matched_manifest.json');matched=[];b,c=models['zod']
 for p in pairs:
  for ds in ['comma','zod']:
   q=p[ds];sid=q['key'].split(':',1)[1];i=q['i'];x=(cs if ds=='comma' else z)[sid]['x'][i-5:i+6];truth=p['truth'];rr={'pair':p['pair_id'],'dataset':ds,'id':sid,'index':i,'truth':truth}
   for name,m in [('baseline',b),('candidate',c)]:
    zz=m[0].transform(x);base=m[-1].predict(zz);changes=0;right=0;newwrong=0;fixed=0;probes=[]
    for roi in range(12):
     masked=zz.copy();ix=[bl*144+roi*12+j for bl in range(6) for j in range(12)];masked[:,ix]=0;pred=m[-1].predict(masked)
     changes+=int((pred!=base).sum());right+=int((pred==truth).sum());newwrong+=int(((base==truth)&(pred!=truth)).sum());fixed+=int(((base!=truth)&(pred==truth)).sum());probes.append(pred.tolist())
    rr[name]={'clean_predictions':base.tolist(),'clean_correct_of11':int((base==truth).sum()),'masked_correct_of132':right,'prediction_changes_of132':changes,'correct_to_wrong_probe':newwrong,'wrong_to_correct_probe':fixed,'probe_predictions':probes}
   matched.append(rr)
 write(O/'matched_diagnostic.json',matched)
 write(O/'diagnosis_checks.json',{'changed_rows_attribution_checked':len(out),'same_scaler':True,'matched_windows':len(matched),'no_candidate_selection_on_results':True})
 print('exact margin changes',len(out),'matched windows',len(matched))
