from pathlib import Path
import json,hashlib
import numpy as np,pandas as pd,joblib
O=Path(__file__).resolve().parent;B=O.parent;R=B.parents[1]
d=pd.read_csv(B/'predictions.csv');f=json.loads((B/'freeze.json').read_text());prov=json.loads((B/'model_provenance.json').read_text())
for path,h in f['inputs'].items():
 with (R/path).open('rb') as z:assert hashlib.file_digest(z,'sha256').hexdigest()==h,path
models=[joblib.load(R/prov['production']['path'])['accel'],joblib.load(R/prov['expanded']['path'])]
x=np.stack([np.load(B/'features'/(r.segment.replace('/','_').replace('|','_')+'.npz'))['base'][r.center_index] for r in d.itertuples()])
rows=[];contrib=[];logits=[]
for name,m in zip(['production','expanded'],models):
 # Algebraic readback of existing 18 inputs, not a new validation batch or model.
 s,c=m[0],m[-1];z=((x.astype(np.float64)-s.mean_)/s.scale_);l=z@c.coef_.T+c.intercept_;logits.append(l)
 pp=np.exp(l-l.max(1,keepdims=True));pp/=pp.sum(1,keepdims=True)
 stored=d[[f'{name}_p{k}' for k in range(4)]].to_numpy();assert np.allclose(pp,stored,atol=2e-6)
 for j,r in enumerate(d.itertuples()):
  p=stored[j];order=np.argsort(p);win=int(order[-1]);runner=int(order[-2]);truth=int(r.truth)
  v=(c.coef_[win]-c.coef_[truth])*z[j];inter=float(c.intercept_[win]-c.intercept_[truth])
  contrib.append({'window':r.window,'model':name,'predicted':win,'truth':truth,'intercept':inter,'logit_pred_minus_truth':float(l[j,win]-l[j,truth]),'temporal_blocks':v.reshape(6,144).sum(1).tolist(),'spatial_rows':v.reshape(6,3,4,4,3).sum((0,2,3,4)).tolist()})
  rows.append({'window':r.window,'vehicle':r.vehicle,'model':name,'truth':truth,'prediction':win,'runner_up':runner,'p_winner':p[win],'p_runner':p[runner],'probability_margin':p[win]-p[runner],'p_truth':p[truth],'pred_vs_truth_logit_margin':l[j,win]-l[j,truth]})
out=pd.DataFrame(rows);out.to_csv(O/'margins.csv',index=False)
# Both decision functions in the SAME production-standardized coordinate system.
s0=models[0][0];weights=[]
for m in models:weights.append(m[-1].coef_*s0.scale_[None,:]/m[0].scale_[None,:])
cos=[]
for a in range(4):
 for b in range(a+1,4):
  u=weights[0][a]-weights[0][b];v=weights[1][a]-weights[1][b]
  cos.append({'classes':[a,b],'cosine_same_coordinates':float(u@v/(np.linalg.norm(u)*np.linalg.norm(v)))})
cs={r['id']:r for r in json.loads((R/'artifacts/stage3_training_basis_20260917/cases.json').read_text())}
old_segments={r['segment'] for r in prov['production']['training_route_manifest']};new_segments={cs[s]['segment'] if 'segment' in cs[s] else cs[s].get('raw_path','') for s,i in prov['expanded']['training_selection']}
# Use route IDs for source identity without inferring frame/label equality.
old_routes={r['route'] for r in prov['production']['training_route_manifest']};new_routes=set(prov['expanded']['training_routes'])
summary={'same_18_labels':bool((d.production==d.expanded).all()),'algebra_reproduces_saved_probabilities_atol':2e-6,'same_hyperparameters':models[0][-1].get_params()==models[1][-1].get_params(),'classifier_parameters':models[0][-1].get_params(),'common_training_route_count':len(old_routes&new_routes),'production_training_route_count':len(old_routes),'expanded_training_route_count':len(new_routes),'production_only_routes':sorted(old_routes-new_routes),'expanded_only_routes':sorted(new_routes-old_routes),'pairwise_weight_cosines':cos,'margin_summary':{},'remaining_gap_wrong':{}}
for name in ['production','expanded']:
 for vehicle in ['RAV4','Civic']:
  a=out[(out.model==name)&(out.vehicle==vehicle)];summary['margin_summary'][name+'_'+vehicle]={'min':float(a.probability_margin.min()),'median':float(a.probability_margin.median()),'max':float(a.probability_margin.max()),'winner_p_min':float(a.p_winner.min()),'winner_p_max':float(a.p_winner.max())}
for j,r in enumerate(d.itertuples()):
 if r.truth!=r.production:
  pred=int(r.production);y=int(r.truth);old=float(logits[0][j,pred]-logits[0][j,y]);new=float(logits[1][j,pred]-logits[1][j,y]);summary['remaining_gap_wrong'][str(r.window)]={'production':old,'expanded':new,'gap_change':new-old}
(O/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n');(O/'contributions.json').write_text(json.dumps(contrib,indent=2)+'\n')
print(json.dumps(summary,ensure_ascii=False,indent=2));print(out.to_string(index=False))
