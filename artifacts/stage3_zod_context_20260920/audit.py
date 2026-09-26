"""Small independent recomputation of matching, predictions and neutralization."""
import run,json
import numpy as np,joblib
O,R,old=run.O,run.R,run.old
cs=old.load();pairs=old.read(O/'matched_manifest.json');seen={}
for p in pairs:
 windows={}
 for ds in ['comma','zod']:
  q=p[ds];d=cs[q['key']]['d'];i=q['i'];assert d['diagnostic_accel_mask'][i]
  assert np.all(d['accel_candidate'][i-5:i+6]==p['truth'])
  assert all(abs(i-j)>60 for j in seen.get(q['key'],[]));seen.setdefault(q['key'],[]).append(i)
  windows[ds]={k:d[k][i-30:i+31] for k in ['speed','acceleration_proxy']}
  assert all(len(v)==61 and np.isfinite(v).all() for v in windows[ds].values())
 for k,name,limit in [('speed','context6s_speed_rmse',1.5),('acceleration_proxy','context6s_accel_rmse',.25)]:
  value=np.sqrt(np.mean((windows['comma'][k]-windows['zod'][k])**2))
  assert np.isclose(value,p[name]) and value<=limit
models={'production':joblib.load(R/'releases/v7/source/model/stage3/motion_model.joblib')['accel'],'comma2395':joblib.load(old.Z/'models/comma_only.joblib')}
probe=old.read(O/'roi_sensitivity.json');pairmap={p['pair_id']:p for p in pairs}
for r in probe:
 p=pairmap[r['pair']];q=p[r['dataset']];m=models[r['model']];x=old.feature(cs[q['key']])[q['i']-5:q['i']+6];z=m[0].transform(x)
 indices=[block*144+r['roi']*12+j for block in range(6) for j in range(12)]
 assert len(set(indices))==72
 assert m[-1].predict(z).tolist()==r['base']
 z[:,indices]=0
 assert m[-1].predict(z).tolist()==r['probe']
checks={'complete6s_matching_recalculated':True,'central_labels_unchanged':True,'full6s_center_windows_nonoverlap':True,'roi_probe_predictions_recomputed':len(probe),'probe_dimensions_per_roi':72,
 'protected_hashes_unchanged':all(old.sha(R/p)==h for p,h in old.read(O/'freeze.json')['protected'].items()),'feature_cache_hashes_unchanged':all(old.sha(R/p)==h for p,h in old.read(O/'feature_inputs.json').items()),'exact_margin_attribution_checks':all(x['sum_checked'] for x in old.read(O/'margin_attribution.json')),'training_runs':0,'downloads':0,'operational_changes':0,'score_improvement_claim':False}
assert checks['protected_hashes_unchanged'] and checks['feature_cache_hashes_unchanged']
old.write(O/'final_checks.json',checks);print(json.dumps(checks,indent=2))
