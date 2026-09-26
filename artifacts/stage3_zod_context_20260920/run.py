"""Complete-history sensor selection, then reuse frozen feature/model analysis."""
from pathlib import Path
import sys,json
import numpy as np
O=Path(__file__).resolve().parent
R=O.parents[1]
sys.path.insert(0,str(R/'artifacts/stage3_zod_matched_20260920'))
import compare as old
from threadpoolctl import threadpool_limits
old.O=O
read,write=old.read,old.write

def select():
 assert not (O/'freeze.json').exists()
 previous=read(R/'artifacts/stage3_zod_matched_20260920/freeze.json')
 previous.update(scope='Development diagnosis only. No fitting, label changes, feature changes, or new downloads.',
  selection='Same strict center and central1s gates as previous. REQUIRE complete61samples ±3s speed and acceleration; speed RMSE<=1.5m/s and acceleration RMSE<=.25m/s2. Sort by full-context normalized error vRMSE/1.5+aRMSE/.25, then central cost and IDs. D,A,C order; max6/class, one per ZOD clip/class; full6s windows do not overlap within either source (>60 indices). C range fixed to previous selected A/D minmax±1m/s. No prediction-based selection.',
  within_comma_control='Same complete-history gates, other route; diagnostic controls may repeat, explicitly not independent.',
  scenes='Export7 source frames/window at -3,-2,-1,0,1,2,3s. AI visual straight/curve/nearby objects/context changes only; no calibrated rotation control claim. Comma steering descriptive, not ZOD-equivalent yaw.',
  candidate='Choose one follow-up hypothesis after diagnosis; no candidate fitting or score improvement claim in this run.')
 previous['protected'][str(Path(__file__).relative_to(R))]=old.sha(Path(__file__))
 for p in [old.Z/'models/comma_only.joblib',old.Z/'models/comma_plus_zod.joblib',R/'artifacts/stage3_zod_controls_20260920/mass_capped/zod.joblib']:
  previous['protected'][str(p.relative_to(R))]=old.sha(p)
 write(O/'freeze.json',previous)
 cs=old.load();prev=read(R/'artifacts/stage3_zod_matched_20260920/matched_manifest.json')
 speeds=[q[ds]['v'] for q in prev if q['truth']!=2 for ds in ['comma','zod']]
 selected=[];coverage=[];used={};eligible=[]
 for k in [1,0,2]:
  cc=old.candidates(cs,'comma',k);zz=old.candidates(cs,'zod',k)
  if k==2:zz=[z for z in zz if min(speeds)-1<=z['v']<=max(speeds)+1]
  edges=[]
  for z in zz:edges.extend(old.match(cs,z,cc))
  passed=[e for e in edges if e['context6s_matched']]
  passed.sort(key=lambda e:(e['context6s_speed_rmse']/1.5+e['context6s_accel_rmse']/.25,e['cost'],e['zod']['key'],e['zod']['i'],e['comma']['key'],e['comma']['i']))
  byvideo=[]
  for key in sorted({z['key'] for z in zz}):
   ze=[e for e in edges if e['zod']['key']==key];complete=[e for e in ze if e['context6s_complete']]
   byvideo.append({'key':key,'candidate_centers':sum(z['key']==key for z in zz),'central_matched_edges':len(ze),'complete_edges':len(complete),'passing_edges':sum(e['context6s_matched'] for e in ze),'best_complete_accel_rmse':min((e['context6s_accel_rmse'] for e in complete),default=None)})
  took=set();n=0
  for e in passed:
   if e['zod']['key'] in took:continue
   if any(any(abs(q['i']-j)<=60 for j in used.get(q['key'],[])) for q in [e['comma'],e['zod']]):continue
   n+=1;e.update(pair_id=f'{old.ACC[k][0]}{n:02}',truth=k);selected.append(e);took.add(e['zod']['key'])
   for q in [e['comma'],e['zod']]:used.setdefault(q['key'],[]).append(q['i'])
   if n==6:break
  coverage.append({'class':k,'comma_centers':len(cc),'zod_centers':len(zz),'central_edges':len(edges),'incomplete_edges':sum(not e['context6s_complete'] for e in edges),'complete_but_mismatch_edges':sum(e['context6s_complete'] and not e['context6s_matched'] for e in edges),'passing_edges':len(passed),'passing_zod_videos':len({e['zod']['key'] for e in passed}),'selected':n,'by_video':byvideo})
  eligible.extend(passed)
 for e in selected:
  q=e['comma'];route=cs[q['key']]['row']['route'];opts=[c for c in old.candidates(cs,'comma',e['truth']) if cs[c['key']]['row']['route']!=route]
  refs=[x for x in old.match(cs,q,opts) if x['context6s_matched']]
  refs.sort(key=lambda x:(x['context6s_speed_rmse']/1.5+x['context6s_accel_rmse']/.25,x['cost'],x['comma']['key'],x['comma']['i']))
  e['within_comma']=refs[0] if refs else None
  for name in ['comma','zod']:
   q=e[name];c=cs[q['key']];q.update(original_path=c['row'].get('raw_path',str((old.Z/'raw/sequences'/c['row']['id']).relative_to(R))),labels_path=str(c['labels_path'].relative_to(R)),time_s=q['i']/10,development_exposed=True,truth_kind='sensor_proxy_not_official')
 write(O/'matched_manifest.json',selected);write(O/'coverage.json',coverage)
 print(json.dumps(coverage,indent=2));print('selected',len(selected))

def analyze():
 old.analyze()
 import joblib
 cs=old.load();production=joblib.load(R/'releases/v7/source/model/stage3/motion_model.joblib')['accel']
 rows=[]
 for p in read(O/'matched_manifest.json'):
  r={'pair':p['pair_id'],'truth':p['truth']}
  for name in ['comma','zod']:
   q=p[name];d=cs[q['key']]['d'];i=q['i'];x=old.feature(cs[q['key']])[i-5:i+6]
   prob=production.predict_proba(x)
   r[name]={'center_prediction':int(production.classes_[prob[5].argmax()]),'classes':production.classes_.tolist(),'probabilities':prob.tolist(),'predictions':production.classes_[prob.argmax(1)].tolist()}
   if name=='comma':
    a=d['steering_angle'][i-30:i+31]
    r['comma_steering_degrees']={'complete':bool(np.isfinite(a).all()),'min':float(np.nanmin(a)),'max':float(np.nanmax(a)),'note':'Steering wheel angle; not calibrated camera rotation or comparable to ZOD yaw.'}
  rows.append(r)
 write(O/'production_predictions.json',rows)
 write(O/'checks.json',{'selected_all_full_context':all(p['context6s_matched'] for p in read(O/'matched_manifest.json')),'protected_hashes_unchanged':all(old.sha(R/p)==h for p,h in read(O/'freeze.json')['protected'].items()),'training_runs':0,'downloads':0,'score_improvement_claim':False})
 print(json.dumps(rows,indent=2))

if __name__=='__main__':
 with threadpool_limits(limits=2):
  if sys.argv[1]=='select':select()
  else:analyze()
