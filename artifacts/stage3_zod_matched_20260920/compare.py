"""Sensor-selected same-state pairs; descriptive comparison, no fitting."""
from pathlib import Path
import json,sys,hashlib
import numpy as np,pandas as pd,joblib
from threadpoolctl import threadpool_limits
R=Path(__file__).resolve().parents[2];O=Path(__file__).resolve().parent
B=R/'artifacts/stage3_training_basis_20260917';Z=R/'artifacts/stage3_zod_20260920';F=R/'artifacts/stage3_factor_comparison_20260918/dis'
read=lambda p:json.loads(p.read_text());write=lambda p,x:p.write_text(json.dumps(x,indent=2,ensure_ascii=False))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
ACC=['ACCELERATING','DECELERATING','CONSTANT','STOPPED'];BLOCK=['raw','mean0.5s','mean1.5s','mean3.1s','difference1s','difference3s'];CHANNEL=['fx','fy','magnitude','radial'];STAT=['median','p20','p80']
def load():
 out={}
 for ds,rows in [('comma',read(B/'cases.json')),('zod',[r for r in read(Z/'split_manifest.json') if r['role']!='excluded'])]:
  for row in rows:
   p=B/row['labels_npz'] if ds=='comma' else Z/'labels'/(row['id']+'.npz')
   out[ds+':'+row['id']]={'dataset':ds,'row':row,'labels_path':p,'d':dict(np.load(p))}
 return out
def candidates(cs,ds,k):
 out=[]
 for key,c in cs.items():
  if c['dataset']!=ds:continue
  d=c['d'];ix=np.flatnonzero(d['diagnostic_accel_mask']&(d['accel_candidate']==k));ix=ix[(ix>=35)&(ix<len(d['time'])-35)]
  for i in ix:
   if not np.isfinite(d['speed'][i-5:i+6]).all() or not np.isfinite(d['acceleration_proxy'][i-5:i+6]).all():continue
   out.append({'key':key,'i':int(i),'v':float(d['speed_smoothed'][i]),'a':float(d['acceleration_proxy'][i])})
 return out
def match(cs,z,cc):
 dz=cs[z['key']]['d'];i=z['i'];rows=[]
 for c in cc:
  if abs(c['v']-z['v'])>1 or abs(c['a']-z['a'])>.15:continue
  dc=cs[c['key']]['d'];j=c['i']
  vr=float(np.sqrt(np.mean((dz['speed'][i-5:i+6]-dc['speed'][j-5:j+6])**2)));ar=float(np.sqrt(np.mean((dz['acceleration_proxy'][i-5:i+6]-dc['acceleration_proxy'][j-5:j+6])**2)))
  if vr>1 or ar>.2:continue
  vv=dz['speed'][i-30:i+31]-dc['speed'][j-30:j+31];aa=dz['acceleration_proxy'][i-30:i+31]-dc['acceleration_proxy'][j-30:j+31]
  finite=bool(np.isfinite(vv).all() and np.isfinite(aa).all());lv=float(np.sqrt(np.mean(vv**2))) if finite else None;la=float(np.sqrt(np.mean(aa**2))) if finite else None
  long=finite and lv<=1.5 and la<=.25
  rows.append({'comma':c,'zod':z,'speed_profile_rmse_m_s':vr,'accel_profile_rmse_m_s2':ar,'context6s_complete':finite,'context6s_speed_rmse':lv,'context6s_accel_rmse':la,'context6s_matched':bool(long),'cost':vr+ar/.2})
 return rows
def select():
 assert not (O/'freeze.json').exists();cs=load()
 write(O/'freeze.json',{'selection_before_features':True,'scope':'All data development exposed. One pair per ZOD video/class, at most6perclass A/D/C. No STOP source coverage. No new labels,feature/model changes or fitting.',
 'central_window':'11 samples spanning1s, existing strict center mask; same class; centers >=3.5s from ends; |speed difference|<=1m/s,|acceleration difference|<=.15m/s²; 1s raw-speed RMSE<=1m/s and slope-profile RMSE<=.2m/s².',
 'context':'±3s sensor comparison; complete values and speedRMSE<=1.5m/s and accelRMSE<=.25 qualify as context-matched. Prioritize context-matched then lowest normalized central mismatch; no prediction/feature values used for selection. Features of neighbor windows still overlap.',
 'selection':'Rare classD first thenA thenC; within eachclass greedy ascending(context mismatch,cost,ZODid,index,commaid,index), one per ZODclip/class; prevent central windows overlapping within either dataset (>10indices). C speed range limited to selected A/D union minmax±1m/s. Max6 pairs/class is cap, not quota.',
 'within_comma_control':'For each selected primary comma window select closest sensor-matched comma from another route using same gates and context priority. Missing controls stay missing; no looser threshold.',
 'analysis':'Use frozen comma2395 scaler and classifier. Standardized864 RMS cross vs within-comma descriptive distance; six blocks and12ROIs; exact linear A-D and true-vs-CONSTANT margin contributions. Not a learned domain classifier or F1 improvement experiment. Matching v/a does not hold depth,FOV,turns,objects,illumination,road,vehicle or label noise fixed.',
 'sensor_truth':'Same existing speed slope proxy. Sign check is recomputation from original aligned speed, not independent GT. ZOD generic-valid semantics and hardware latency remain unverified. Independent CAN crosscheck only available in2mini clips.',
 'protected':{**read(Z/'freeze.json')['protected'],**read(Z/'freeze.json')['label_files'],**read(Z/'comparison_inputs.json'),str(Path(__file__).relative_to(R)):sha(Path(__file__))}})
 selected=[];coverage=[];used={};ad_speeds=[]
 for k in [1,0,2]:
  cc=candidates(cs,'comma',k);zz=candidates(cs,'zod',k)
  if k==2 and ad_speeds:zz=[z for z in zz if min(ad_speeds)-1<=z['v']<=max(ad_speeds)+1]
  edges=[]
  for z in zz:edges.extend(match(cs,z,cc))
  edges.sort(key=lambda e:(not e['context6s_matched'],e['cost'],e['zod']['key'],e['zod']['i'],e['comma']['key'],e['comma']['i']))
  took=set();n=0
  for e in edges:
   zk=e['zod']['key']
   if zk in took:continue
   if any(any(abs(x['i']-old)<=10 for old in used.get(x['key'],[])) for x in [e['comma'],e['zod']]):continue
   e.update(pair_id=f'{ACC[k][0]}{n+1:02}',truth=k);selected.append(e);took.add(zk);n+=1
   for x in [e['comma'],e['zod']]:used.setdefault(x['key'],[]).append(x['i'])
   if k!=2:ad_speeds.append(e['zod']['v'])
   if n==6:break
  coverage.append({'class':k,'comma_centers':len(cc),'zod_centers':len(zz),'zod_videos':len({z['key'] for z in zz}),'eligible_edges':len(edges),'context_matched_edges':sum(e['context6s_matched'] for e in edges),'selected':n})
 for e in selected:
  c=e['comma'];route=cs[c['key']]['row']['route'];opts=[x for x in candidates(cs,'comma',e['truth']) if cs[x['key']]['row']['route']!=route]
  refs=match(cs,c,opts);refs.sort(key=lambda q:(not q['context6s_matched'],q['cost'],q['comma']['key'],q['comma']['i']))
  e['within_comma']=refs[0] if refs else None
 write(O/'matched_manifest.json',selected);write(O/'coverage.json',coverage)
 print(json.dumps(coverage,indent=2));print('selected',len(selected),flush=True)
def feature(c):return np.load(F/(c['row']['id']+'.npz') if c['dataset']=='comma' else Z/'features'/(c['row']['id']+'.npz'))['base']
def analyze():
 assert not (O/'results.json').exists();cs=load();pairs=read(O/'matched_manifest.json');model=joblib.load(Z/'models/comma_only.joblib');sc=model[0];lr=model[-1]
 models={'comma':model,'original_mix':joblib.load(Z/'models/comma_plus_zod.joblib'),'mass_capped':joblib.load(R/'artifacts/stage3_zod_controls_20260920/mass_capped/zod.joblib')}
 protected={str(Z/'models/comma_only.joblib'):sha(Z/'models/comma_only.joblib')};rows=[];detailed=[];sensor_checks=[]
 for p in pairs:
  xs={};zs={};r={k:v for k,v in p.items() if k!='within_comma'};r['predictions']={}
  for name in ['comma','zod']:
   q=p[name];c=cs[q['key']];i=q['i'];x=feature(c);xs[name]=x[i-5:i+6];zs[name]=sc.transform(xs[name]).mean(0)
   d=c['d'];xx=np.arange(-5,6)*.1;a=float(d['speed'][i-5:i+6]@xx/(xx@xx));assert abs(a-q['a'])<1e-8
   raw_delta=float(d['speed'][i+5]-d['speed'][i-5]);sensor_checks.append({'pair':p['pair_id'],'dataset':name,'id':c['row']['id'],'sample_index':i,'proxy_slope':a,'raw_endpoint_speed_change_m_s':raw_delta,'sign_consistent':bool(raw_delta*a>0) if p['truth']!=2 else None,'max_frame_time_error_s':float(abs(d['frame_time'][i-5:i+6]-d['time'][i-5:i+6]).max())})
   path=F/(c['row']['id']+'.npz') if name=='comma' else Z/'features'/(c['row']['id']+'.npz');protected[str(path.relative_to(R))]=sha(path)
   rr={}
   for mn,m in models.items():
    prob=m.predict_proba(xs[name]);logit=m.decision_function(xs[name]);rr[mn]={'center_prediction':int(prob[5].argmax()),'center_probability':prob[5].tolist(),'window_prediction_counts':np.bincount(prob.argmax(1),minlength=4).tolist(),'mean_logit':logit.mean(0).tolist(),'A_minus_D_margin':float((logit[:,0]-logit[:,1]).mean()),'true_minus_constant_margin':float((logit[:,p['truth']]-logit[:,2]).mean())}
   r['predictions'][name]=rr
  delta=zs['zod']-zs['comma'];r['standardized_rms']=float(np.sqrt(np.mean(delta**2)));r['block_rms']=np.sqrt(np.mean(delta.reshape(6,144)**2,axis=1)).tolist();r['within_comma_rms']=None;r['cross_within_ratio']=None
  if p['within_comma']:
   q=p['within_comma']['comma'];c=cs[q['key']];x=feature(c);path=F/(c['row']['id']+'.npz');protected[str(path.relative_to(R))]=sha(path);zi=sc.transform(x[q['i']-5:q['i']+6]).mean(0);wd=zi-zs['comma'];r['within_comma_rms']=float(np.sqrt(np.mean(wd**2)));r['within_comma_block_rms']=np.sqrt(np.mean(wd.reshape(6,144)**2,axis=1)).tolist();r['cross_within_ratio']=r['standardized_rms']/max(r['within_comma_rms'],1e-12)
  for label,a,b in [('A_minus_D',0,1),('true_minus_C',p['truth'],2)]:
   contrib=delta*(lr.coef_[a]-lr.coef_[b]);expected=(r['predictions']['zod']['comma']['mean_logit'][a]-r['predictions']['zod']['comma']['mean_logit'][b])-(r['predictions']['comma']['comma']['mean_logit'][a]-r['predictions']['comma']['comma']['mean_logit'][b])
   assert np.isclose(contrib.sum(),expected,atol=1e-6),p['pair_id']
   z=contrib.reshape(6,12,4,3);r[label+'_contribution']={'total':float(contrib.sum()),'by_block':z.sum((1,2,3)).tolist(),'by_roi':z.sum((0,2,3)).tolist(),'by_channel':z.sum((0,1,3)).tolist()}
  for j in range(864):
   block,rem=divmod(j,144);roi,rem=divmod(rem,12);chan,stat=divmod(rem,3)
   detailed.append({'pair':p['pair_id'],'index':j,'block':BLOCK[block],'roi':roi,'channel':CHANNEL[chan],'stat':STAT[stat],'comma_raw':float(xs['comma'].mean(0)[j]),'zod_raw':float(xs['zod'].mean(0)[j]),'standardized_delta':float(delta[j]),'A_minus_D_contribution':float(delta[j]*(lr.coef_[0,j]-lr.coef_[1,j]))})
  rows.append(r)
 write(O/'results.json',rows);write(O/'sensor_checks.json',sensor_checks);pd.DataFrame(detailed).to_csv(O/'feature_details.csv',index=False)
 write(O/'feature_inputs.json',protected)
 f=read(O/'freeze.json');assert all(sha(R/p)==h for p,h in f['protected'].items());assert all(sha(R/p)==h for p,h in protected.items())
 print(pd.DataFrame([{'pair':r['pair_id'],'class':r['truth'],'comma':r['comma']['key'],'zod':r['zod']['key'],'context':r['context6s_matched'],'distance':r['standardized_rms'],'within':r['within_comma_rms'],'ratio':r['cross_within_ratio'],'comma_margin':r['predictions']['comma']['comma']['A_minus_D_margin'],'zod_margin':r['predictions']['zod']['comma']['A_minus_D_margin']} for r in rows]).to_string(index=False))
if __name__=='__main__':
 with threadpool_limits(limits=2):
  if sys.argv[1]=='select':select()
  else:analyze()
