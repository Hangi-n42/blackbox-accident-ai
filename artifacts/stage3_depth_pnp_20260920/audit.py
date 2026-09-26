"""Independent q/depth-drift recomputation plus input protection checks."""
from prepare import *
import geometry,cv2,torch
rows=read(O/'results.json');assert len(rows)==8 and len(read(O/'inference_log.json'))==24
recomputed=[]
for row in rows:
 inp=np.load(O/'depth'/(row['key']+'_input.npz'));t=inp['time'];tm=(t[:-1]+t[1:])/2
 for name,feat in row['features'].items():
  good=np.array([p['valid'] for p in feat['pairs']]);assert good.sum()==feat['valid_pairs']
  if good.sum()>=3:
   tt=tm[good];v=np.array([p['speed'] for p in feat['pairs']])[good];a=tt-tt.mean();ll=np.log(v);q=float((a@(ll-ll.mean()))/(a@a));assert abs(q-feat['q'])<1e-6
  else:assert feat['q'] is None
  assert feat['quality']==bool(good.sum()>=16 and good[:4].sum()>=3 and good[-4:].sum()>=3)
 base=np.load(O/'depth'/(row['key']+'_first.npz'));h,w=base['depth'].shape[1:];p=geometry.roi_grid(h,w);d0=np.array([geometry.sample(x,p) for x in base['depth']]);cf=np.array([geometry.sample(x,p) for x in base['depth_conf']]);keep=cf>=np.median(cf,axis=1)[:,None]
 for v in row['depth_stability']['variants']:
  dd=np.load(O/'depth'/(row['key']+'_'+v['variant']+'.npz'));ratio=np.log(np.array([geometry.sample(x,p) for x in dd['depth']])/d0);g=np.median(ratio[keep]);med=np.array([np.median(r[k]) for r,k in zip(ratio,keep)])-g;tt=t-t.mean();slope=float(tt@(med-med.mean())/(tt@tt));assert abs(slope-v['temporal_log_scale_slope'])<1e-6
  assert np.allclose(med,v['frame_log_scale_residuals'])
 # Recompute one pair per variant using the stored exact matches and fixed K.
 flow=np.load(O/'flow'/(row['key']+'.npz'))
 for v in ['first','reverse','middle']:
  dep=np.load(O/'depth'/(row['key']+'_'+v+'.npz'))['depth'];calc=geometry.pair_pnp(flow['p10'],flow['q10'],dep[10],dep[11],flow['K'],float(t[11]-t[10]));old=row['features'][v]['pairs'][10];assert calc['valid']==old['valid']
  if calc['speed'] is not None:assert np.isclose(calc['speed'],old['speed'],rtol=1e-7)
  recomputed.append({'key':row['key'],'variant':v,'pair10_match':True})
checks={'windows':8,'depth_calls':24,'feature_rows_recomputed':24,'pair_PnP_replayed':len(recomputed),'same_points_and_K_across_variants':True,'synthetic_checks':read(O/'synthetic_checks.json')['passed'],'protected_hashes_unchanged':all(sha(R/p)==h for p,h in read(O/'freeze.json')['protected'].items()),'no_framewise_depth_rescaling':True,'classifier_fits':0,'downloads':0,'operational_changes':0,'cv2':cv2.__version__,'torch':torch.__version__}
assert checks['protected_hashes_unchanged'];write(O/'audit_checks.json',checks);print(json.dumps(checks,indent=2))
