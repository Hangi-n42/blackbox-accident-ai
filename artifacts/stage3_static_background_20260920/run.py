"""One-factor static-background subset control; no inference or training."""
from masks import *
import sys
from collections import Counter
sys.path.insert(0,str(B))
import geometry as g
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
read=lambda p:json.loads(p.read_text())
write=lambda p,v:p.write_text(json.dumps(v,ensure_ascii=False,indent=2))

def main():
 cv2.setNumThreads(2)
 assert not (O/'freeze.json').exists(), 'Preserve completed experiment'
 protected=read(B/'freeze.json')['protected'].copy()
 files=[B/'results.json',B/'geometry.py',B/'freeze.json',B/'manifest.json',O/'masks.py',O/'run.py']+list((B/'depth').glob('*.npz'))+list((B/'flow').glob('*.npz'))+list((O/'masks').glob('*.npz'))
 protected.update({str(p.relative_to(R)):sha(p) for p in files})
 write(O/'freeze.json',{'factor':'Only filter existing DIS correspondences by RGB-reviewed static-surface masks at BOTH endpoints. Same depth, K, dt, RANSAC, thresholds, q formula and labels. No replacement or densification of points.','review':'AI review of all168 RGB frames and all168 mask overlays. Not human/official/fully verified static-point truth. Reject vehicles even if parked, hood, sky, foliage and uncertain regions; retain conservative road/rigid surface interiors.','mask_design':'One RGB-reviewed polygon intersection across each clip, 3px erosion. ZOD2 right guardrail only frame>=8 after truck clears. ZOD26 frames5/11 fully excluded for wiper; frontcar and close rightcar excluded. expanded19 suspected building strip removed on overlay review because it covered hedge; before any new PnP results.','mask_limits':'Conservative polygons do not exhaust every static point. Road shadows/reflections and correspondence correctness remain uncertain. Mostly road-plane points can have insufficient geometric diversity. Frame21 endpoint has no outgoing pair.','outcome_rule':'Reuse original gates including >=30 inliers, spatial spread and >=16/20 valid pairs; no relaxation. Missing q is not zero or improvement. Compare error only on valid shared support; if no valid windows, accuracy effect is not estimable.','evaluation_exposure':'All8 development exposed; RGB reviewer knows prior experiment history. No claim of blinded annotation or independent validation. New outcomes not used for masks.','protected':protected})
 (O/'selected').mkdir(exist_ok=True);results=[];replays=0
 for old in read(B/'results.json'):
  key=old['key'];inp=np.load(B/'depth'/(key+'_input.npz'));t=inp['time'];flow=np.load(B/'flow'/(key+'.npz'));m=np.load(O/'masks'/(key+'.npz'))['mask'];matches=[];cache={};stats=[]
  for j in range(20):
   p,q=flow[f'p{j}'],flow[f'q{j}'];ix=np.flatnonzero(inside(m[j],p)&inside(m[j+1],q));pp,qq=p[ix],q[ix];matches.append((pp,qq));cache[f'indices{j}']=ix
   assert np.array_equal(pp,p[ix]) and np.array_equal(qq,q[ix])
   stats.append({'pair':j,'original_points':len(p),'retained_points':len(ix),'fraction':len(ix)/len(p) if len(p) else None})
  np.savez_compressed(O/'selected'/(key+'.npz'),**cache)
  variants={}
  for name in ['first','reverse','middle']:
   dep=np.load(B/'depth'/(key+'_'+name+'.npz'))['depth'];basecheck=g.pair_pnp(flow['p10'],flow['q10'],dep[10],dep[11],flow['K'],float(t[11]-t[10]));ref=old['features'][name]['pairs'][10]
   assert basecheck['valid']==ref['valid'] and np.isclose(basecheck['speed'],ref['speed']);replays+=1
   pairs=[g.pair_pnp(p,q,dep[j],dep[j+1],flow['K'],float(t[j+1]-t[j])) if len(p) else {'valid':False,'reason':'no_points','points':0,'speed':None} for j,(p,q) in enumerate(matches)]
   # Same minimum point/quality rules. No permissive fit on inadequate pairs.
   variants[name]={**g.q_feature(pairs,t),'pairs':pairs}
  qs=[x['q'] for x in variants.values()];stable=all(x is not None for x in qs) and max(qs)-min(qs)<=.02
  valid=bool(old['depth_stability']['pass'] and stable and all(x['quality'] for x in variants.values()))
  counts=[s['retained_points'] for s in stats];row={'key':key,'dataset':old['dataset'],'label':old['label'],'sensor_q':old['sensor_q'],'original_first_q':old['features']['first']['q'],'original_first_valid_pairs':old['features']['first']['valid_pairs'],'depth_stability_unchanged':old['depth_stability']['pass'],'selection':stats,'points_min_median_max':[min(counts),float(np.median(counts)),max(counts)],'features':variants,'q_stable':stable,'valid':valid};results.append(row)
  print(key,row['points_min_median_max'],[(k,v['valid_pairs'],v['q']) for k,v in variants.items()],flush=True)
 write(O/'results.json',results)
 summary={'windows':8,'pairs_per_variant':160,'depth_variants':3,'retained_point_observations':sum(s['retained_points'] for r in results for s in r['selection']),'original_point_observations':sum(s['original_points'] for r in results for s in r['selection']),'pairs_below_30_points':sum(s['retained_points']<30 for r in results for s in r['selection']),'valid_pairs_first':sum(r['features']['first']['valid_pairs'] for r in results),'valid_windows':sum(r['valid'] for r in results),'q_available_first':sum(r['features']['first']['q'] is not None for r in results),'first_pair_reasons':dict(Counter(p['reason'] for r in results for p in r['features']['first']['pairs'])),'score_measured':False,'classifier_fits':0,'new_depth_calls':0,'downloads':0,'thresholds_changed':False}
 write(O/'summary.json',summary)
 assert all(sha(R/p)==h for p,h in protected.items())
 checks={'exit_status':0,'baseline_pair_replays':replays,'same_functions_and_parameters':True,'subset_endpoints_verified':True,'protected_hashes_unchanged':True,'no_invalid_q_filled':True,'all168_RGB_and_mask_frames_AI_reviewed':True}
 write(O/'checks.json',checks);print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
