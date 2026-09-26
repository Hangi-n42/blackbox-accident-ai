"""Post-result attribution control; no feature promotion or threshold changes."""
from prepare import *
import geometry,cv2
assert not (O/'fixed_inlier_freeze.json').exists()
write(O/'fixed_inlier_freeze.json',{'scope':'Post-hoc descriptive control. All8windows; no result-based subset or retuning. No new depth inference.',
 'held_fixed':'Same DIS correspondences, clipK, first-variant RANSAC inlier indices, first-variant valid-pair mask. Same iterative solver and first-variant pose initialization for all3depth variants.',
 'changed':'Only source-frame depth values from first/reverse/middle maps. No forward/backward quality re-selection. New depths are not rescaled.',
 'interpretation':'This removes RANSAC membership/initialization variation, but does not certify true static points, calibration or physical translation. Othervariants are not declared valid using first mask.',
 'script_sha256':sha(Path(__file__))})
out=[]
for row in read(O/'results.json'):
 key=row['key'];flow=np.load(O/'flow'/(key+'.npz'));t=flow['time'];k=flow['K'];first=row['features']['first']['pairs'];good=np.array([p['valid'] for p in first]);speeds={}
 for variant in ['first','reverse','middle']:
  depth=np.load(O/'depth'/(key+'_'+variant+'.npz'))['depth'];vv=np.full(20,np.nan)
  for j,pair in enumerate(first):
   if not good[j]:continue
   ii=np.array(pair['inlier_indices']);p=flow[f'p{j}'][ii];q=flow[f'q{j}'][ii];x=geometry.backproject(p,geometry.sample(depth[j],p),k).astype('float64')
   ok,r,tr=cv2.solvePnP(x,q.astype('float64'),k,None,np.array(pair['rvec'],dtype='float64').reshape(3,1),np.array(pair['translation'],dtype='float64').reshape(3,1),True,flags=cv2.SOLVEPNP_ITERATIVE)
   assert ok;vv[j]=np.linalg.norm(tr)/(t[j+1]-t[j])
  speeds[variant]=vv
 tm=(t[:-1]+t[1:])/2;qs={v:float(np.polyfit(tm[good]-t[10],np.log(s[good]),1)[0]) if good.sum()>=3 else None for v,s in speeds.items()}
 orig=[p['q'] for p in row['features'].values()];vals=[q for q in qs.values() if q is not None]
 out.append({'key':key,'sensor_q':row['sensor_q'],'first_valid_pairs':int(good.sum()),'q_fixed_inliers':qs,'q_fixed_span':float(max(vals)-min(vals)) if vals else None,'original_q_span':float(max(orig)-min(orig)) if all(x is not None for x in orig) else None,'first_q_reoptimization_delta':qs['first']-orig[0] if qs['first'] is not None else None,'scope':'descriptive_only; no othervariant quality claim'})
write(O/'fixed_inlier_results.json',out)
print(json.dumps(out,indent=2))
