"""Semantic correction only: remove RAV4 bonnet top; preserve all raw attempts."""
from run import *
import copy
(O/'corrected').mkdir();(O/'masks_corrected').mkdir()
for p in (O/'masks').glob('*.npz'):
 m=np.load(p)['mask'].copy()
 if p.stem=='expanded_09_335':m[:,int(.69*m.shape[1]):,:]=0
 np.savez_compressed(O/'masks_corrected'/p.name,mask=m)
write(O/'corrected/freeze.json',{'reason':'AI crop inspection found bonnet-top pixels around y270-280/378 in expanded09 retained by old y<.74 mask. Exclude y>=.69H for all21frames. No q/sensor-based boundary choice. Old masks/results retained.','change':'Filter existing sampled source AND destination points only, with same identities and positions. No resampling or threshold change. Recompute PnP and sufficiency for expanded09 only; reuse all7 other windows.','scope':'Both older static baseline and all current candidates. Follows previously reviewed all21 RGB frames; close crops of0/10/19 confirm bonnet position.','source_hash':sha(Path(__file__))})
key='expanded_09_335';inp=np.load(B/'depth'/(key+'_input.npz'));t=inp['time'];dep=np.load(B/'depth'/(key+'_first.npz'))['depth'];k=np.load(B/'flow'/(key+'.npz'))['K'];mask=np.load(O/'masks_corrected'/(key+'.npz'))['mask'];h,w=mask.shape[1:];cv2.setNumThreads(2)

def revise(row,pts,prefix):
 stats=[];pairs=[];cache={}
 for j in range(20):
  p,q=pts[f'{prefix}p{j}'],pts[f'{prefix}q{j}'];keep=inside(mask[j],p)&inside(mask[j+1],q);p,q=p[keep],q[keep];cache[f'{prefix}p{j}']=p;cache[f'{prefix}q{j}']=q
  span=np.ptp(p,axis=0)/[w,h] if len(p) else np.zeros(2);stats.append({'pair':j,'points':len(p),'span':span.tolist(),'sufficient':bool(len(p)>=30 and span[0]>=.4 and span[1]>=.2)})
  pairs.append(g.pair_pnp(p,q,dep[j],dep[j+1],k,float(t[j+1]-t[j])) if len(p) else {'valid':False,'reason':'no_points','points':0,'speed':None})
 s=np.array([s['sufficient'] for s in stats]);f=g.q_feature(pairs,t);return {'key':key,'stats':stats,'pnp':pairs,'median_points':float(np.median([s['points'] for s in stats])),'point_sufficient_pairs':int(s.sum()),'point_sufficient_window':bool(s.sum()>=16 and s[:4].sum()>=3 and s[-4:].sum()>=3),'pnp_valid_pairs':f['valid_pairs'],'pnp_quality':f['quality'],'q_diagnostic':f['q'],'note':'Semantic bonnet correction; unchanged point identities'},cache
for stage in ['grid4_oldmask','grid4_expanded','adaptive_expanded']:
 rows=read(O/stage/'results.json');ix=next(i for i,r in enumerate(rows) if r['key']==key);rows[ix],cache=revise(rows[ix],np.load(O/stage/(key+'.npz')),'');np.savez_compressed(O/'corrected'/(stage+'_'+key+'.npz'),**cache);write(O/'corrected'/(stage+'.json'),rows);print(stage,rows[ix]['median_points'],rows[ix]['point_sufficient_pairs'],rows[ix]['pnp_valid_pairs'])
rows=read(O/'source_pair/results.json');ix=next(i for i,r in enumerate(rows) if r['key']==key);cache={}
for name in ['DIS','LK']:
 r,c=revise(rows[ix]['methods'][name],np.load(O/'source_pair'/(key+'.npz')),name+'_');rows[ix]['methods'][name]=r;cache.update(c);print(name,r['median_points'],r['point_sufficient_pairs'],r['pnp_valid_pairs'])
np.savez_compressed(O/'corrected'/('source_pair_'+key+'.npz'),**cache);write(O/'corrected/source_pair.json',rows)
# Same semantic correction to original static subset baseline, without replacing it.
flow=np.load(B/'flow'/(key+'.npz'));sel=np.load(S/'selected'/(key+'.npz'));pts={}
for j in range(20):
 ix=sel[f'indices{j}'];pts[f'p{j}']=flow[f'p{j}'][ix];pts[f'q{j}']=flow[f'q{j}'][ix]
r,c=revise({},pts,'');write(O/'corrected/baseline_expanded09.json',r)
write(O/'corrected/checks.json',{'exit_status':0,'only_expanded09_changed':True,'others_reused':7,'old_results_preserved':True,'cutoff_from_RGB_not_sensor':True})
