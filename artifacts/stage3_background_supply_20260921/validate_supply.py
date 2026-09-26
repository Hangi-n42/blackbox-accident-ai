"""Final strict supply = tracker consensus plus unchanged depth confidence."""
from run import *
from scipy.spatial.distance import pdist
from collections import Counter
O.joinpath('validated').mkdir();cv2.setNumThreads(2);rows=[];manifest=[]
write(O/'validated/freeze.json',{'input':'depth_separated consensus points AND depth_ready flags. Same DIS q, prior first depth,K,PnP. No quality gate relaxation. No further masking/selection changes from PnP outputs.','goal':'Separate point supply sufficiency from full PnP/window quality; no sensor fitting or score evaluation.'})
for rr in read(O/'depth_separated/results.json'):
 key=rr['key'];pts=np.load(O/'depth_separated'/(key+'.npz'));inp=np.load(B/'depth'/(key+'_input.npz'));t=inp['time'];base=np.load(B/'depth'/(key+'_first.npz'));dep=base['depth'];k=np.load(B/'flow'/(key+'.npz'))['K'];m=np.load(O/'masks_corrected'/(key+'.npz'))['mask'];h,w=m.shape[1:];pairs=[];stats=[];cache={}
 for j in range(20):
  keep=pts[f'depth_ready{j}'];p,q=pts[f'p{j}'][keep],pts[f'q{j}'][keep];lq=pts[f'lk_q{j}'][keep];assert inside(m[j],p).all() and inside(m[j+1],q).all();assert np.all(np.linalg.norm(q-lq,axis=1)<=1.00001)
  if len(p)>1:assert pdist(p).min()>4-1e-6
  cache[f'p{j}']=p;cache[f'q{j}']=q;span=np.ptp(p,axis=0)/[w,h] if len(p) else np.zeros(2);suff=bool(len(p)>=30 and span[0]>=.4 and span[1]>=.2)
  pair=g.pair_pnp(p,q,dep[j],dep[j+1],k,float(t[j+1]-t[j])) if len(p) else {'valid':False,'reason':'no_points','points':0,'speed':None};pairs.append(pair);stats.append({'pair':j,'points':len(p),'span':span.tolist(),'sufficient':suff,'PnP_valid':pair['valid']})
  manifest.append({'key':key,'pair':j,'source_frame':j,'target_frame':j+1,'source_time':float(t[j]),'target_time':float(t[j+1]),'source_manifest':str((B/'manifest.json').relative_to(R)),'points_file':str((O/'validated'/(key+'.npz')).relative_to(R)),'point_arrays':[f'p{j}',f'q{j}'],'points':len(p),'point_supply_sufficient':suff,'PnP_valid':pair['valid'],'reason':pair['reason'],'review':'AI-reviewed masks; tracker-consensus proxy; not human correspondence truth','exposure':'development_diagnostic'})
 np.savez_compressed(O/'validated'/(key+'.npz'),**cache);f=g.q_feature(pairs,t);s=np.array([s['sufficient'] for s in stats]);rows.append({'key':key,'stats':stats,'pnp':pairs,'median_points':float(np.median([s['points'] for s in stats])),'point_sufficient_pairs':int(s.sum()),'point_sufficient_window':bool(s.sum()>=16 and s[:4].sum()>=3 and s[-4:].sum()>=3),'pnp_valid_pairs':f['valid_pairs'],'pnp_quality':f['quality'],'q_diagnostic':f['q']});print(key,rows[-1]['median_points'],int(s.sum()),f['valid_pairs'],f['quality'],flush=True)
write(O/'validated/results.json',rows);write(O/'validated/manifest.json',manifest)
assert all(sha(R/p)==h for p,h in read(S/'freeze.json')['protected'].items())
for stage in ['grid4_oldmask','grid4_expanded','adaptive_expanded','source_pair']:
 assert all(sha(R/p)==h for p,h in read(O/stage/'freeze.json')['protected'].items())
write(O/'validated/checks.json',{'exit_status':0,'all160_pairs_verified':True,'min_source_spacing_gt4px':True,'both_endpoint_masks_verified':True,'tracker_agreement_verified':True,'all_old_inputs_results_and_production_hashes_unchanged':True,'candidate_freeze_hashes_unchanged':True,'score_measured':False,'classifier_fits':0,'new_depth_calls':0,'downloads':0})
