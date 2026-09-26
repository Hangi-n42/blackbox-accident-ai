"""Same source-selected points, DIS vs LK: isolate rejection/tracker effects."""
from adaptive import *

def patch_error(a,b,p,q):
 offsets=np.array([[x,y] for y in [-6,-3,0,3,6] for x in [-6,-3,0,3,6]],np.float32)
 return np.mean(np.stack([abs(g.sample(a.astype('float32'),p+o)-g.sample(b.astype('float32'),q+o)) for o in offsets]),axis=0)

def main():
 cv2.setNumThreads(2);out=O/'source_pair';out.mkdir();rows=[]
 protected={str(p.relative_to(R)):sha(p) for p in list((O/'masks').glob('*.npz'))+[Path(__file__),O/'adaptive.py',O/'run.py',B/'geometry.py']}
 write(out/'freeze.json',{'comparison':'Identical source-only selected points; DIS vs pyramidLK. Not comparable as tracker-only change against adaptive_expanded because selection timing differs. Both methods always included.','source_selection':'2px candidategrid within expanded static masks, existing7px std>=4 and depthconfidence>=original8px median. Corner-prior NMS radius4 BEFORE destination/flow filtering. No threshold search.','common_quality':'Both endpoint static masks, texture std>=4, depth confidence>=original8px medians; FB<=1px, additional25sample patch mean abs brightness error<=20. Same K,depth,PnP.','tracker':'DIS_FAST cached vs OpenCV LK21x21,maxLevel3,30iter,.01eps with both LK status checks. No initial flow or sensor input.','sufficiency':'same>=30 points,xspan>=.4,yspan>=.2 and >=16/20 pairs,>=3 early/late. Independent4px support not proven.','protected':protected})
 for old in read(B/'results.json'):
  key=old['key'];inp=np.load(B/'depth'/(key+'_input.npz'));rgb=inp['rgb'];t=inp['time'];base=np.load(B/'depth'/(key+'_first.npz'));dep=base['depth'];conf=base['depth_conf'];k=np.load(B/'flow'/(key+'.npz'))['K'];m=np.load(O/'masks'/(key+'.npz'))['mask'];h,w=m.shape[1:];flows=np.load(O/'flow'/(key+'.npz'));grid=g.roi_grid(h,w,2);grid8=g.roi_grid(h,w);gray=[cv2.cvtColor(im,cv2.COLOR_RGB2GRAY) for im in rgb];std=[np.sqrt(np.maximum(cv2.blur(a.astype('float32')**2,(7,7))-cv2.blur(a.astype('float32'),(7,7))**2,0)) for a in gray];corner=[cv2.cornerMinEigenVal(a,7) for a in gray];ct=[np.median(g.sample(c,grid8)) for c in conf];methods={name:{'stats':[],'pnp':[]} for name in ['DIS','LK']};cache={}
  for j in range(20):
   source=inside(m[j],grid)&(g.sample(std[j],grid)>=4)&(g.sample(conf[j],grid)>=ct[j]);p=grid[source];score=g.sample(corner[j],p) if len(p) else np.array([]);ix=thin(p,score,h,w);p=p[ix];cache[f'source_p{j}']=p;matches={}
   if len(p):
    q=p+g.sample(flows[f'f{j}'],p);matches['DIS']=(q,np.linalg.norm(g.sample(flows[f'b{j}'],q)+q-p,axis=1),np.ones(len(p),bool))
    q,s,_=cv2.calcOpticalFlowPyrLK(gray[j],gray[j+1],p[:,None,:],None,winSize=(21,21),maxLevel=3,criteria=(3,30,.01));back,bs,_=cv2.calcOpticalFlowPyrLK(gray[j+1],gray[j],q,None,winSize=(21,21),maxLevel=3,criteria=(3,30,.01));matches['LK']=(q[:,0],np.linalg.norm(back[:,0]-p,axis=1),(s[:,0]>0)&(bs[:,0]>0))
   else:matches={name:(np.empty((0,2),np.float32),np.array([]),np.array([],bool)) for name in methods}
   for name,(q,fb,status) in matches.items():
    good=np.zeros(len(p),bool);pe=np.array([])
    if len(p):
     pe=patch_error(gray[j],gray[j+1],p,q);good=status&np.isfinite(q).all(1)&(q[:,0]>=7)&(q[:,0]<w-7)&(q[:,1]>=7)&(q[:,1]<h-7)&(fb<=1)&(pe<=20)&inside(m[j+1],q)&(g.sample(std[j+1],q)>=4)&(g.sample(conf[j+1],q)>=ct[j+1])
    pp,qq=p[good],q[good];span=np.ptp(pp,axis=0)/[w,h] if len(pp) else np.zeros(2);suf=bool(len(pp)>=30 and span[0]>=.4 and span[1]>=.2);cache[f'{name}_p{j}']=pp;cache[f'{name}_q{j}']=qq;cache[f'{name}_fb{j}']=fb[good];cache[f'{name}_photometric{j}']=pe[good] if len(p) else pe
    methods[name]['stats'].append({'pair':j,'source_points':len(p),'points':len(pp),'span':span.tolist(),'sufficient':suf,'fb_p90':float(np.quantile(fb[good],.9)) if len(pp) else None,'photo_p90':float(np.quantile(pe[good],.9)) if len(pp) else None})
    methods[name]['pnp'].append(g.pair_pnp(pp,qq,dep[j],dep[j+1],k,float(t[j+1]-t[j])) if len(pp) else {'valid':False,'reason':'no_points','points':0,'speed':None})
  for name,v in methods.items():
   suf=np.array([s['sufficient'] for s in v['stats']]);feat=g.q_feature(v['pnp'],t);v.update(point_sufficient_pairs=int(suf.sum()),point_sufficient_window=bool(suf.sum()>=16 and suf[:4].sum()>=3 and suf[-4:].sum()>=3),pnp_valid_pairs=feat['valid_pairs'],pnp_quality=feat['quality'],median_points=float(np.median([s['points'] for s in v['stats']])),q_diagnostic=feat['q'])
  np.savez_compressed(out/(key+'.npz'),**cache);rows.append({'key':key,'methods':methods});write(out/'results.json',rows);print(key,[(n,v['median_points'],v['point_sufficient_pairs'],v['pnp_valid_pairs']) for n,v in methods.items()],flush=True)
 assert all(sha(R/p)==h for p,h in protected.items());write(out/'checks.json',{'exit_status':0,'frozen_inputs_unchanged':True,'same_source_points':True,'no_quality_relaxation':True,'new_depth_calls':0,'classifier_fits':0})
if __name__=='__main__':main()
