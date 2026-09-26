from pathlib import Path
import sys,json,hashlib,time
import numpy as np
import cv2
O=Path(__file__).resolve().parent;R=O.parents[1];B=R/'artifacts/stage3_depth_pnp_20260920';S=R/'artifacts/stage3_static_background_20260920'
sys.path.insert(0,str(B));import geometry as g
sys.path.insert(0,str(S));from masks import inside
read=lambda p:json.loads(p.read_text());write=lambda p,v:p.write_text(json.dumps(v,ensure_ascii=False,indent=2));sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()

def process(stage):
 cv2.setNumThreads(2);out=O/stage;out.mkdir();(O/'flow').mkdir(exist_ok=True);rows=[]
 maskroot=S/'masks' if stage=='grid4_oldmask' else O/'masks'
 protected={str(p.relative_to(R)):sha(p) for p in list(maskroot.glob('*.npz'))+[Path(__file__),B/'geometry.py']}
 write(out/'freeze.json',{'stage':stage,'change':'8px to4px grid only' if stage=='grid4_oldmask' else 'expanded RGB-reviewed masks only vs grid4_oldmask','fixed':'DIS_FAST; 7px texture std>=4; FB<=1px; confidence medians from ORIGINAL8px grid; source and destination masks; same depth,K,dt,PnP and quality gates. No score or sensor used in selection.','point_sufficiency':'At least30 correspondences and xspan>=.4width,yspan>=.2height; window >=16/20 and >=3 first/last4. Also independently report original PnP inlier/quality pass. Counts are not independent 3D points or exact correspondence truth.','protected':protected})
 for old in read(B/'results.json'):
  key=old['key'];inp=np.load(B/'depth'/(key+'_input.npz'));rgb=inp['rgb'];t=inp['time'];base=np.load(B/'depth'/(key+'_first.npz'));dep=base['depth'];conf=base['depth_conf'];k=np.load(B/'flow'/(key+'.npz'))['K'];m=np.load(maskroot/(key+'.npz'))['mask'];h,w=m.shape[1:];grid=g.roi_grid(h,w,4);grid8=g.roi_grid(h,w);grays=[cv2.cvtColor(im,cv2.COLOR_RGB2GRAY) for im in rgb];std=[np.sqrt(np.maximum(cv2.blur(a.astype('float32')**2,(7,7))-cv2.blur(a.astype('float32'),(7,7))**2,0)) for a in grays];threshold=[np.median(g.sample(c,grid8)) for c in conf];cache={};stats=[];pnp=[]
  flowpath=O/'flow'/(key+'.npz')
  if flowpath.exists():flows=dict(np.load(flowpath))
  else:
   dis=cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_FAST);flows={}
   for j in range(20):flows[f'f{j}']=dis.calc(grays[j],grays[j+1],None);flows[f'b{j}']=dis.calc(grays[j+1],grays[j],None)
   np.savez_compressed(flowpath,**flows)
  for j in range(20):
   p=grid.copy();q=p+g.sample(flows[f'f{j}'],p);fb=np.linalg.norm(g.sample(flows[f'b{j}'],q)+q-p,axis=1)
   good=np.isfinite(q).all(1)&(q[:,0]>=2)&(q[:,0]<w-2)&(q[:,1]>=2)&(q[:,1]<h-2)&(fb<=1)&(g.sample(std[j],p)>=4)&(g.sample(std[j+1],q)>=4)&(g.sample(conf[j],p)>=threshold[j])&(g.sample(conf[j+1],q)>=threshold[j+1])&inside(m[j],p)&inside(m[j+1],q)
   p=p[good];q=q[good];cache[f'p{j}']=p;cache[f'q{j}']=q;cache[f'fb{j}']=fb[good]
   span=np.ptp(p,axis=0)/[w,h] if len(p) else np.zeros(2);ok=bool(len(p)>=30 and span[0]>=.4 and span[1]>=.2)
   # Additional support diagnostic: occupied 32px cells, not asserted independent evidence.
   cells=len(np.unique((p//32).astype(int),axis=0));stats.append({'pair':j,'points':len(p),'span':span.tolist(),'cells32':cells,'fb_p90':float(np.quantile(fb[good],.9)) if len(p) else None,'sufficient':ok})
   pair=g.pair_pnp(p,q,dep[j],dep[j+1],k,float(t[j+1]-t[j])) if len(p) else {'valid':False,'reason':'no_points','points':0,'speed':None};pnp.append(pair)
  np.savez_compressed(out/(key+'.npz'),**cache);sufficient=np.array([s['sufficient'] for s in stats]);feature=g.q_feature(pnp,t)
  row={'key':key,'stats':stats,'point_sufficient_pairs':int(sufficient.sum()),'point_sufficient_window':bool(sufficient.sum()>=16 and sufficient[:4].sum()>=3 and sufficient[-4:].sum()>=3),'pnp':pnp,'pnp_valid_pairs':feature['valid_pairs'],'pnp_quality':feature['quality'],'q_diagnostic':feature['q'],'median_points':float(np.median([s['points'] for s in stats]))};rows.append(row);write(out/'results.json',rows)
  print(stage,key,row['median_points'],row['point_sufficient_pairs'],row['pnp_valid_pairs'],flush=True)
 assert all(sha(R/p)==v for p,v in protected.items());write(out/'checks.json',{'exit_status':0,'frozen_inputs_unchanged':True,'thresholds_not_relaxed':True,'new_depth_calls':0,'classifier_fits':0})
if __name__=='__main__':process(sys.argv[1])
