"""Separate 2D matching supply from learned-depth confidence; no PnP adoption."""
from source_pair import patch_error
from adaptive import *

def support(p,h,w):
 span=np.ptp(p,axis=0)/[w,h] if len(p) else np.zeros(2)
 return {'n':len(p),'span':span.tolist(),'sufficient':bool(len(p)>=30 and span[0]>=.4 and span[1]>=.2)}

def main():
 cv2.setNumThreads(2);out=O/'depth_separated';out.mkdir();write(out/'freeze.json',{'purpose':'Distinguish 2D background correspondence availability from depth-confidence acceptance. Low-confidence depth points are NOT promoted to PnP input.','selection':'Same2px grid, static masks with expanded09 bonnet correction, source std>=4, corner-prior4px NMS; only remove depth-confidence from source selection. Both trackers use same p.','two_D_quality':'DIS and LK both status/FB<=1px, endpointmask, destinationstd>=4,25sample photometricMAE<=20; agreement between their destination coordinates<=1px for primary two_D set.','depth_ready':'Intersection two_D set with unchanged original8px-median confidence at source and DIS destination.','limits':'Tracker consensus and brightness tests are proxies, not known true correspondence; no metric/score claim. No new depth/weights.'})
 rows=[]
 for old in read(B/'results.json'):
  key=old['key'];rgb=np.load(B/'depth'/(key+'_input.npz'))['rgb'];base=np.load(B/'depth'/(key+'_first.npz'));conf=base['depth_conf'];m=np.load(O/'masks_corrected'/(key+'.npz'))['mask'];h,w=m.shape[1:];grid=g.roi_grid(h,w,2);grid8=g.roi_grid(h,w);flows=np.load(O/'flow'/(key+'.npz'));gray=[cv2.cvtColor(im,cv2.COLOR_RGB2GRAY) for im in rgb];std=[np.sqrt(np.maximum(cv2.blur(a.astype('float32')**2,(7,7))-cv2.blur(a.astype('float32'),(7,7))**2,0)) for a in gray];ct=[np.median(g.sample(c,grid8)) for c in conf];stats=[];cache={}
  for j in range(20):
   p=grid[inside(m[j],grid)&(g.sample(std[j],grid)>=4)];score=g.sample(cv2.cornerMinEigenVal(gray[j],7),p) if len(p) else np.array([]);p=p[thin(p,score,h,w)];dgood=lgood=both=ready=np.zeros(len(p),bool);dq=lq=np.empty((len(p),2),np.float32)
   if len(p):
    dq=p+g.sample(flows[f'f{j}'],p);dfb=np.linalg.norm(g.sample(flows[f'b{j}'],dq)+dq-p,axis=1)
    lq,ls,_=cv2.calcOpticalFlowPyrLK(gray[j],gray[j+1],p[:,None,:],None,winSize=(21,21),maxLevel=3,criteria=(3,30,.01));back,bs,_=cv2.calcOpticalFlowPyrLK(gray[j+1],gray[j],lq,None,winSize=(21,21),maxLevel=3,criteria=(3,30,.01));lq=lq[:,0];lfb=np.linalg.norm(back[:,0]-p,axis=1)
    def good(q,fb):return np.isfinite(q).all(1)&(q[:,0]>=7)&(q[:,0]<w-7)&(q[:,1]>=7)&(q[:,1]<h-7)&(fb<=1)&inside(m[j+1],q)&(g.sample(std[j+1],q)>=4)&(patch_error(gray[j],gray[j+1],p,q)<=20)
    dgood=good(dq,dfb);lgood=good(lq,lfb)&(ls.ravel()>0)&(bs.ravel()>0);both=dgood&lgood&(np.linalg.norm(dq-lq,axis=1)<=1);ready=both&(g.sample(conf[j],p)>=ct[j])&(g.sample(conf[j+1],dq)>=ct[j+1])
   stats.append({'pair':j,'source':len(p),'DIS_2D':support(p[dgood],h,w),'LK_2D':support(p[lgood],h,w),'consensus_2D':support(p[both],h,w),'consensus_depth_ready':support(p[ready],h,w)})
   cache[f'p{j}']=p[both];cache[f'q{j}']=dq[both];cache[f'lk_q{j}']=lq[both];cache[f'depth_ready{j}']=ready[both]
  np.savez_compressed(out/(key+'.npz'),**cache);counts={};wins={}
  for name in ['DIS_2D','LK_2D','consensus_2D','consensus_depth_ready']:
   a=np.array([s[name]['sufficient'] for s in stats]);counts[name]=int(a.sum());wins[name]=bool(a.sum()>=16 and a[:4].sum()>=3 and a[-4:].sum()>=3)
  rows.append({'key':key,'stats':stats,'sufficient_pairs':counts,'sufficient_window':wins});write(out/'results.json',rows);print(key,counts,flush=True)
 write(out/'checks.json',{'exit_status':0,'new_depth_calls':0,'classifier_fits':0,'low_confidence_depth_not_promoted':True})
if __name__=='__main__':main()
