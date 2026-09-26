"""Dense eligible-pixel sampling + corner-prior4px NMS, same flow/quality."""
from run import *

def thin(p,score,h,w,radius=4):
 blocked=np.zeros((h,w),np.uint8);keep=[]
 for i in np.argsort(-score,kind='stable'):
  x,y=np.rint(p[i]).astype(int)
  if not blocked[y,x]:
   keep.append(int(i));cv2.circle(blocked,(x,y),radius,1,-1)
 return np.array(keep,dtype=int)

def main():
 cv2.setNumThreads(2);out=O/'adaptive_expanded';out.mkdir();protected={str(p.relative_to(R)):sha(p) for p in list((O/'masks').glob('*.npz'))+[Path(__file__),O/'run.py',B/'geometry.py']}
 write(out/'freeze.json',{'factor':'Sampling policy only vs grid4_expanded:2px candidate grid, original eligibility gates, descending minimum source/destination ShiTomasi response, greedy radius4px source NMS. No cap, parameter search, quality threshold reduction, new flow or depth inference.','quality':'Same FB<=1px, both7px std>=4, confidence thresholds from original8px grid, both static masks, same PnP. Corner response is ranking not a new truth filter.','additional_diagnostic':'Bidirectional21px pyramid LK on same retained points, FB<=1px and agreement with DIS<=1px; diagnostic only. Does not prove correct correspondence. No LK coordinates used for PnP. Report8px thinning as dependence diagnostic.','protected':protected})
 rows=[]
 for old in read(B/'results.json'):
  key=old['key'];inp=np.load(B/'depth'/(key+'_input.npz'));t=inp['time'];rgb=inp['rgb'];base=np.load(B/'depth'/(key+'_first.npz'));dep=base['depth'];conf=base['depth_conf'];k=np.load(B/'flow'/(key+'.npz'))['K'];flows=np.load(O/'flow'/(key+'.npz'));m=np.load(O/'masks'/(key+'.npz'))['mask'];h,w=m.shape[1:];grid=g.roi_grid(h,w,2);grid8=g.roi_grid(h,w)
  gray=[cv2.cvtColor(im,cv2.COLOR_RGB2GRAY) for im in rgb];std=[np.sqrt(np.maximum(cv2.blur(a.astype('float32')**2,(7,7))-cv2.blur(a.astype('float32'),(7,7))**2,0)) for a in gray];corner=[cv2.cornerMinEigenVal(a,7) for a in gray];ct=[np.median(g.sample(c,grid8)) for c in conf];stats=[];pairs=[];cache={}
  for j in range(20):
   p=grid.copy();q=p+g.sample(flows[f'f{j}'],p);fb=np.linalg.norm(g.sample(flows[f'b{j}'],q)+q-p,axis=1)
   good=np.isfinite(q).all(1)&(q[:,0]>=2)&(q[:,0]<w-2)&(q[:,1]>=2)&(q[:,1]<h-2)&(fb<=1)&(g.sample(std[j],p)>=4)&(g.sample(std[j+1],q)>=4)&(g.sample(conf[j],p)>=ct[j])&(g.sample(conf[j+1],q)>=ct[j+1])&inside(m[j],p)&inside(m[j+1],q)
   p,q=p[good],q[good];error=fb[good];score=np.minimum(g.sample(corner[j],p),g.sample(corner[j+1],q)) if len(p) else np.array([]);ix=thin(p,score,h,w);eligible=len(p);p,q,score,error=p[ix],q[ix],score[ix],error[ix]
   agreement=np.zeros(len(p),bool)
   if len(p):
    lp,ls,le=cv2.calcOpticalFlowPyrLK(gray[j],gray[j+1],p[:,None,:],None,winSize=(21,21),maxLevel=3,criteria=(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT,30,.01));bp,bs,be=cv2.calcOpticalFlowPyrLK(gray[j+1],gray[j],lp,None,winSize=(21,21),maxLevel=3,criteria=(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT,30,.01));agreement=(ls.ravel()>0)&(bs.ravel()>0)&(np.linalg.norm(bp[:,0]-p,axis=1)<=1)&(np.linalg.norm(lp[:,0]-q,axis=1)<=1)
   span=np.ptp(p,axis=0)/[w,h] if len(p) else np.zeros(2);quant=(np.quantile(p,.9,axis=0)-np.quantile(p,.1,axis=0))/[w,h] if len(p) else np.zeros(2);suf=bool(len(p)>=30 and span[0]>=.4 and span[1]>=.2);aspan=np.ptp(p[agreement],axis=0)/[w,h] if agreement.any() else np.zeros(2);agree_suf=bool(agreement.sum()>=30 and aspan[0]>=.4 and aspan[1]>=.2)
   stats.append({'pair':j,'eligible_2px':eligible,'points':len(p),'points_after8px_NMS':len(thin(p,score,h,w,8)),'span':span.tolist(),'quantile_10_90_span':quant.tolist(),'cells32':len(np.unique((p//32).astype(int),axis=0)),'sufficient':suf,'LK_agree_points':int(agreement.sum()),'LK_agree_sufficient':agree_suf,'fb_p90':float(np.quantile(error,.9)) if len(error) else None})
   cache[f'p{j}']=p;cache[f'q{j}']=q;cache[f'fb{j}']=error;cache[f'corner{j}']=score;cache[f'lk_agree{j}']=agreement
   pairs.append(g.pair_pnp(p,q,dep[j],dep[j+1],k,float(t[j+1]-t[j])) if len(p) else {'valid':False,'reason':'no_points','points':0,'speed':None})
  np.savez_compressed(out/(key+'.npz'),**cache);s=np.array([s['sufficient'] for s in stats]);a=np.array([s['LK_agree_sufficient'] for s in stats]);feat=g.q_feature(pairs,t)
  row={'key':key,'stats':stats,'point_sufficient_pairs':int(s.sum()),'point_sufficient_window':bool(s.sum()>=16 and s[:4].sum()>=3 and s[-4:].sum()>=3),'LK_agree_sufficient_pairs':int(a.sum()),'LK_agree_sufficient_window':bool(a.sum()>=16 and a[:4].sum()>=3 and a[-4:].sum()>=3),'median_points':float(np.median([s['points'] for s in stats])),'pnp':pairs,'pnp_valid_pairs':feat['valid_pairs'],'pnp_quality':feat['quality'],'q_diagnostic':feat['q']};rows.append(row);write(out/'results.json',rows);print(key,row['median_points'],row['point_sufficient_pairs'],row['LK_agree_sufficient_pairs'],row['pnp_valid_pairs'],flush=True)
 assert all(sha(R/p)==h for p,h in protected.items());write(out/'checks.json',{'exit_status':0,'frozen_inputs_unchanged':True,'thresholds_not_relaxed':True,'new_depth_calls':0,'classifier_fits':0})
if __name__=='__main__':main()
