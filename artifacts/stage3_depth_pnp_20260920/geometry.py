"""Fixed DIS correspondences, depth-backed bidirectional PnP and q checks."""
from prepare import *
import cv2

def sample(a,p):
 out=cv2.remap(a,p[:,0].astype('float32')[:,None],p[:,1].astype('float32')[:,None],cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT,borderValue=float('nan'))
 return out.reshape(len(p),-1) if a.ndim==3 else out.ravel()

def roi_grid(h,w,step=8):
 yy,xx=np.mgrid[int(.25*h):int(.80*h):step,int(.08*w):int(.92*w):step]
 return np.c_[xx.ravel(),yy.ravel()].astype('float32')

def backproject(p,z,k):return (np.c_[p,np.ones(len(p))]@np.linalg.inv(k).T)*z[:,None]

def solve(x,q,k):
 if len(x)<30:return None
 cv2.setRNGSeed(42)
 ok,r,t,ii=cv2.solvePnPRansac(np.ascontiguousarray(x,np.float64),np.ascontiguousarray(q,np.float64),k,None,iterationsCount=300,reprojectionError=2.,confidence=.999,flags=cv2.SOLVEPNP_EPNP)
 if not ok or ii is None or len(ii)<6:return None
 ii=ii.ravel();r,t=cv2.solvePnPRefineLM(x[ii].astype('float64'),q[ii].astype('float64'),k,None,r,t)
 rot=cv2.Rodrigues(r)[0];pred=cv2.projectPoints(x,r,t,k,None)[0].reshape(-1,2)
 return rot,t.ravel(),ii,np.linalg.norm(pred-q,axis=1),r.ravel()

def pair_pnp(p,q,d1,d2,k,dt):
 z1=sample(d1,p);z2=sample(d2,q);valid=np.isfinite(z1)&np.isfinite(z2)&(z1>0)&(z2>0);p=p[valid];q=q[valid];z1=z1[valid];z2=z2[valid];h,w=d1.shape
 ans=solve(backproject(p,z1,k),q,k);rev=solve(backproject(q,z2,k),p,k)
 if ans is None or rev is None:return {'valid':False,'reason':'solve_failure','points':len(p),'speed':None}
 rot,t,ii,err,rvec=ans;rr,tt,ri,re,rvr=rev;norm=np.linalg.norm(t);newz=(backproject(p,z1,k)@rot.T+t)[:,2]
 deptherr=float(np.median(abs(np.log(np.maximum(newz[ii],1e-12)/z2[ii]))));cycle=float(np.linalg.norm(rr@t+tt)/max(norm,1e-12));angle=float(np.linalg.norm(cv2.Rodrigues(rr@rot)[0])*180/np.pi)
 xy=p[ii];checks={'inliers':len(ii)>=30 and len(ri)>=30,'ratio':len(ii)/len(p)>=.5 and len(ri)/len(p)>=.5,'reprojection':np.median(err[ii])<=1 and np.median(re[ri])<=1,'spread':np.ptp(xy[:,0])>=.4*w and np.ptp(xy[:,1])>=.2*h,'depth_consistency':deptherr<=.10,'cycle_translation':cycle<=.25,'cycle_rotation':angle<=.5,'positive_translation':norm>1e-8}
 return {'valid':bool(all(checks.values())),'reason':','.join(k for k,v in checks.items() if not v) or 'passed','points':len(p),'inliers':len(ii),'reverse_inliers':len(ri),'reprojection_median':float(np.median(err[ii])),'depth_log_error':deptherr,'cycle_translation_ratio':cycle,'cycle_rotation_deg':angle,'translation':t.tolist(),'rvec':rvec.tolist(),'speed':float(norm/dt),'inlier_indices':ii.tolist()}

def correspondences(rgb,depth,conf):
 gray=[cv2.cvtColor(im,cv2.COLOR_RGB2GRAY) for im in rgb];h,w=gray[0].shape;grid=roi_grid(h,w);result=[];cache={}
 dis=cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_FAST)
 for j,(a,b) in enumerate(zip(gray[:-1],gray[1:])):
  flow=dis.calc(a,b,None);back=dis.calc(b,a,None);p=grid.copy();q=p+sample(flow,p)
  error=np.linalg.norm(sample(back,q)+q-p,axis=1)
  af=a.astype('float32');bf=b.astype('float32');std1=np.sqrt(np.maximum(cv2.blur(af*af,(7,7))-cv2.blur(af,(7,7))**2,0));std2=np.sqrt(np.maximum(cv2.blur(bf*bf,(7,7))-cv2.blur(bf,(7,7))**2,0))
  good=np.isfinite(q).all(1)&(q[:,0]>=2)&(q[:,0]<w-2)&(q[:,1]>=2)&(q[:,1]<h-2)&(error<=1)&(sample(std1,p)>=4)&(sample(std2,q)>=4)
  good&=(sample(conf[j],p)>=np.median(sample(conf[j],grid)))&(sample(conf[j+1],q)>=np.median(sample(conf[j+1],grid)))
  pp=p[good].astype('float32');qq=q[good].astype('float32');result.append((pp,qq));cache[f'p{j}']=pp;cache[f'q{j}']=qq
 return result,cache

def depth_stability(base,variants,t):
 dep=base['depth'];h,w=dep.shape[1:];p=roi_grid(h,w);d0=np.array([sample(d,p) for d in dep]);cf=np.array([sample(d,p) for d in base['depth_conf']]);keep=cf>=np.median(cf,axis=1)[:,None];rows=[]
 kk=base['intrinsics'];km=np.median(kk,axis=0);foc=kk[:,[0,1],[0,1]];jitter=float(np.max(abs(foc/np.median(foc,axis=0)-1)))
 for name,var in variants.items():
  dd=np.array([sample(d,p) for d in var['depth']]);lr=np.log(dd/d0);global_log=float(np.median(lr[keep]));med=np.array([np.median(lr[j][keep[j]]) for j in range(21)])-global_log
  slope=float(np.polyfit(t-t[10],med,1)[0]);res=float(np.quantile(abs(lr[keep]-global_log),.9));drift=float(abs(med).max())
  ki=np.median(var['intrinsics'],axis=0);fd=float(np.max(abs(ki[[0,1],[0,1]]/km[[0,1],[0,1]]-1)));pd=float(np.max(abs(ki[:2,2]-km[:2,2])))
  rows.append({'variant':name,'global_scale_to_base':float(np.exp(-global_log)),'frame_log_scale_residuals':med.tolist(),'temporal_log_scale_slope':slope,'max_frame_log_scale_drift':drift,'spatial_log_ratio_p90':res,'focal_relative_change':fd,'principal_change_px':pd,'pass':bool(abs(slope)<=.02 and drift<=.05 and res<=.15 and fd<=.05 and pd<=3)})
 return {'base_intrinsics_focal_jitter':jitter,'variants':rows,'pass':bool(jitter<=.05 and all(r['pass'] for r in rows))},km

def q_feature(rows,t):
 good=np.array([r['valid'] for r in rows]);alltime=(t[:-1]+t[1:])/2;v=np.array([r['speed'] if r['speed'] is not None else np.nan for r in rows]);quality=bool(good.sum()>=16 and good[:4].sum()>=3 and good[-4:].sum()>=3)
 value=float(np.polyfit(alltime[good]-t[10],np.log(v[good]),1)[0]) if good.sum()>=3 else None
 return {'q':value,'quality':quality,'valid_pairs':int(good.sum()),'valid_indices':np.flatnonzero(good).tolist(),'relative_speeds':v.tolist()}

def synthetic():
 rng=np.random.default_rng(42);k=np.array([[450.,0,252],[0,450,189],[0,0,1]]);p=roi_grid(378,504,10).astype('float64');z=rng.uniform(15,70,len(p));x=backproject(p,z,k);results=[]
 # Exact3D/2D correspondences and known source-depth make this an algebra/solver test, not image accuracy.
 for acc in [-1,0,1]:
  time=np.linspace(-1,1,21);speeds=[];scaled=[]
  for j in range(20):
   mid=(time[j]+time[j+1])/2;v=10+acc*mid;rv=np.array([.001,-.002,.0005]);rot=cv2.Rodrigues(rv)[0];tr=np.array([0,0,-v*.1]);q=cv2.projectPoints(x,rv,tr,k,None)[0].reshape(-1,2)
   a=solve(x,q,k);b=solve(x*3,q,k);assert a is not None and b is not None
   assert np.linalg.norm(a[0]-rot)<1e-6 and np.linalg.norm(a[1]-tr)<1e-5
   speeds.append(np.linalg.norm(a[1])/.1);scaled.append(np.linalg.norm(b[1])/.1)
  tt=(time[:-1]+time[1:])/2;truth=float(np.polyfit(tt,np.log(10+acc*tt),1)[0]);q=float(np.polyfit(tt,np.log(speeds),1)[0]);qs=float(np.polyfit(tt,np.log(scaled),1)[0]);assert abs(q-truth)<1e-6 and abs(qs-q)<1e-6
  # A smooth framewise scale drift directly contaminates q, unlike one global window scale.
  qdrift=float(np.polyfit(tt,np.log(np.array(speeds)*np.exp(.1*tt)),1)[0]);assert abs(qdrift-q-.1)<1e-8
  results.append({'acceleration':acc,'expected_q':truth,'q':q,'global3x_q':qs,'injected_scale_drift_q':qdrift})
 write(O/'synthetic_checks.json',{'passed':True,'rows':results,'scope':'Exact geometry solver/formula checks; no claim of real-image accuracy'})

def main():
 cv2.setNumThreads(2);synthetic();manifest=read(O/'manifest.json');out=[];O.joinpath('flow').mkdir()
 for row in manifest:
  key=row['key'];raw=np.load(O/'depth'/(key+'_input.npz'));variants={v:dict(np.load(O/'depth'/(key+'_'+v+'.npz'))) for v in ['first','reverse','middle']};base=variants['first'];t=raw['time'];stability,k=depth_stability(base,{v:variants[v] for v in ['reverse','middle']},t)
  matches,cache=correspondences(raw['rgb'],base['depth'],base['depth_conf']);np.savez_compressed(O/'flow'/(key+'.npz'),**cache,K=k,time=t)
  res={}
  for name,dep in variants.items():
   pairs=[pair_pnp(p,q,dep['depth'][j],dep['depth'][j+1],k,float(t[j+1]-t[j])) for j,(p,q) in enumerate(matches)]
   feature=q_feature(pairs,t);res[name]={**feature,'pairs':pairs}
  qs=[r['q'] for r in res.values()];stable_q=all(q is not None for q in qs) and max(qs)-min(qs)<=.02
  valid=bool(stability['pass'] and stable_q and all(r['quality'] for r in res.values()))
  r={'key':key,'dataset':row['dataset'],'label':row['label'],'sensor_q':row['sensor_logspeed_slope'],'center_sensor_q':row['sensor_ratio_a_v'],'depth_stability':stability,'features':res,'q_stable':stable_q,'valid':valid};out.append(r);write(O/'results.json',out)
  print(key,'valid',valid,'depthstable',stability['pass'],[(n,v['valid_pairs'],v['q']) for n,v in res.items()],flush=True)
 valid=[r for r in out if r['valid']];counts={k:sum(r['label']==k for r in valid) for k in [0,1,2]};qerr=[abs(r['features']['first']['q']-r['sensor_q']) for r in valid];zero=[abs(r['sensor_q']) for r in valid]
 gates={'quality_coverage6':len(valid)>=6,'class_coverage':counts[0]>=1 and counts[1]>=1 and counts[2]>=3,
  'valid_AD_correct_and_close':all(r['features']['first']['q']*r['sensor_q']>0 and abs(r['features']['first']['q']-r['sensor_q'])<=.03 for r in valid if r['label']!=2) if counts[0]+counts[1] else False,
  'valid_C_near_zero':all(abs(r['features']['first']['q'])<=.02 for r in valid if r['label']==2) if counts[2] else False,
  'MAE_below_zero':bool(qerr and np.mean(qerr)<np.mean(zero))}
 write(O/'decision.json',{'gates':gates,'expand':bool(all(gates.values())),'valid_counts':counts,'valid_q_MAE':float(np.mean(qerr)) if qerr else None,'zero_baseline_MAE':float(np.mean(zero)) if zero else None,'classifier_fits':0,'score_measured':False})
 assert all(sha(R/p)==h for p,h in read(O/'freeze.json')['protected'].items())
if __name__=='__main__':main()
