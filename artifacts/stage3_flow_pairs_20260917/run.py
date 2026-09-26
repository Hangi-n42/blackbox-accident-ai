"""Direct DIS field comparison on frozen same-video case/control pairs; no model fitting."""
from pathlib import Path
import json,sys,hashlib
import numpy as np,pandas as pd,cv2,av
from PIL import Image,ImageDraw
from scipy.ndimage import uniform_filter1d
O=Path(__file__).resolve().parent;R=O.parents[1];B=R/'artifacts/stage3_training_basis_20260917';D=R/'artifacts/stage3_reversal_diagnosis_20260917'
sys.path.insert(0,str(R/'artifacts/submissions/verify_v6/model/stage2/code'))
from solution.stage3_v5_compatible import FrameFeatureComputer
write=lambda n,x:(O/n).write_text(json.dumps(x,ensure_ascii=False,indent=2));sha=lambda p:hashlib.file_digest(p.open('rb'),'sha256').hexdigest()
cv2.setNumThreads(2)

def main():
 assert not (O/'results.json').exists()
 cases=json.load(open(B/'cases.json'));full=pd.read_csv(D/'full_context.csv');chosen=pd.read_csv(D/'selected_74.csv');pairs=[];matches=[]
 for id in ['expanded_14','extra_01']:
  c=next(c for c in cases if c['id']==id);d=np.load(B/c['labels_npz']);g=full[full.id==id];controls=g[g.strict_mask&(g.proxy_truth==0)&(g.rav4_prediction==0)&(g.mixed_budget_rav4_prediction==0)]
  for z in chosen[chosen.id==id].itertuples():
   score=((controls.speed_mps-z.speed_mps)/2)**2+((controls.accel_proxy-z.accel_proxy)/.3)**2
   idx=score.idxmin();ctrl=controls.loc[idx];matches.append({'id':id,'error_index':int(z.index),'control_index':int(ctrl['index']),'speed_delta':float(ctrl.speed_mps-z.speed_mps),'accel_delta':float(ctrl.accel_proxy-z.accel_proxy),'within_descriptive_caliper':bool(abs(ctrl.speed_mps-z.speed_mps)<=2 and abs(ctrl.accel_proxy-z.accel_proxy)<=.3)})
 # Fixed representatives: midpoint of largest error run in expanded_14, and longest run
 # in each of the two extra_01 episodes. Selection is independent of flow results.
 for id,i in [('expanded_14',169),('extra_01',283),('extra_01',439)]:
  m=next(m for m in matches if m['id']==id and m['error_index']==i);pairs.append(m)
 write('freeze.json',{'representatives':pairs,'all74_matches':matches,'matching':'same-video both-correct strict acceleration; minimize (delta speed/2mps)^2+(delta proxy acceleration/.3)^2; steering not matched','caliper':'2m/s,.3m/s2 diagnostic rule, not validated equivalence; report unmatched representatives','flow':'same width256 DIS_FAST, sequential10Hz native-frame selection; raw field normalized per second','residual':'subtract full-frame median vector; diagnostic translation proxy, not physical ego-motion compensation','script_sha256':sha(Path(__file__))})
 outputs=[];pictures={};checks={}
 for id in ['expanded_14','extra_01']:
  c=next(c for c in cases if c['id']==id);d=np.load(B/c['labels_npz']);targets=set()
  for p in pairs:
   if p['id']==id:
    for k in ['error_index','control_index']:targets.update(range(max(1,p[k]-15),min(len(d['time'])-1,p[k]+17)))
  native_to_index={int(k):i for i,k in enumerate(d['frame_index'])};dis=cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_FAST);ff=FrameFeatureComputer();prev=None;raw=[];fields={};last=max(native_to_index)
  with av.open(str(R/c['raw_path'])) as con:
   for native,frame in enumerate(con.decode(video=0)):
    if native in native_to_index:
     i=native_to_index[native];bgr=frame.to_ndarray(format='bgr24');h=max(96,int(round(bgr.shape[0]*256/bgr.shape[1])));rgb=cv2.resize(bgr,(256,h));gray=cv2.cvtColor(rgb,cv2.COLOR_BGR2GRAY)
     if prev is not None:
      flow=dis.calc(prev,gray,None)*10;raw.append(ff(flow))
      if i in targets:fields[i]=flow.copy()
     if any(p['id']==id and i in [p['error_index'],p['control_index']] for p in pairs):pictures[(id,i)]=cv2.cvtColor(rgb,cv2.COLOR_BGR2RGB)
     prev=gray
    if native>=last:break
  raw=np.array(raw);centers=np.arange(1,len(raw)+1)-.5;cached_path=R/c['feature_cache'] if c['feature_cache'] else B/(id+'_features.npy');a=np.load(cached_path);cached=a['features'] if isinstance(a,np.lib.npyio.NpzFile) else a
  reconstructed=np.stack([np.interp(np.arange(len(d['time'])),centers,raw[:,j]) for j in range(144)],axis=1).astype(np.float32)
  err=float(np.max(abs(reconstructed-cached[:,:144])));assert err<1e-6,(id,err);checks[id]={'raw144_cache_max_abs_difference':err,'selected_native_frames':len(d['time']),'source_path':c['raw_path']}
  for p in pairs:
   if p['id']!=id:continue
   record=dict(p,observations={})
   for kind,key in [('error','error_index'),('control','control_index')]:
    i=p[key];field=fields[i];h,w=field.shape[:2];glob=np.median(field.reshape(-1,2),axis=0);res=field-glob;rois={};avg=raw[max(0,i-16):min(len(raw),i+16)].mean(0).reshape(12,4,3)
    for roi in ([1,5,6] if id=='expanded_14' else [6,10]):
     ya,yb=[(.15,.50),(.50,.72),(.72,.93)][roi//4];xa,xb=[(.03,.27),(.27,.5),(.5,.73),(.73,.97)][roi%4];sl=(slice(int(ya*h),int(yb*h)),slice(int(xa*w),int(xb*w)));f=field[sl];re=res[sl]
     # Same model mean31, sample-center interpolated; matched cached temporal block.
     smooth=uniform_filter1d(raw,size=31,axis=0,mode='nearest');at=np.array([np.interp(i,centers,smooth[:,j]) for j in range(144)]).reshape(12,4,3)
     assert np.allclose(at.ravel(),cached[i,432:576],atol=1e-6,rtol=0)
     rois[str(roi)]={'median_flow_px_per_s':np.median(f.reshape(-1,2),axis=0).tolist(),'median_residual_px_per_s':np.median(re.reshape(-1,2),axis=0).tolist(),'median_residual_magnitude_px_per_s':float(np.median(np.linalg.norm(re,axis=2))),'model_mean31_median_channels':dict(zip(['horizontal','vertical','magnitude','radial'],at[roi,:,0].tolist()))}
    record['observations'][kind]={'index':i,'time_s':i/10,'speed_mps':float(d['speed_smoothed'][i]),'accel_proxy':float(d['acceleration_proxy'][i]),'steering_deg':float(d['steering_smoothed'][i]),'global_median_px_per_s':glob.tolist(),'rois':rois,'flow_interval_sample_indices':[i-1,i],'native_indices':[int(d['frame_index'][i-1]),int(d['frame_index'][i])],'source_frame_times':[float(d['frame_time'][i-1]),float(d['frame_time'][i])]}
    np.savez_compressed(O/f'{id}_{i}_flow.npz',flow_px_per_s=field,median_translation=glob,residual_px_per_s=res)
    # Equal arrow scale across cases: per-frame displacement *3 for visibility.
    im=Image.fromarray(pictures[(id,i)]).resize((512,h*2));draw=ImageDraw.Draw(im)
    for yy in range(10,h-5,10):
     for xx in range(10,w-5,10):
      dx,dy=field[yy,xx]/10*6;draw.line((xx*2,yy*2,xx*2+float(dx),yy*2+float(dy)),fill=(255,235,20),width=1)
    for roi in map(int,rois):
     ya,yb=[(.15,.50),(.50,.72),(.72,.93)][roi//4];xa,xb=[(.03,.27),(.27,.5),(.5,.73),(.73,.97)][roi%4];draw.rectangle((int(xa*512),int(ya*h*2),int(xb*512),int(yb*h*2)),outline='red',width=2);draw.text((int(xa*512)+3,int(ya*h*2)+3),str(roi),fill='red')
    im.save(O/f'{id}_{i}_overlay.png')
   outputs.append(record)
 write('results.json',{'pairs':outputs,'checks':checks,'new_fits':0,'all74_match_within_caliper':sum(m['within_descriptive_caliper'] for m in matches),'n_matches':74})
 print(json.dumps(outputs,indent=2))
if __name__=='__main__':main()
