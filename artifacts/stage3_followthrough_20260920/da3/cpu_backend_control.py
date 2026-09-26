"""Bounded backend-only follow-up prompted by a large same-input CPU/MPS discrepancy."""
from smoke import O,R,load,InputProcessor
from pilot import feature,align_error,sha,write
import numpy as np,pandas as pd,torch,time,json
P=R/'artifacts/stage3_point_motion_20260919'
def main():
 dest=O/'cpu_control';dest.mkdir()
 windows=json.loads((P/'windows.json').read_text())[:3];assert [w['label'] for w in windows]==[0,1,2]
 write(dest/'freeze.json',{'source_sha256':sha(__import__('pathlib').Path(__file__)),'reason':'CPU/MPS same-input21-frame discrepancy observed; isolate backend before attributing all failure to geometric representation','keys':[w['key'] for w in windows],'fixed':'sameweights,21frames504resolution,float32,officialcamerahead; exact first3windows from originalprefrozenorder;first/reverse restored/middle; no refitting or thresholds change','changed':'device MPS toCPU only','cap_seconds':180,'first_case_first_variant':'reuses saved CPU crosscheck'})
 torch.set_num_threads(2);torch.manual_seed(42);start=time.perf_counter();model=load('cpu');rows=[];summary=[]
 for w in windows:
  z=np.load(P/(w['key']+'_frames.npz'));x,_,_=InputProcessor()(list(z['rgb']),process_res=504,num_workers=1)
  if x.ndim==4:x=x.unsqueeze(0)
  centers={}
  for variant,rev,ref in [('first',False,'first'),('reverse',True,'first'),('middle',False,'middle')]:
   assert time.perf_counter()-start<180
   began=time.perf_counter()
   if w==windows[0] and variant=='first':
    saved=np.load(O/'cpu_crosscheck.npz');ext=saved['extrinsics'];c=saved['centers'];reused=True
   else:
    with torch.inference_mode():out=model(x.flip(1) if rev else x,ref_view_strategy=ref,use_ray_pose=False,infer_gs=False)
    ext=out['extrinsics'][0].numpy()
    if rev:ext=ext[::-1].copy()
    c=-np.einsum('nij,nj->ni',ext[:,:3,:3].transpose(0,2,1),ext[:,:3,3]);reused=False
   assert np.isfinite(c).all();centers[variant]=c
   np.savez_compressed(dest/f'{w["key"]}_{variant}.npz',extrinsics=ext,centers=c,time=z['time'])
   feat=feature(c,z['time']);mps=np.load(O/f'poses/{w["key"]}_{variant}.npz')['centers'];m=feature(mps,z['time'])
   rows.append({'key':w['key'],'label':w['label'],'variant':variant,'sensor_q':w['sensor_ratio_a_v'],**feat,'MPS_q':m['q'],'CPU_MPS_q_difference':abs(feat['q']-m['q']),'CPU_MPS_Sim3_error':align_error(mps,c),'seconds':time.perf_counter()-began,'reused':reused})
   print(w['key'],variant,feat['q'],flush=True)
  q=[feature(centers[k],z['time'])['q'] for k in ['first','reverse','middle']]
  diffs=[abs(q[i]-q[0]) for i in [1,2]];errs=[align_error(centers['first'],centers[k]) for k in ['reverse','middle']]
  summary.append({'key':w['key'],'label':w['label'],'sensor_q':w['sensor_ratio_a_v'],'q_first_reverse_middle':q,'q_differences':diffs,'sim3_errors':errs,'stable':max(diffs)<=.02 and max(errs)<=.1})
 pd.DataFrame(rows).to_csv(dest/'results.csv',index=False);write(dest/'summary.json',{'windows':summary,'seconds':time.perf_counter()-start,'stable':sum(a['stable'] for a in summary),'n':3,'new_fit':False,'model_or_features_changed':False,'scope':'selected development windows; backend-only failure diagnostic, not independent performance'})
if __name__=='__main__':main()
