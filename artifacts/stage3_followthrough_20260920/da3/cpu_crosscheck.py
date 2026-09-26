"""One same-input CPU/MPS numerical cross-check, no candidate selection."""
from smoke import O,R,load,InputProcessor
from pilot import feature,align_error
import numpy as np,torch,time,json

def main():
 torch.set_num_threads(2);torch.manual_seed(42);start=time.perf_counter();key='expanded_11_225'
 z=np.load(R/f'artifacts/stage3_point_motion_20260919/{key}_frames.npz');model=load('cpu')
 x,_,_=InputProcessor()(list(z['rgb']),process_res=504,num_workers=1)
 if x.ndim==4:x=x.unsqueeze(0)
 with torch.inference_mode():out=model(x,ref_view_strategy='first',use_ray_pose=False,infer_gs=False)
 e=out['extrinsics'][0].numpy();c=-np.einsum('nij,nj->ni',e[:,:3,:3].transpose(0,2,1),e[:,:3,3]);assert np.isfinite(c).all()
 mps=np.load(O/f'poses/{key}_first.npz');np.savez_compressed(O/'cpu_crosscheck.npz',extrinsics=e,centers=c,time=z['time'])
 a=feature(c,z['time']);b=feature(mps['centers'],z['time'])
 report={'key':key,'CPU':a,'MPS':b,'q_abs_difference':abs(a['q']-b['q']),'extrinsics_max_abs_difference':float(abs(e-mps['extrinsics']).max()),'trajectory_sim3_error':align_error(mps['centers'],c),'seconds':time.perf_counter()-start,'scope':'single same21frames, samefloat32 weights/preprocessing/reference. Neither result is pose ground truth.'}
 (O/'cpu_crosscheck.json').write_text(json.dumps(report,indent=2));print(report,flush=True)
if __name__=='__main__':main()
