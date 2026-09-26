"""CPU-only depth/uncertainty/intrinsics; ignore direct learned trajectory."""
from prepare import *
import time,torch,importlib.util
spec=importlib.util.spec_from_file_location('old_smoke',D/'smoke.py');prior=importlib.util.module_from_spec(spec);spec.loader.exec_module(prior)
from depth_anything_3.utils.io.input_processor import InputProcessor

def main():
 f=read(O/'freeze.json');assert all(sha(R/p)==h for p,h in f['protected'].items());dest=O/'depth';dest.mkdir()
 torch.set_num_threads(2);torch.manual_seed(42);model=prior.load('cpu');logs=[]
 for row in read(O/'manifest.json'):
  frames=np.load(R/row['frames_path']);x,_,_=InputProcessor()(list(frames['rgb']),process_res=504,num_workers=1)
  if x.ndim==4:x=x.unsqueeze(0)
  rgb=((x[0].permute(0,2,3,1).numpy()*np.array([.229,.224,.225])+np.array([.485,.456,.406]))*255).round().clip(0,255).astype('uint8')
  np.savez_compressed(dest/(row['key']+'_input.npz'),rgb=rgb,time=frames['time'])
  for variant,rev,ref in [('first',False,'first'),('reverse',True,'first'),('middle',False,'middle')]:
   start=time.monotonic()
   with torch.inference_mode():out=model(x.flip(1) if rev else x,ref_view_strategy=ref,use_ray_pose=False,infer_gs=False)
   data={k:out[k][0].detach().cpu().numpy() for k in ['depth','depth_conf','intrinsics']}
   if rev:data={k:v[::-1].copy() for k,v in data.items()}
   assert all(np.isfinite(v).all() for v in data.values()) and np.all(data['depth']>0)
   np.savez_compressed(dest/(row['key']+'_'+variant+'.npz'),**data)
   logs.append({'key':row['key'],'variant':variant,'seconds':time.monotonic()-start,'shapes':{k:list(v.shape) for k,v in data.items()}});write(O/'inference_log.json',logs)
   print(row['key'],variant,round(logs[-1]['seconds'],2),'sec',flush=True)
 assert len(logs)==24
 write(O/'inference_checks.json',{'completed':True,'calls':24,'CPU_FP32':True,'same_frozen_weights':True,'no_direct_extrinsic_used':True,'protected_hashes_unchanged':all(sha(R/p)==h for p,h in f['protected'].items())})
if __name__=='__main__':main()
