from pathlib import Path
import sys,json,time
import numpy as np,torch
O=Path(__file__).resolve().parent;R=O.parents[2]
sys.path[:0]=[str(O/'deps'),str(O/'vendor/Depth-Anything-3/src'),str(R/'artifacts/stage3_state_learning_20260919/vendor')]
from omegaconf import OmegaConf
from safetensors.torch import load_model
from depth_anything_3.cfg import create_object
from depth_anything_3.utils.io.input_processor import InputProcessor

def load(device):
    config=json.loads((O/'weights/config.json').read_text())
    model=create_object(OmegaConf.create(config['config'])).eval()
    wrapper=torch.nn.Module();wrapper.add_module('model',model)
    missing,unexpected=load_model(wrapper,str(O/'weights/model.safetensors'),strict=True)
    assert not missing and not unexpected  # safetensors omits shared LayerNorm aliases.
    return model.to(device)
def main():
    torch.set_num_threads(2);torch.manual_seed(42)
    device=sys.argv[1] if len(sys.argv)>1 else 'mps';start=time.perf_counter();model=load(device)
    z=np.load(R/'artifacts/stage3_point_motion_20260919/expanded_11_225_frames.npz')
    x,_,_=InputProcessor()(list(z['rgb'][[0,10,20]]),process_res=504,num_workers=1)
    if x.ndim==4:x=x.unsqueeze(0)
    print('input',x.shape,'parameters',sum(p.numel() for p in model.parameters()),flush=True)
    with torch.inference_mode():out=model(x.to(device),ref_view_strategy='first',use_ray_pose=False,infer_gs=False)
    e=out['extrinsics'].detach().cpu().numpy();assert np.isfinite(e).all()
    np.savez_compressed(O/f'smoke_{device}.npz',extrinsics=e,intrinsics=out['intrinsics'].detach().cpu().numpy())
    info={'device':device,'torch':torch.__version__,'seconds':time.perf_counter()-start,'input_shape':list(x.shape),'pose_shape':list(e.shape),'finite':True,'parameters':sum(p.numel() for p in model.parameters()),'float32':True,'strict_load':True,'no_vendor_edits':True}
    (O/f'smoke_{device}.json').write_text(json.dumps(info,indent=2));print(info,flush=True)
if __name__=='__main__':main()
