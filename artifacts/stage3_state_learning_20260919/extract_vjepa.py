"""Frozen official EMA encoder only; local PyAV, no official CUDA data pipeline."""
from pathlib import Path
import sys,json,time,hashlib,subprocess,warnings
O=Path(__file__).resolve().parent;R=O.parents[1]
sys.path[:0]=[str(O/'vendor'),str(O/'vjepa2_source')]
import numpy as np
import torch
import cv2,av
from app.vjepa_2_1.models.vision_transformer import vit_base
from run import data,sha,write
torch.set_num_threads(4);cv2.setNumThreads(2)

def frames(c):
    public=c['public'];want=None if public else set(map(int,np.load(R/'artifacts/stage3_training_basis_20260917'/c['labels_npz'])['frame_index']))
    path=R/(f'artifacts/public_eval_10hz/stage3/videos/{c["id"]}.mp4' if public else c['raw_path'])
    out=[]
    with av.open(str(path)) as con:
        for k,f in enumerate(con.decode(video=0)):
            if want is not None and k not in want:continue
            im=f.to_ndarray(format='rgb24');h,w=im.shape[:2];short=int(384*256/224)
            if h<w:nh,nw=short,int(w*short/h)
            else:nh,nw=int(h*short/w),short
            im=cv2.resize(im,(nw,nh),interpolation=cv2.INTER_LINEAR)
            y,x=(nh-384)//2,(nw-384)//2;out.append(im[y:y+384,x:x+384])
            if len(out)==c['n']:break
    assert len(out)==c['n'];return out

def tensor(ims):
    x=torch.from_numpy(np.stack(ims)).permute(3,0,1,2).float()/255
    return ((x-torch.tensor([.485,.456,.406])[:,None,None,None])/torch.tensor([.229,.224,.225])[:,None,None,None])[None]

def infer(m,x,dev):
    t=time.perf_counter()
    with torch.inference_mode():z=m(x.to(dev)).reshape(1,8,24*24,768).mean(2)[0].cpu().numpy()
    assert np.isfinite(z).all();return z,time.perf_counter()-t

def main():
    cs,_=data('dis');p=O/'weights/vjepa2_1_vitb_dist_vitG_384.pt'
    m=vit_base(patch_size=16,img_size=(384,384),num_frames=64,tubelet_size=2,use_sdpa=True,
               use_SiLU=False,wide_SiLU=True,uniform_power=False,use_rope=True,
               img_temporal_dim_size=1,interpolate_rope=True)
    checkpoint=torch.load(p,map_location='cpu',weights_only=True)
    state={k.replace('module.','').replace('backbone.',''):v for k,v in checkpoint['ema_encoder'].items()}
    m.load_state_dict(state,strict=True);del checkpoint,state
    m.requires_grad_(False).eval()
    # Freeze source, runtime and adapter details before inspecting feature/classification results.
    write(O/'vjepa_provenance.json',{'source_commit':subprocess.check_output(['git','-C',str(O/'vjepa2_source'),'rev-parse','HEAD'],text=True).strip(),
        'url':'https://dl.fbaipublicfiles.com/vjepa2/vjepa2_1_vitb_dist_vitG_384.pt','weight_sha256':sha(p),
        'weight_bytes':p.stat().st_size,'parameters':sum(p.numel() for p in m.parameters()),'strict_load':True,
        'source_license':'vjepa2_source/LICENSE (MIT; individual files retain notices)',
        'vendor':'timm1.0.15,einops0.8.1 --no-deps isolated --target; original venv unchanged',
        'adapter':'official encoder unchanged, bypass hub localhost URL and unused predictor/decord; PyAV RGB, official short-side int(384*256/224)=438 and384center crop, OpenCV bilinear; ImageNet normalize; 16frames@10Hz; disjoint windows; spatial mean only, tubelet timestamps i+0.5 every2frames; linear interpolation to10Hz, final frame repeat padding; no temporal averaging',
        'freeze_amendment':'pre-extraction correction of initial shortside384 to official438; no labels/results used; no change to DIS recipe',
        'case_collision':'official git configs vitg/vitG collide on case-insensitive macOS; these config files unused by direct ViT-B encoder adapter',
        'script_sha256':sha(Path(__file__))})
    ims=frames(cs['extra_02']);x=tensor(ims[:16]);cpu,ct=infer(m,x,'cpu')
    m.to('mps');mp,mt=infer(m,x,'mps')
    check={'torch':torch.__version__,'mps_available':torch.backends.mps.is_available(),'cpu_seconds':ct,'mps_seconds':mt,
           'mean_abs':float(abs(mp-cpu).mean()),'max_abs':float(abs(mp-cpu).max()),
           'relative_l2':float(np.linalg.norm(mp-cpu)/np.linalg.norm(cpu)),
           'pooled_shape':list(mp.shape),'finite':True,'memory_allocated':torch.mps.current_allocated_memory(),
           'driver_memory':torch.mps.driver_allocated_memory(),'comparison':'same real16frame RGB input, float32; feature-level only, not all model/class predictions equivalence'}
    write(O/'vjepa_runtime_check.json',check);print('smoke',json.dumps(check),flush=True)
    assert check['relative_l2']<.01,'CPU/MPS feature discrepancy blocks extraction'
    dest=O/'vjepa_features';dest.mkdir(exist_ok=True)
    for id,c in cs.items():
        if (dest/(id+'.npz')).exists():continue
        start=time.perf_counter();ims=frames(c);zs=[];anchors=[];enc=0.
        for k in range(0,len(ims),16):
            chunk=ims[k:k+16];chunk+= [chunk[-1]]*(16-len(chunk))
            z,elapsed=infer(m,tensor(chunk),'mps');zs.append(z);enc+=elapsed
            anchors.extend((k+np.arange(8)*2+.5).tolist())
        z=np.concatenate(zs);out=np.stack([np.interp(np.arange(c['n']),anchors,z[:,j]) for j in range(768)],1).astype(np.float32)
        np.savez_compressed(dest/(id+'.npz'),base=out,anchor_index=np.array(anchors),anchor_features=z)
        write(dest/(id+'_timing.json'),{'frames':c['n'],'windows':len(zs),'seconds':time.perf_counter()-start,'encoder_seconds':enc,'input_path':c.get('raw_path',f'artifacts/public_eval_10hz/stage3/videos/{id}.mp4')})
        print(id,c['n'],round(time.perf_counter()-start,2),'seconds',flush=True)
    write(O/'vjepa_extraction_checks.json',{'completed':True,'videos':len(cs),'total_frames':sum(c['n'] for c in cs.values()),'10hz_shape_valid':True})

if __name__=='__main__':main()
