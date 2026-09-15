"""Actual saved processor, synthetic labeled tiles only. No model or GPU."""
import os,sys,json,hashlib,importlib.util,socket,time
from pathlib import Path
from types import SimpleNamespace
sys.dont_write_bytecode=True
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='2'
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];OUT=HERE/'native_video_processor_probe'
HELPER=HERE/'solution/native_video_v7.py';MODEL=ROOT/'artifacts/submissions/verify_v6/model/stage2/vlm'
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def put(name,obj):
    with (OUT/name).open('x',encoding='utf8') as f:json.dump(obj,f,ensure_ascii=False,indent=2,allow_nan=False)
def deny(*a,**k):raise RuntimeError('CPU processor probe offline')
def main():
    assert not OUT.exists();OUT.mkdir();start=time.perf_counter()
    files=[HELPER,Path(__file__)]+[p for p in MODEL.iterdir() if p.is_file() and p.suffix in ('.json','.txt','.jinja') and 'safetensors' not in p.name and 'MANIFEST' not in p.name]
    bindings={str(p):sha(p) for p in files}
    put('frozen_plan.json',{'K':[3,15,18],'tile_dimensions':[384,256],'budget':1200000,'divisor':'encoded_K=2*ceil(K/2)',
        'synthetic_fps':2,'physical_FPS_or_GT_used':False,'model_loaded':False,'GPU_used':False,'files':bindings})
    socket.socket.connect=deny;socket.socket.connect_ex=deny;socket.create_connection=deny
    import torch,numpy as np
    from PIL import Image,ImageDraw
    from transformers import AutoProcessor
    torch.set_num_threads(2);torch.set_num_interop_threads(2)
    processor=AutoProcessor.from_pretrained(str(MODEL),local_files_only=True,trust_remote_code=False)
    vlm=SimpleNamespace(processor=processor,torch=torch,pixel_budget=1200000,device='cpu')
    spec=importlib.util.spec_from_file_location('native_video_v7_probe',HELPER);helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
    reports=[]
    for K in (3,15,18):
        ids=[7+11*i*i+3*i for i in range(K)];tiles=[]
        for i,ID in enumerate(ids):
            im=Image.new('RGB',(384,256),((i*31+21)%255,(i*53+47)%255,(i*71+83)%255))
            ImageDraw.Draw(im).text((8,6),f'frame {ID}',fill='white');tiles.append(im)
        before=[sha_bytes:=hashlib.sha256(np.asarray(im).tobytes()).hexdigest() for im in tiles]
        prompt=f'Return JSON with collision_frame. Allowed frames: {ids}.'
        inputs,diag=helper.prepare_native_inputs(vlm,tiles,prompt)
        assert diag['metadata_before_processor']['frames_indices']==list(range(K))
        assert diag['metadata_before_processor']['total_num_frames']==K and diag['metadata_before_processor']['fps']==2.0
        assert diag['encoded_pixels_including_padding']<=1200000 and diag['original_candidate_count']==K
        assert diag['input_tile_RGB_sha256']==before and before==[hashlib.sha256(np.asarray(im).tobytes()).hexdigest() for im in tiles]
        # Each processor patch row has C,T,16,16 columns. Last temporal pair must
        # be identical only when odd K was padded, while even K preserves distinct frames.
        grid=diag['video_grid_thw'][0];patches=inputs['pixel_values_videos'].reshape(-1,3,2,16,16)
        last=patches[-grid[1]*grid[2]:];last_pair_equal=bool(torch.equal(last[:,:,0],last[:,:,1]))
        assert last_pair_equal==(K%2==1)
        diag.update(fake_irregular_original_frame_ids=ids,last_encoded_temporal_pair_RGB_normalized_exact_equal=last_pair_equal,
            input_tiles_unmodified=True,model_invoked=False,GPU_invoked=False)
        reports.append(diag);print(json.dumps({'K':K,'grid':grid,'tokens':diag['input_tokens'],'pixels':diag['encoded_pixels_including_padding'],'padding':diag['temporal_padding_frames']}),flush=True)
    rejects=[]
    for label,tiles in [('empty',[]),('wrong_size',[Image.new('RGB',(320,256))]),('too_many',[Image.new('RGB',(384,256))]*19)]:
        try:helper.prepare_native_inputs(vlm,tiles,'Synthetic contract prompt')
        except ValueError:rejects.append(label)
        else:raise AssertionError(label+' accepted')
    assert all(sha(p)==h for p,h in bindings.items())
    put('report.json',{'status':'passed','cases':reports,'rejection_checks':rejects,'all_bindings_unchanged':True,
        'elapsed_seconds':time.perf_counter()-start,'actual_saved_processor':str(MODEL),'model_or_weights_loaded':False,'GPU_used':False,
        'scope':'CPU tensor/processor contracts only. No accuracy or GPU memory/runtime proof.'})

if __name__=='__main__':main()
