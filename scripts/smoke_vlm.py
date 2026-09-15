from pathlib import Path
import sys,json,time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from PIL import Image
from solution.vlm import LocalVLM
import torch

if __name__=='__main__':
    root=Path(__file__).resolve().parents[1]
    start=time.perf_counter()
    with LocalVLM(root/'artifacts/model/stage2/vlm') as model:
        loaded=time.perf_counter()
        im=Image.open(root/'artifacts/data_audit/stage2_videos_000001.jpg')
        answer=model.ask([im], 'These are frames from one dashcam clip, ordered left to right, top to bottom. Which numbered frame first shows physical contact of the camera vehicle with the crossing dark car? Return JSON with collision_frame and entry_side (LEFT or RIGHT).',100)
        print(json.dumps({'load_seconds':loaded-start,'ask_seconds':time.perf_counter()-loaded,
                          'peak_gpu_gib':torch.cuda.max_memory_allocated()/2**30,'answer':answer}),flush=True)
