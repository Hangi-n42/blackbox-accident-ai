"""Four-call diagnostic, not a new candidate or accuracy evaluation."""
import os,sys,json,hashlib,socket,time
from pathlib import Path
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='2'
os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).parent/'visual_input_diagnostic'
def put(p,v):
    with p.open('x',encoding='utf8') as f:json.dump(v,f,ensure_ascii=False,indent=2)
def deny(*a,**k):raise RuntimeError('Network forbidden')
def main():
    assert sys.flags.isolated and not OUT.exists()
    OUT.mkdir()
    put(OUT/'plan.json',dict(scope='One exposed development source, 4 fixed diagnostic calls, no accuracy or parameter selection',
        conditions=['single real frame description','same-size black image description','20-frame real sheet description','same real sheet with simple V6 coarse prompt'],max_tokens=[64,64,64,64]))
    socket.socket.connect=deny;socket.create_connection=deny
    sys.path.insert(0,str(ROOT/'artifacts/submissions/verify_v6/model/stage2/code'))
    from solution.vlm_candidate import CandidateVLM
    from PIL import Image
    source=ROOT/'research/v6_review_tool/dist/cases/NEXAR_REVIEW_00000/frame_000584.png'
    sheet=ROOT/'research/v7/stage2_dev_run1/00000/call_1/input_0.png'
    full=Image.open(source).convert('RGB');montage=Image.open(sheet).convert('RGB')
    prompt='Describe the vehicles and road scene visible in this image. State if the image is blank. Do not invent objects. Be concise.'
    conditions=[('single',full,prompt),('blank',Image.new('RGB',full.size),prompt),('sheet',montage,prompt),
        ('simple_event',montage,'These are chronological dashcam frames. Which frame best shows the first collision with the camera car? Return JSON with collision_frame and entry_side (LEFT or RIGHT) only. Choose an offered frame number.')]
    rows=[]
    with CandidateVLM(ROOT/'artifacts/submissions/verify_v6/model/stage2/vlm',precision='nf4') as model:
        model.torch.set_num_threads(2)
        for name,image,p in conditions:
            t=time.perf_counter();raw=model.ask([image],p,max_new_tokens=64)
            row=dict(condition=name,prompt=p,raw=raw,seconds=time.perf_counter()-t,size=image.size,rgb_sha256=hashlib.sha256(image.tobytes()).hexdigest())
            rows.append(row);put(OUT/(name+'.json'),row);print(json.dumps(row),flush=True)
    put(OUT/'report.json',dict(status='complete',rows=rows,limitation='Description comparison is an input-sensitivity diagnostic, not ground truth or submission evidence'))
if __name__=='__main__':main()
