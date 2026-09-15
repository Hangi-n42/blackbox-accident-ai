"""One CPU profile of fixed features only; no classifiers or CAN labels."""
from pathlib import Path
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[k]='2'
import sys,json,time,cProfile,pstats,io,hashlib
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import cv2,numpy as np
from threadpoolctl import threadpool_limits
from solution.stage3_v5 import GlobalResidualComputer
from solution.stage3_fast import FrameFeatureComputer
OUT=ROOT/'research/v5_stage3/equivalent_optimization'

def get_flows(path,fps,n=12):
    cap=cv2.VideoCapture(str(path));dis=cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_FAST)
    result=[];prev=None;i=0;stride=round(fps/10)
    while len(result)<n:
        ok,bgr=cap.read()
        if not ok:break
        j=i;i+=1
        if j%stride:continue
        h=max(96,round(bgr.shape[0]*256/bgr.shape[1]))
        gray=cv2.cvtColor(cv2.resize(bgr,(256,h)),cv2.COLOR_BGR2GRAY)
        if prev is not None:result.append(dis.calc(prev,gray,None)*(fps/stride))
        prev=gray
    cap.release();return result

def main():
    OUT.mkdir(parents=True,exist_ok=False)
    manifests=[]
    for p in sorted((ROOT/'external_data/comma2k19').glob('*_manifest.json')):manifests.extend(json.loads(p.read_text()))
    civic=next(r for r in manifests if '99c94dc769b5d96e' in r['route'])
    plan={'timestamp_unix':time.time(),'scope':'Profile computation only, before optimized implementation and accuracy predictions',
        'public_file':'artifacts/public_eval_10hz/stage3/videos/OPEN_001.mp4','external_record':civic,
        'real_flows_per_source':12,'profile_repetitions':10,'cpu_threads':2,'gpu':False,
        'allowed_intervention':'One mathematically equivalent1128 implementation; fixed IRLS3/sample/ROI/temporal/feature dimensions unchanged.',
        'required_followup_gate':'All1128 float32 bit exact on synthetic/public5/real external; original local runtime <=1.25 versus productionfast, 3 alternating paired runs; no accuracy before passing.',
        'original_audit_sha256':hashlib.sha256((ROOT/'research/v5_stage3/audit_report.json').read_bytes()).hexdigest()}
    (OUT/'profile_plan_frozen.json').write_text(json.dumps(plan,indent=2),encoding='utf8')
    cv2.setNumThreads(2)
    flows=get_flows(ROOT/plan['public_file'],10)+get_flows(ROOT/civic['local']/'video.hevc',20)
    np.savez_compressed(OUT/'fixed_real_flows.npz',**{str(i):f for i,f in enumerate(flows)})
    outputs={}
    for name,cls in [('original_base144',FrameFeatureComputer),('original_extra44',GlobalResidualComputer)]:
        comp=cls();prof=cProfile.Profile();prof.enable()
        for _ in range(10):
            for f in flows:comp(f)
        prof.disable();prof.dump_stats(str(OUT/(name+'.prof')))
        s=io.StringIO();pstats.Stats(prof,stream=s).sort_stats('cumulative').print_stats(25)
        (OUT/(name+'.txt')).write_text(s.getvalue(),encoding='utf8')
        outputs[name]=s.getvalue()
    print(json.dumps(outputs,indent=2),flush=True)

if __name__=='__main__':
    with threadpool_limits(limits=2):main()
