"""Read-only production comparison; no labels, fitting, or production writes."""
import hashlib
import json
import platform
import time
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import cv2
import numpy as np
from threadpoolctl import threadpool_limits
from solution import stage3 as old, stage3_fast as fast

OUT=ROOT/'research/stage3_fast_benchmark'
OUT.mkdir(exist_ok=True)

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def create_stress():
    target=OUT/'external_stress/videos/comma_repeat5.mp4'
    target.parent.mkdir(parents=True,exist_ok=True)
    source=next((ROOT/'external_data/comma2k19/Chunk_1').rglob('video.hevc'))
    if not target.exists():
        cap=cv2.VideoCapture(str(source))
        size=(int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        cap.release()
        writer=cv2.VideoWriter(str(target),cv2.VideoWriter_fourcc(*'mp4v'),10.,size)
        assert writer.isOpened()
        count=0
        for repeat in range(5):
            cap=cv2.VideoCapture(str(source));index=0
            while True:
                ok,frame=cap.read()
                if not ok:break
                if index%2==0:writer.write(frame);count+=1
                index+=1
            cap.release()
        writer.release()
        print('stress created',count,flush=True)
    return target,source

def equal_features(a,b):
    return {'shape':list(a.shape),'bit_equal':bool(np.array_equal(a.view(np.uint32),b.view(np.uint32))),
            'unequal_values':int(np.count_nonzero(a.view(np.uint32)!=b.view(np.uint32))),
            'max_absolute_error':float(np.max(np.abs(a.astype(np.float64)-b.astype(np.float64))))}

def same_flow_tests(video):
    rng=np.random.default_rng(42)
    flows=[]
    for h in (96,192,256,341):
        flows.extend([np.zeros((h,256,2),np.float32),np.ones((h,256,2),np.float32),
                      rng.normal(size=(h,256,2)).astype(np.float32),
                      rng.normal(size=(h,256,2)).astype(np.float32)*1000])
    cap=cv2.VideoCapture(str(video));dis=cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_FAST);prev=None
    for index in range(120):
        ok,frame=cap.read()
        if not ok:break
        gray=cv2.cvtColor(cv2.resize(frame,(256,max(96,round(frame.shape[0]*256/frame.shape[1])))),cv2.COLOR_BGR2GRAY)
        if prev is not None and index%4==0:flows.append(dis.calc(prev,gray,None)*10.)
        prev=gray
    cap.release()
    cache=fast.FrameFeatureComputer()
    a=np.array([old._frame_feature(f) for f in flows]);b=np.array([cache(f) for f in flows])
    result=equal_features(a,b);result['synthetic_cases']=16;result['real_flow_cases']=len(flows)-16
    timings={}
    # One warmed, identical-input feature microbenchmark per implementation.
    for name,fn in [('old',old._frame_feature),('fast',fast.FrameFeatureComputer())]:
        fn(flows[-1]);start=time.perf_counter()
        for _ in range(3):
            for f in flows:fn(f)
        timings[name]=time.perf_counter()-start
    result['seconds_for_three_passes']=timings
    return result

def run(module,data_dir):
    captured={};original=module.extract_motion
    def capture(path,*args,**kwargs):
        started=time.perf_counter();features=original(path,*args,**kwargs)
        captured[Path(path).name]={'features':features,'seconds':time.perf_counter()-started}
        return features
    module.extract_motion=capture
    try:
        start=time.perf_counter();prediction=module.predict_stage3(data_dir,ROOT/'model/stage3');elapsed=time.perf_counter()-start
    finally:module.extract_motion=original
    return prediction,captured,elapsed

def compare(name,data_dir,reverse=False):
    runs={}
    for label,module in ([('fast',fast),('old',old)] if reverse else [('old',old),('fast',fast)]):
        print('running',name,label,flush=True)
        runs[label]=run(module,data_dir)
        print('completed',name,label,runs[label][2],flush=True)
    a,fa,ta=runs['old'];b,fb,tb=runs['fast']
    a.to_csv(OUT/f'{name}_old.csv',index=False);b.to_csv(OUT/f'{name}_fast.csv',index=False)
    comparisons={filename:{**equal_features(fa[filename]['features'],fb[filename]['features']),
                         'old_extract_seconds':fa[filename]['seconds'],'fast_extract_seconds':fb[filename]['seconds']} for filename in fa}
    return {'rows':len(a),'dataframe_equal':a.equals(b),'accel_changed':int((a.accel_label!=b.accel_label).sum()),
            'steer_changed':int((a.steer_label!=b.steer_label).sum()),'old_seconds':ta,'fast_seconds':tb,
            'speedup':ta/tb,'time_reduction_percent':100*(1-tb/ta),'features':comparisons,
            'execution_order':['fast','old'] if reverse else ['old','fast']}

def main():
    paths=[ROOT/'solution/stage3.py',ROOT/'model/stage3/motion_model.joblib']
    before={str(p.relative_to(ROOT)):sha(p) for p in paths}
    cv2.setNumThreads(2)
    started=time.perf_counter()
    with threadpool_limits(limits=2):
        numerical=same_flow_tests(ROOT/'artifacts/public_eval_10hz/stage3/videos/OPEN_001.mp4')
        print('same flow',numerical,flush=True)
        assert numerical['bit_equal']
        stress,source=create_stress()
        report={'numpy':np.__version__,'opencv':cv2.__version__,'platform':platform.platform(),
                'threads':2,'same_flow':numerical,'production_sha256_before':before,
                'external_source':str(source.relative_to(ROOT)),
                'external_stress_description':'One 60-second external 20Hz clip sampled every second frame and repeated five times as 10Hz MP4. Synthetic runtime stress only; no independence or accuracy claim.',
                'public':compare('public',ROOT/'artifacts/public_eval_10hz/stage3'),
                'external':compare('external',stress.parent.parent,reverse=True)}
    report['production_sha256_after']={str(p.relative_to(ROOT)):sha(p) for p in paths}
    assert before==report['production_sha256_after']
    report['total_seconds']=time.perf_counter()-started
    (OUT/'report.json').write_text(json.dumps(report,indent=2),encoding='utf8')
    print(json.dumps(report,indent=2),flush=True)

if __name__=='__main__':main()
