"""CPU-only, independent sequential-reference audit of Stage1 sampling."""
from pathlib import Path
import sys,json,csv,time,hashlib
from unittest.mock import patch
import numpy as np
import cv2

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from solution import stage1
OUT=Path(__file__).resolve().parent


def digest(frame):return hashlib.sha256(frame.tobytes()).hexdigest()


def reference(path,count=12):
    cap=cv2.VideoCapture(str(path));n=0
    if not cap.isOpened():raise RuntimeError('reference open failed')
    while cap.grab():n+=1
    cap.release()
    if n==0:raise RuntimeError('reference empty')
    targets=np.unique(np.linspace(0,n-1,min(count,n)).round().astype(int)).tolist()
    cap=cv2.VideoCapture(str(path));frames=[];i=0
    while cap.grab():
        if i in targets:
            ok,frame=cap.retrieve()
            if not ok:raise RuntimeError('reference retrieve failed')
            frames.append(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))
        i+=1
    cap.release()
    assert i==n and len(frames)==len(targets)
    return n,targets,frames


def main():
    cv2.setNumThreads(2)
    rows=[]
    factory=cv2.VideoCapture
    class Traced:
        def __init__(self,path):self.cap=factory(path);self.read_log=[];trace.append(self)
        def __getattr__(self,k):return getattr(self.cap,k)
        def read(self):
            before=self.cap.get(cv2.CAP_PROP_POS_FRAMES);ok,f=self.cap.read()
            self.read_log.append({'reported_before':before,'reported_after':self.cap.get(cv2.CAP_PROP_POS_FRAMES),'ok':ok})
            return ok,f
    labels=list(csv.DictReader((ROOT/'Baseline/data/stage1/labels.csv').open()))
    for row in labels:
        path=ROOT/'Baseline/data/stage1'/row['path'];trace=[]
        before=time.perf_counter()
        with patch.object(stage1.cv2,'VideoCapture',Traced):old=stage1.sample_video(path)
        old_time=time.perf_counter()-before
        before=time.perf_counter();n,indices,expected=reference(path);ref_time=time.perf_counter()-before
        same=len(old)==len(expected) and all(np.array_equal(a,b) for a,b in zip(old,expected))
        result={'ID':row['ID'],'path':str(path.relative_to(ROOT)),'decoded_total':n,'expected_indices':indices,
                'old_count':len(old),'old_read_trace':[v for t in trace for v in t.read_log],
                'old_rgb_sha256':[digest(x) for x in old],'reference_rgb_sha256':[digest(x) for x in expected],
                'rgb_exact_match':same,'old_seconds':old_time,'reference_two_pass_seconds':ref_time}
        rows.append(result)
        (OUT/'public_decoder_audit.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')
        print(row['ID'],'frames',n,'match',same,'old',round(old_time,3),'reference',round(ref_time,3),flush=True)
    print('exact matches',sum(r['rgb_exact_match'] for r in rows),len(rows),flush=True)


if __name__=='__main__':main()
