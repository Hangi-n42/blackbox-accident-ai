"""Bounded source-only overlap screen; no annotations or model output read."""
import os, sys
sys.dont_write_bytecode = True
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'): os.environ[k]='2'
from pathlib import Path
import hashlib, json, itertools, datetime, time
import cv2
import numpy as np
ROOT=Path(__file__).resolve().parents[3]
OUT=Path(__file__).resolve().parent
cv2.setNumThreads(2)
def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()
def save(n,x): (OUT/n).write_text(json.dumps(x,indent=2),encoding='utf8')
def signature(im):
    g=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    low=cv2.dct(cv2.resize(g,(32,32),interpolation=cv2.INTER_AREA).astype(np.float32))[:8,:8].ravel()[1:]
    return low>np.median(low),cv2.resize(g,(64,64),interpolation=cv2.INTER_AREA).astype(np.float32)/255
def main():
    start=time.perf_counter()
    sources=[]
    for ID in ['00000','00003','00004','00005','00006','00007']:
        sources.append(dict(id='NEXAR_'+ID,group='dev' if ID in ['00000','00003','00004'] else 'reserved',path=ROOT/'research/v6/nexar_review_candidates'/f'{ID}.mp4'))
    for p in sorted((ROOT/'Baseline/data/stage2/videos').glob('*.mp4')):sources.append(dict(id='PUBLIC_'+p.stem,group='public_dev',path=p))
    for s in sources:s['sha256']=sha(s['path']);s['path']=str(s['path'])
    plan=dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),script_sha256=sha(__file__),sources=sources,
      samples_per_video=12,sampling='Rounded linspace from first to final metadata frame; sequential decode and exact count assertion',
      views=['full','center80percent'],phash_bits=63,near_frame_hamming_max=8,near_frame_gray_MAE_max=.08,
      candidate_video_rule='Any exact sampled full-frame match OR >=3 distinct frames on both sides with same-view near matches. All individual near matches also reported for review.',
      comparison='Every cross-video sampled-frame pair; no labels, predictions, event timing or external identity information',
      limitations=['Negative sparse matches do not prove different incidents.','Screening thresholds are heuristic flags, not calibrated statistical guarantees.','No pretrained-model independence certification.'])
    save('plan_frozen.json',plan)
    data={};meta=[]
    for s in sources:
        cap=cv2.VideoCapture(s['path']);n=int(cap.get(cv2.CAP_PROP_FRAME_COUNT));fps=cap.get(cv2.CAP_PROP_FPS)
        ix=np.unique(np.rint(np.linspace(0,n-1,12)).astype(int));chosen=set(ix.tolist());frames=[];i=0
        while True:
            ok,im=cap.read()
            if not ok:break
            if i in chosen:
                h,w=im.shape[:2];crop=im[int(.1*h):int(.9*h),int(.1*w):int(.9*w)]
                frames.append(dict(index=i,shape=list(im.shape),sha256=hashlib.sha256(im.tobytes()).hexdigest(),full=signature(im),center80percent=signature(crop),thumb=cv2.resize(im,(256,144),interpolation=cv2.INTER_AREA)))
            i+=1
        cap.release();assert i==n and len(frames)==len(ix),(s['id'],i,n)
        data[s['id']]=frames
        meta.append(dict(id=s['id'],group=s['group'],frames=n,fps=fps,samples=[dict(index=f['index'],shape=f['shape'],sha256=f['sha256']) for f in frames]))
        print(s['id'],n,'sampled',len(frames),flush=True)
    pairs=[]
    for a,b in itertools.combinations(sources,2):
        exact=[];near=[];mins={v:64 for v in ['full','center80percent']}
        for f,g in itertools.product(data[a['id']],data[b['id']]):
            if f['shape']==g['shape'] and f['sha256']==g['sha256']:exact.append([f['index'],g['index']])
            for v in mins:
                hp,gp=f[v],g[v];dist=int(np.count_nonzero(hp[0]!=gp[0]));mae=float(np.mean(np.abs(hp[1]-gp[1])))
                mins[v]=min(mins[v],dist)
                if dist<=8 and mae<=.08:near.append(dict(a=f['index'],b=g['index'],view=v,hamming=dist,gray_mae=mae))
        flagged=bool(exact) or any(len({x['a'] for x in near if x['view']==v})>=3 and len({x['b'] for x in near if x['view']==v})>=3 for v in mins)
        pairs.append(dict(a=a['id'],b=b['id'],groups=[a['group'],b['group']],same_file_sha=a['sha256']==b['sha256'],exact=exact,near=near,min_hamming=mins,candidate=flagged))
    # Compact evidence sheets: five evenly spread samples from the fixed 12, not event-selected.
    for sheet,ss in [('nexar',sources[:6]),('public',sources[6:])]:
        canvas=np.zeros((len(ss)*176,5*256,3),np.uint8)
        for r,s in enumerate(ss):
            fs=data[s['id']]
            for c,j in enumerate([0,3,6,8,11]):
                f=fs[min(j,len(fs)-1)];y=r*176;x=c*256
                canvas[y+32:y+176,x:x+256]=f['thumb']
                cv2.putText(canvas,f"{s['id']} idx {f['index']}",(x+3,y+21),cv2.FONT_HERSHEY_SIMPLEX,.43,(255,255,255),1,cv2.LINE_AA)
        ok,encoded=cv2.imencode('.jpg',canvas);assert ok
        (OUT/f'{sheet}_contactsheet.jpg').write_bytes(encoded.tobytes())
    assert sha(__file__)==plan['script_sha256'] and all(sha(s['path'])==s['sha256'] for s in sources)
    save('report.json',dict(status='sample_screen_complete_visual_review_pending',source_count=len(sources),pair_count=len(pairs),sample_count=sum(len(v) for v in data.values()),candidates=[p for p in pairs if p['candidate']],pairs=pairs,sources=meta,inputs_unchanged=True,seconds=time.perf_counter()-start))
    print(json.dumps(dict(pairs=len(pairs),flagged=sum(p['candidate'] for p in pairs),individual_near=sum(len(p['near']) for p in pairs))),flush=True)
if __name__=='__main__':main()
