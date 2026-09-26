"""Public-video timing of exact V7 indices and additional interval midpoints."""
import sys,json,time,socket,hashlib
from pathlib import Path
import numpy as np,cv2,torch
ROOT=Path.cwd();OUT=ROOT/'artifacts/stage1_temporal23_20260919';RELEASE=ROOT/'releases/v7/source'
sys.path.insert(0,str(RELEASE/'model/stage2/code'))
from solution.stage1_v4 import sample_video_diagnostic
from solution.stage1_tpo_merged import TPODetector
network=[]
def block(*a,**k):network.append('attempt');raise RuntimeError('Network blocked')
socket.socket.connect=block;socket.create_connection=block
cv2.setNumThreads(2);torch.set_num_threads(2)
def nested(xs):return sorted(set(xs+[(a+b)//2 for a,b in zip(xs,xs[1:])]))
for total in [1,2,5,11,12,13,23,24,60,300,1000]:
 base=np.unique(np.linspace(0,total-1,min(12,total)).round().astype(int)).tolist();c=nested(base)
 assert set(base)<=set(c) and len(c)<=23 and min(c)>=0 and max(c)<total and len(c)==len(set(c))
assert len(nested(list(range(0,120,10))))==23
model=TPODetector(RELEASE/'model/stage1/tpo',device='cpu');model.score([np.zeros((360,640,3),np.uint8)]*46)
rows=[]
def infer(p,extra):
 start=time.perf_counter();frames,meta=sample_video_diagnostic(p,12)
 assert meta['method']=='seek_positions_checked' and meta['complete'],meta
 idx=meta['selected_indices'];allidx=nested(idx) if extra else idx
 if extra:
  cached=dict(zip(idx,frames));cap=cv2.VideoCapture(str(p))
  try:
   for i in allidx:
    if i not in cached:
     assert cap.set(cv2.CAP_PROP_POS_FRAMES,i);ok,im=cap.read();assert ok and im is not None
     assert abs(cap.get(cv2.CAP_PROP_POS_FRAMES)-(i+1))<=.5
     cached[i]=cv2.cvtColor(im,cv2.COLOR_BGR2RGB)
  finally:cap.release()
  frames=[cached[i] for i in allidx]
 decode=time.perf_counter()-start
 s=model.score(frames+[np.ascontiguousarray(f[:,::-1,:]) for f in frames]);n=len(frames);prob=.5*float(s[:n].mean())+.5*float(s[n:].mean())
 return {'probability':prob,'seconds':time.perf_counter()-start,'decode_seconds':decode,'indices':allidx,'answer':'RERECORDED' if prob>=.5 else 'ORIGINAL'}
try:
 for p in sorted((ROOT/'artifacts/public_eval/stage1/videos').iterdir()):
  if p.suffix.lower() not in ['.mp4','.avi','.mov','.mkv']:continue
  if len(rows)%2==0:b=infer(p,False);c=infer(p,True)
  else:c=infer(p,True);b=infer(p,False)
  rows.append({'ID':p.stem,'baseline':b,'candidate':c})
finally:model.close()
prior=json.load(open(ROOT/'artifacts/v7_validation_20260918/v7_stage1_public/report.json'));expected={r['ID']:r['probability'] for r in prior['score_calls']}
assert len(rows)==10 and max(abs(r['baseline']['probability']-expected[r['ID']]) for r in rows)<1e-6
assert not network
report={'scope':'10 public videos; pretrained/previously seen, not independent performance evaluation. CPU 2 threads; warmed model; includes MP4 decoding. Fail-fast diagnostic, not submission recovery implementation.', 'rows':rows,'network_attempts':len(network),'baseline_matches_actual_v7':True,'total_seconds':{m:sum(r[m]['seconds'] for r in rows) for m in ['baseline','candidate']},'decode_seconds':{m:sum(r[m]['decode_seconds'] for r in rows) for m in ['baseline','candidate']},'changed_decisions':[r['ID'] for r in rows if r['baseline']['answer']!=r['candidate']['answer']]}
(OUT/'runtime_probe.json').write_text(json.dumps(report,indent=2));print(json.dumps({k:v for k,v in report.items() if k!='rows'},indent=2))
