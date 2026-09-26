"""Frozen fitted head on pre-existing authentic road originals and public fixtures."""
import sys,importlib.util,json,time,hashlib,socket
from pathlib import Path
import numpy as np,cv2,torch
from scipy.special import expit
ROOT=Path.cwd();OUT=ROOT/'artifacts/stage1_head_pilot_20260919';REL=ROOT/'releases/v7/source'
def load(name,p):
 s=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
helper=load('comma_guard_helper',ROOT/'research/stage1/comma_original_diagnostic.py')
tpo=load('guard_v7_tpo',REL/'model/stage2/code/solution/stage1_tpo_merged.py')
# The prior helper is imported only for its sequential decoder, not its models/main.
sys.path.insert(0,str(REL/'model/stage2/code'));sampler=load('solution.guard_sampler',REL/'model/stage2/code/solution/stage1_v4.py')
net=[]
def block(*a,**k):net.append('network_attempt');raise RuntimeError('offline')
socket.socket.connect=block;socket.create_connection=block
cv2.setNumThreads(2);torch.set_num_threads(2);model=tpo.TPODetector(REL/'model/stage1/tpo',device='cpu');co=np.load(OUT/'head.npz');h=helper.sha(OUT/'head.npz')
def score(frames):
 aa=frames+[np.ascontiguousarray(a[:,::-1,:]) for a in frames];old=[];new=[];start=time.perf_counter()
 for i in range(0,len(aa),8):
  batch=[]
  for a in aa[i:i+8]:
   im=cv2.resize(a,(model.size,model.size),interpolation=cv2.INTER_LINEAR).astype(np.float32)/255;batch.append(np.transpose((im-model.mean)/model.std,(2,0,1)))
  with torch.inference_mode():
   f=torch.nn.functional.normalize(model.visual(torch.from_numpy(np.stack(batch))).float(),dim=-1);s=model.head(f).softmax(1)[:,0]
  old.extend(s.tolist());new.extend(expit(f.numpy()@co['coef'].reshape(-1)+float(co['intercept'][0])).tolist())
 return float(np.mean(old)),float(np.mean(new)),time.perf_counter()-start
sources=json.load(open(ROOT/'research/stage1/comma_original_diagnostic/sources.json'))['videos'];rows=[];start=time.perf_counter()
try:
 for r in sources:
  path=ROOT/r['video_path'].replace('\\','/');times=ROOT/r['local'].replace('\\','/')/'global_pose/frame_times'
  assert helper.sha(path)==r['video_sha256'];assert helper.sha(times)==r['frame_times_sha256']
  frames,meta=helper.sequential_frames(path,len(np.load(times).ravel()));b,c,t=score(frames);rows.append({'segment':r['segment'],'baseline':b,'candidate':c,'sampling':meta,'forward_seconds':t})
  (OUT/'comma_guard_predictions.json').write_text(json.dumps(rows,indent=2));print(len(rows),'of',len(sources),'FP',b>=.5,c>=.5,flush=True)
 summary={'n_original':len(rows),'baseline_fp':sum(r['baseline']>=.5 for r in rows),'candidate_fp':sum(r['candidate']>=.5 for r in rows),'new_fp':sum(r['baseline']<.5 and r['candidate']>=.5 for r in rows),'fixed_fp':sum(r['baseline']>=.5 and r['candidate']<.5 for r in rows),'seconds_including_decode':time.perf_counter()-start,'scope':'23 pre-existing comma2k19 authentic road originals; original-only FPR, NOT Macro-F1 or unseen overall competition validation; no threshold/head changes'}
 (OUT/'comma_guard_summary.json').write_text(json.dumps(summary,indent=2));print(summary,flush=True)
 fixtures=[]
 for p in sorted((ROOT/'artifacts/public_eval/stage1/videos').glob('*.mp4')):
  frames=sampler.sample_video(p,12);b,c,t=score(frames);fixtures.append({'ID':p.stem,'label':'original' if '_O_' in p.stem else 'recapture','baseline':b,'candidate':c,'forward_seconds':t})
 (OUT/'public_guard_predictions.json').write_text(json.dumps(fixtures,indent=2))
 assert len(fixtures)==10 and helper.sha(OUT/'head.npz')==h and not net
 (OUT/'guard_integrity.json').write_text(json.dumps({'head_unchanged':True,'network_attempts':len(net),'public_videos':len(fixtures)},indent=2))
finally:model.close()
