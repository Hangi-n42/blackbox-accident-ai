"""Real extracted entrypoint, offline network guard, output completeness and timing."""
import os
for k in ['HF_HUB_OFFLINE','TRANSFORMERS_OFFLINE','HF_HUB_DISABLE_IMPLICIT_TOKEN']:os.environ[k]='1'
os.environ['PYTHONDONTWRITEBYTECODE']='1'
import argparse,hashlib,importlib.util,json,socket,sys,time
from pathlib import Path
sys.dont_write_bytecode=True
ROOT=Path('/Users/hyeongi/projects/blackbox-accident-ai')
def sha(p):
 with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('--version',choices=['v6','v7'],required=True);p.add_argument('--stage',type=int,required=True);p.add_argument('--data',type=Path,required=True);p.add_argument('--name',required=True);p.add_argument('--reference',type=Path);a=p.parse_args()
 package=ROOT/'artifacts/submissions/v8_20260920/extracted';out=ROOT/'artifacts/submissions/v8_20260920'/a.name;out.mkdir(exist_ok=False)
 report={'version':a.version,'stage':a.stage,'package':str(package),'data':str(a.data),'network_attempts':0,'status':'RUNNING'}
 def deny(*args,**kwargs):report['network_attempts']+=1;raise AssertionError('Unexpected network connection')
 socket.socket.connect=deny;socket.socket.connect_ex=deny;socket.create_connection=deny
 import cv2,numpy as np,pandas as pd,torch
 torch.set_num_threads(2);cv2.setNumThreads(2)
 spec=importlib.util.spec_from_file_location('submitted_inference',package/'inference.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
 files_before={str(p.relative_to(package)):sha(p) for p in package.rglob('*') if p.is_file()}
 score_calls=[]
 if a.stage==1 and a.version=='v7':
  module._runtime(package/'model/stage1')
  from solution.stage1_tpo_merged import TPODetector
  original=TPODetector.score
  def tracked(self,frames,*args,**kwargs):
   n=len(frames)//2;assert n>0 and len(frames)==2*n
   assert all(np.array_equal(y,x[:,::-1,:]) for x,y in zip(frames[:n],frames[n:]))
   t=time.perf_counter();scores=original(self,frames,*args,**kwargs);elapsed=time.perf_counter()-t
   score_calls.append({'frames_per_view':n,'probability':.5*float(scores[:n].mean())+.5*float(scores[n:].mean()),'seconds':elapsed})
   return scores
  TPODetector.score=tracked
 started=time.perf_counter()
 try:
  result=getattr(module,f'predict_stage{a.stage}')(a.data,package/'model'/f'stage{a.stage}')
  report['seconds']=time.perf_counter()-started;assert isinstance(result,pd.DataFrame) and not result.isna().any().any()
  if a.stage==1:
   assert list(result.columns)==['ID','answer'] and result.ID.is_unique
   assert set(result.answer)<={'ORIGINAL','RERECORDED'}
   from solution.stage1 import VIDEO_EXTS
   videos=sorted(p for p in (a.data/'videos').iterdir() if p.suffix.lower() in VIDEO_EXTS)
   assert set(result.ID)=={p.stem for p in videos};assert len(result)==len(videos)
   if a.version=='v7':
    assert len(score_calls)==len(videos)
    for row,call,video in zip(result.itertuples(),score_calls,videos):
     assert row.ID==video.stem;assert row.answer==('RERECORDED' if call['probability']>=.5 else 'ORIGINAL');call['ID']=row.ID
   if a.reference:
    refs={r['ID']:r for r in json.loads(a.reference.read_text())};assert set(refs)==set(result.ID)
    differences=[]
    for row,call in zip(result.itertuples(),score_calls):
     assert row.answer==refs[row.ID]['expected_answer'];differences.append(abs(call['probability']-refs[row.ID]['expected_probability']))
    assert max(differences)<1e-6;report['experiment_prediction_match']=True;report['max_probability_difference']=max(differences)
  elif a.stage==3:
   assert list(result.columns)==['ID','sample_index','accel_label','steer_label']
   assert set(result.accel_label)<={'ACCELERATING','DECELERATING','CONSTANT','STOPPED'};assert set(result.steer_label)<={'LEFT','STRAIGHT','RIGHT'}
   assert not result.duplicated(['ID','sample_index']).any()
   videos=sorted((a.data/'videos').glob('*.mp4'));assert set(result.ID)=={p.stem for p in videos}
   for path in videos:
    cap=cv2.VideoCapture(str(path));count=0
    while cap.grab():count+=1
    cap.release();assert np.array_equal(result.loc[result.ID==path.stem,'sample_index'].to_numpy(),np.arange(count))
   if a.reference:
    expected=pd.read_csv(a.reference,dtype={'ID':str});pd.testing.assert_frame_equal(result.reset_index(drop=True),expected.reset_index(drop=True));report['reference_output_identical']=True
  result.to_csv(out/'output.csv',index=False);report.update(status='PASS',rows=len(result),score_calls=score_calls)
 except Exception as e:
  report.update(status='BLOCKED' if a.stage==2 and not torch.cuda.is_available() else 'FAIL',error_type=type(e).__name__,error=str(e),seconds=time.perf_counter()-started)
  if report['status']=='FAIL':raise
 finally:
  report['package_files_unchanged']=all(sha(package/n)==d for n,d in files_before.items());assert report['package_files_unchanged'];assert report['network_attempts']==0
  for name,loaded in sys.modules.items():
   if name.startswith('solution') and getattr(loaded,'__file__',None):assert Path(loaded.__file__).resolve().is_relative_to(package)
  (out/'report.json').write_text(json.dumps(report,indent=2));print(json.dumps({k:v for k,v in report.items() if k!='score_calls'},indent=2),flush=True)
if __name__=='__main__':main()
