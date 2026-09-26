"""Conditional unseen-DLC evaluation and actual Stage1 submission entry validation."""
import os
os.environ['OMP_NUM_THREADS']='2';os.environ['OPENBLAS_NUM_THREADS']='2'
os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
import argparse,json,sys,time,socket,shutil,importlib.util
from pathlib import Path
import numpy as np,torch,cv2
from scipy.special import expit
from sklearn.metrics import confusion_matrix,f1_score
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'scripts/data'))
import experiment_stage1_anchored_head as helper
import verify_stage1_strong_anchor as frozen
NAME='anchored_0.01';REL=helper.REL

def extract():
 frozen.check_hashes();assert helper.read(OUT/'decision.json')['gate_pass']
 helper.OUT=OUT;helper.extract('dlc','confirmation')

def evaluate():
 frozen.check_hashes();assert not (OUT/'confirmation_decision.json').exists()
 assert helper.read(OUT/'decision.json')['gate_pass']
 helper.OUT=OUT
 x,rs=helper.dataset('dlc','confirmation')
 assert x.shape==(24*3*24,512) and len(rs)==72
 assert len({r['group'] for r in rs})==6 and len({r['cluster'] for r in rs})==3
 report=helper.evaluate('dlc','confirmation',[NAME]);rows=helper.read(OUT/'dlc_confirmation_predictions.json')
 y=np.array([r['label']=='recapture' for r in rows]);by={}
 for name in ['v7',NAME]:
  p=np.array([r['scores'][name]>=.5 for r in rows]);tn,fp,fn,tp=confusion_matrix(y,p,labels=[False,True]).ravel();m=report[name]['overall']
  assert (fp,fn)==(m['fp'],m['fn']) and abs(f1_score(y,p,average='macro')-m['macro_f1'])<1e-12
  for key in ['group','cluster','camera']:
   by.setdefault(key,{})[name]={g:helper.metric([r for r in rows if r[key]==g],[r['scores'][name] for r in rows if r[key]==g]) for g in sorted({r[key] for r in rows})}
 b,c=report['v7']['overall'],report[NAME]['overall'];passed=c['fp']<=b['fp'] and c['fn']<=b['fn'] and c['macro_f1']>b['macro_f1']
 rng=np.random.default_rng(20260919);ci={}
 for key in ['group','cluster']:
  groups=sorted({r[key] for r in rows});deltas=[]
  for _ in range(1000):
   sample=[r for g in rng.choice(groups,len(groups),replace=True) for r in rows if r[key]==g]
   deltas.append(helper.metric(sample,[r['scores'][NAME] for r in sample])['macro_f1']-helper.metric(sample,[r['scores']['v7'] for r in sample])['macro_f1'])
  ci[key]={'n_clusters':len(groups),'delta_f1_95_percentile':np.quantile(deltas,[.025,.975]).tolist(),'caveat':'Few source/type clusters, same collection process; not population proof.'}
 result={'gate_pass':passed,'metrics':report,'by_group_type_camera':by,'bootstrap':ci,'independent_metrics_verified':True,'scope':'Unseen document IDs and types, not independent road or capture-device distribution.','no_training_or_retuning':True}
 helper.write(OUT/'confirmation_decision.json',result);print(json.dumps(result,indent=2))

def runtime():
 assert helper.read(OUT/'confirmation_decision.json')['gate_pass']
 for r in helper.read(frozen.OLD/'public_provenance.json'):
  assert helper.sha(ROOT/'artifacts/public_eval/stage1/videos'/f"{r['ID']}.mp4")==r['sha256']
 frozen.check_hashes();assert not (OUT/'runtime_report.json').exists()
 dest=OUT/'candidate_runtime';(dest/'model/stage1/tpo').mkdir(parents=True,exist_ok=False)
 shutil.copyfile(REL/'inference.py',dest/'inference.py')
 shutil.copyfile(REL/'model/stage1/config.json',dest/'model/stage1/config.json')
 for name in ['stage2','stage3']:(dest/'model'/name).symlink_to(REL/'model'/name,target_is_directory=True)
 (dest/'model/stage1/tpo/clip_source').symlink_to(REL/'model/stage1/tpo/clip_source',target_is_directory=True)
 artifact=torch.load(REL/'model/stage1/tpo/merged_visual.pt',map_location='cpu',weights_only=True)
 theta=np.load(OUT/f'{NAME}.npz')['theta'];old_head={k:v.clone() for k,v in artifact['head'].items()}
 # Common two-class logit component preserved; difference replaced by the frozen binary head.
 w=torch.from_numpy(theta[:-1].astype(np.float32));bias=torch.tensor(theta[-1],dtype=torch.float32)
 mw=old_head['weight'].mean(0);mb=old_head['bias'].mean()
 artifact['head']={'weight':torch.stack([mw+w/2,mw-w/2]),'bias':torch.stack([mb+bias/2,mb-bias/2])}
 target=dest/'model/stage1/tpo/merged_visual.pt';torch.save(artifact,target)
 reloaded=torch.load(target,map_location='cpu',weights_only=True)
 assert all(torch.equal(v,reloaded['visual'][k]) for k,v in artifact['visual'].items())
 helper.write(OUT/'export.json',{'weights_sha256':helper.sha(target),'binary_head_sha256':helper.sha(OUT/f'{NAME}.npz'),'visual_tensors_unchanged':True,'head_class_order':artifact['class_order'],'scope':'Local runtime validation fixture; stage2/3/code links point to immutable V7. Not a packaged submission.'})
 (dest/'NOTICE.md').write_text('Local Stage1 validation fixture only. Based on V7 TPO/CLIP with frozen strong anchored head. Original licenses and provenance remain in releases/v7/source/model/stage1/tpo and the previous experiment DATA_SOURCES.md. Stage2/3 are read-only references to V7. No submission ZIP created.\n')
 del artifact,reloaded
 # Confirm float32 two-class export against fixed binary scores on already scored features.
 helper.OUT=OUT;x,rs=helper.dataset('dlc','confirmation');xx=torch.from_numpy(x.astype(np.float32))
 exp=expit(x@theta[:-1]+theta[-1])
 exported=(xx@torch.stack([mw+w/2,mw-w/2]).T+torch.stack([mb+bias/2,mb-bias/2])).softmax(1)[:,0].numpy()
 export_error=float(np.max(abs(exp-exported)));assert export_error<1e-6
 torch.set_num_threads(2);cv2.setNumThreads(2);sys.dont_write_bytecode=True
 net=[]
 def block(*args,**kwargs):net.append('attempt');raise RuntimeError('offline network guard')
 socket.socket.connect=block;socket.socket.connect_ex=block;socket.create_connection=block
 # Diagnostic helper imports may have loaded an older solution package; isolate the real V7 entry.
 for name in list(sys.modules):
  if name=='solution' or name.startswith('solution.'):del sys.modules[name]
 module=helper.load('strong_submitted_inference',dest/'inference.py');module._runtime(dest/'model/stage1')
 from solution.stage1_tpo_merged import TPODetector
 score=TPODetector.score;calls=[]
 def tracked(self,frames,*args,**kwargs):
  assert len(frames)==24 and all(np.array_equal(b,a[:,::-1,:]) for a,b in zip(frames[:12],frames[12:]))
  scores=score(self,frames,*args,**kwargs);calls.append(float(np.mean(scores)));return scores
 TPODetector.score=tracked
 expected_public={r['video']:r for r in helper.read(OUT/'public_guard_predictions.json')}
 timings=[];checks=[]
 def run(label,model_dir,data,expected):
  calls.clear();start=time.perf_counter();df=module.predict_stage1(data,model_dir);elapsed=time.perf_counter()-start
  assert list(df.columns)==['ID','answer'] and not df.isna().any().any() and df.ID.is_unique
  ids=sorted(expected);assert df.ID.tolist()==ids and len(calls)==len(ids)
  delta=[]
  for row,p in zip(df.itertuples(),calls):
   want=expected[row.ID];assert row.answer==('RERECORDED' if want>=.5 else 'ORIGINAL');delta.append(abs(p-want))
  assert max(delta)<1e-6
  df.to_csv(OUT/f'{label}_output.csv',index=False)
  rec={'run':label,'videos':len(df),'seconds_including_model_load_decode_inference':elapsed,'max_probability_difference':max(delta),'missing_videos':0};checks.append(rec);print('ENTRY',rec,flush=True)
  return elapsed
 data=ROOT/'artifacts/public_eval/stage1'
 # Paired execution order reverses between repeats; same real entrypoint, full model reload each call.
 for i,order in enumerate([['v7',NAME],[NAME,'v7']]):
  for name in order:
   md=REL/'model/stage1' if name=='v7' else dest/'model/stage1'
   t=run(f'public_{i}_{name}',md,data,{k:r['scores'][name] for k,r in expected_public.items()});timings.append({'method':name,'seconds':t})
 # Existing lossless32 AVI fixtures exercise the actual decoder on pixel-identical paired source frames.
 proxy=helper.read(ROOT/'artifacts/v7_validation_20260918/proxy_manifest.json');expected={}
 for split,prefix in [('train','development'),('development','holdout')]:
  x,rs=helper.dataset('vd',split);fp=expit(x@theta[:-1]+theta[-1])
  for r in rs:
   if r['mode']=='full_frame':expected[f"{prefix}_{r['source']:03d}_{r['label']}"]=float(fp[r['offset']:r['offset']+24].mean())
 assert set(expected)=={r['ID'] for r in proxy}
 run('paired_proxy_candidate',dest/'model/stage1',ROOT/'artifacts/v7_validation_20260918/proxy/stage1',expected)
 assert not net
 for name,loaded in sys.modules.items():
  if name.startswith('solution') and getattr(loaded,'__file__',None):assert Path(loaded.__file__).resolve().is_relative_to(REL)
 frozen.check_hashes()
 report={'status':'PASS','network_attempts':len(net),'checks':checks,'public_timings':timings,'public_median_seconds':{name:float(np.median([r['seconds'] for r in timings if r['method']==name])) for name in ['v7',NAME]},'export_frame_probability_max_error':export_error,'original_v7_and_prior_results_unchanged':True,'scope':'Mac CPU2threads, Stage1 actual entry only. Stage2/3 unchanged by hash; no CUDA full-contest60minute assertion.'}
 helper.write(OUT/'runtime_report.json',report);print(json.dumps(report,indent=2),flush=True)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('action',choices=['extract','evaluate','runtime']);a=p.parse_args();globals()[a.action]()
