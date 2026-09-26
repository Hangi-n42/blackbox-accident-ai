"""Frozen data/model-recipe ablations: DIS features first, RAFT fixed864 second."""
from pathlib import Path
import sys,json,time,hashlib,warnings,argparse
import numpy as np,pandas as pd,cv2,av,joblib
from scipy.ndimage import uniform_filter1d
from sklearn.base import clone
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import f1_score,confusion_matrix
from threadpoolctl import threadpool_limits
O=Path(__file__).resolve().parent;R=O.parents[1];B=R/'artifacts/stage3_training_basis_20260917';D=R/'artifacts/stage3_mixed_datecheck_20260917'
sys.path.insert(0,str(R/'artifacts/submissions/verify_v6/model/stage2/code'))
from solution.stage3_v5_compatible import FrameFeatureComputer
cv2.setNumThreads(2)
read=lambda p:json.loads(p.read_text());write=lambda p,x:p.write_text(json.dumps(x,ensure_ascii=False,indent=2));sha=lambda p:hashlib.file_digest(p.open('rb'),'sha256').hexdigest()
VARIANTS={'dis_base':['base'],'dis_common':['base','common'],'dis_residual':['base','residual'],'dis_common_residual':['base','common','residual'],'dis_all':['base','common','residual','quality'],'raft_base':['base']}
def combine(raw,n):
 raw=np.asarray(raw,dtype=np.float32);parts=[raw]+[uniform_filter1d(raw,size=w,axis=0,mode='nearest') for w in (5,15,31)];ix=np.arange(len(raw));sm=parts[2]
 for lag in (5,15):parts.append((sm[np.minimum(ix+lag,len(ix)-1)]-sm[np.maximum(ix-lag,0)])/(2*lag/10))
 a=np.concatenate(parts,1);out=np.stack([np.interp(np.arange(n),ix+.5,a[:,j]) for j in range(a.shape[1])],1).astype(np.float32);assert np.isfinite(out).all();return out
class Geometry:
 def __init__(self,h,w):
  y,x=np.mgrid[:h,:w];self.h=h;self.w=w;self.xy=np.stack([np.ones_like(x),x/w-.5,y/h-.5],-1);self.mask=(y/h>=.15)&(y/h<.72)&(x/w>=.03)&(x/w<.97)&(y%4==0)&(x%4==0);self.A=self.xy[self.mask];self.ff=FrameFeatureComputer()
 def __call__(self,flow):
  f=flow/np.array([self.w,self.h]);b=f[self.mask];a=self.A;coef=np.linalg.lstsq(a,b,rcond=None)[0]
  # Fixed robust affine approximation, not calibrated physical ego-motion.
  for _ in range(8):
   err=np.linalg.norm(b-a@coef,axis=1);scale=max(float(np.median(err))*1.4826,1e-6);weight=np.minimum(1,1.345*scale/np.maximum(err,1e-12));s=np.sqrt(weight);coef=np.linalg.lstsq(a*s[:,None],b*s[:,None],rcond=None)[0]
  residual=(f-self.xy@coef)*np.array([self.w,self.h]);err=np.linalg.norm(b-a@coef,axis=1);base=np.linalg.norm(b,axis=1);q=np.array([np.median(err),np.quantile(err,.9),np.median(err)/(np.median(base)+1e-6),np.mean(weight>.5)],np.float32)
  return coef.ravel().astype(np.float32),self.ff(residual.astype(np.float32)),q

def cases():
 cs=read(B/'cases.json')
 for c in cs:c['public']=False
 for p in sorted((R/'artifacts/public_eval_10hz/stage3/videos').glob('*.mp4')):cs.append({'id':p.stem,'raw_path':str(p.relative_to(R)),'public':True})
 return cs

def frames(c):
 if c['public']:
  cap=cv2.VideoCapture(str(R/c['raw_path']))
  while True:
   ok,im=cap.read()
   if not ok:break
   h=max(96,round(im.shape[0]*256/im.shape[1]));yield cv2.cvtColor(cv2.resize(im,(256,h)),cv2.COLOR_BGR2GRAY)
  cap.release()
 else:
  d=np.load(B/c['labels_npz']);indices=set(map(int,d['frame_index']));last=max(indices)
  with av.open(str(R/c['raw_path'])) as con:
   for k,f in enumerate(con.decode(video=0)):
    if k in indices:
     im=f.to_ndarray(format='bgr24');h=max(96,round(im.shape[0]*256/im.shape[1]));yield cv2.cvtColor(cv2.resize(im,(256,h)),cv2.COLOR_BGR2GRAY)
    if k>=last:break

def cached(c):
 if c['public']:return np.load(R/'artifacts/data_pilot_20260916/comma_experiment/mac_public_features'/(c['id']+'.npy'))
 p=R/c['feature_cache'] if c['feature_cache'] else B/(c['id']+'_features.npy');a=np.load(p);return a['features'] if isinstance(a,np.lib.npyio.NpzFile) else a

def freeze():
 if (O/'freeze.json').exists():return
 inputs=[B/'cases.json',B/'expanded_rav4.joblib',R/'Baseline/data/stage3/labels.csv']
 for c in cases():
  if not c['public']:inputs.append(B/c['labels_npz'])
 for k in range(1,4):inputs.extend([D/f'fold_{k}'/x for x in ['training_selection.json','heldout_predictions.csv','mixed_budget_rav4.joblib']])
 production=read(R/'artifacts/pipeline_diagnosis_20260917/freeze.json')['files']
 write(O/'freeze.json',{'variants':VARIANTS,'fixed':'same grayscale width256 frames;10Hz sampling;144 ROI and temporal864 unchanged; identical saved mixed_budget_rav4 training rows, labels, classifier recipe and3date folds; no hyperparameter tuning','affine':'normalized coordinates upper/middle .15<=y<.72,.03<=x<.97 sampled every4pixels;8IRLS iterations Huber1.345;6coef;4quality;144residual;all same6 temporal blocks','raft':'Torchvision Raft_Small_Weights.C_T_V2,12updates,grayscale repeated3channels scaled[-1,1],edgepadding divisible8 if needed,no image resize difference,batch4,MPS','scope':'23 existing short comma clips +5 public clips only; development-exposed sensor proxy labels; no independent estimate, no pooled repeated-fold score; acceleration only','gates':'F1 improvement, no opposite count increase overall/pervehicle/eachdirection on every fold, publicF1 no decrease; no production adoption','inputs':{str(p.relative_to(R)):sha(p) for p in inputs},'production':production,'script_sha256':sha(Path(__file__))})

def extract(engine):
 out=O/engine;out.mkdir(exist_ok=True);torch=None
 if engine=='raft':
  import torch
  from torchvision.models.optical_flow import raft_small,Raft_Small_Weights
  torch.set_num_threads(2);device='mps' if torch.backends.mps.is_available() else 'cpu';torch.hub.set_dir(str(O/'weights'));model=raft_small(weights=Raft_Small_Weights.C_T_V2,progress=False).eval().to(device)
  def infer(a,b,dev=device):
   a=torch.from_numpy(np.stack(a)).unsqueeze(1).repeat(1,3,1,1).float().to(dev)/127.5-1;b=torch.from_numpy(np.stack(b)).unsqueeze(1).repeat(1,3,1,1).float().to(dev)/127.5-1;h,w=a.shape[-2:];pad=(0,(-w)%8,0,(-h)%8)
   a=torch.nn.functional.pad(a,pad,mode='replicate');b=torch.nn.functional.pad(b,pad,mode='replicate')
   with torch.inference_mode():pred=model(a,b,num_flow_updates=12)[-1][...,:h,:w].permute(0,2,3,1).cpu().numpy()*10
   assert np.isfinite(pred).all();return pred
  fs=list(frames(cases()[0]))[:3];t=time.perf_counter();mps=infer(fs[:2],fs[1:]);seconds=time.perf_counter()-t
  model.to('cpu');cpu=infer(fs[:2],fs[1:],dev='cpu');model.to(device)
  write(O/'raft_runtime_check.json',{'torch':torch.__version__,'device':device,'updates':12,'pairs':2,'first_inference_seconds':seconds,'cpu_mps_mean_abs_px_per_frame':float(np.mean(abs(cpu-mps))/10),'cpu_mps_max_abs_px_per_frame':float(np.max(abs(cpu-mps))/10),'weights':{p.name:sha(p) for p in (O/'weights/checkpoints').glob('*')},'note':'real frames, numerical comparison not equivalence guarantee; grayscale input control differs from standard RGB benchmark'})
 for c in cases():
  target=out/(c['id']+'.npz')
  if target.exists():continue
  t=time.perf_counter();ims=list(frames(c));n=len(ims);ff=FrameFeatureComputer();raw={'base':[]}
  if engine=='dis':
   geom=Geometry(*ims[0].shape);dis=cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_FAST);raw.update({k:[] for k in ['common','residual','quality']})
   for a,b in zip(ims,ims[1:]):
    flow=dis.calc(a,b,None)*10;raw['base'].append(ff(flow));vals=geom(flow)
    for key,val in zip(['common','residual','quality'],vals):raw[key].append(val)
  else:
   for i in range(0,n-1,4):
    fields=infer(ims[i:min(i+4,n-1)],ims[i+1:min(i+5,n)])
    for flow in fields:raw['base'].append(ff(flow))
  features={k:combine(v,n) for k,v in raw.items()};assert features['base'].shape==cached(c).shape
  if engine=='dis':assert np.array_equal(features['base'],cached(c)),(c['id'],np.max(abs(features['base']-cached(c))))
  np.savez_compressed(target,**features);write(out/(c['id']+'_timing.json'),{'id':c['id'],'frames':n,'pairs':n-1,'seconds':time.perf_counter()-t,'baseline_exact_match':engine=='dis'});print(engine,c['id'],n,round(time.perf_counter()-t,2),flush=True)

def metrics(y,p):
 y=np.asarray(y);p=np.asarray(p);cm=confusion_matrix(y,p,labels=range(4));den=int(np.isin(y,[0,1]).sum());opp=int(cm[0,1]+cm[1,0]);return {'n':len(y),'macro_f1':float(f1_score(y,p,labels=range(4),average='macro',zero_division=0)),'confusion':cm.tolist(),'opposite':opp,'moving_gt':den,'opposite_rate':opp/den if den else None,'accel_to_decel':int(cm[0,1]),'decel_to_accel':int(cm[1,0]),'constant_errors':int(cm[2].sum()-cm[2,2])}
def evaluate(engine):
 variants={k:v for k,v in VARIANTS.items() if k.startswith(engine+'_')};cs={c['id']:c for c in cases()};arrays={id:dict(np.load(O/engine/(id+'.npz'))) for id in cs};template=joblib.load(B/'expanded_rav4.joblib')['accel'];results=[];public=pd.read_csv(R/'Baseline/data/stage3/labels.csv').rename(columns={'ID':'id'});public['truth']=public.accel_label.map({n:i for i,n in enumerate(['ACCELERATING','DECELERATING','CONSTANT','STOPPED'])});public['vehicle']='public'
 for fold in range(1,4):
  orig=D/f'fold_{fold}';dst=O/f'fold_{fold}';dst.mkdir(exist_ok=True);sel=read(orig/'training_selection.json')['mixed_budget_rav4'];test=pd.read_csv(orig/'heldout_predictions.csv');basepub=pd.read_csv(orig/'public_predictions.csv');yy=np.array([r['truth'] for r in sel]);
  for name,parts in variants.items():
   x={id:np.concatenate([a[k] for k in parts],1) for id,a in arrays.items()};train=np.stack([x[r['id']][r['sample_index']] for r in sel]);m=clone(template);t=time.perf_counter()
   with warnings.catch_warnings(record=True) as ws,threadpool_limits(limits=2):
    warnings.simplefilter('always');m.fit(train,yy)
   assert not any(issubclass(w.category,ConvergenceWarning) for w in ws),name
   joblib.dump({'accel':m,'experimental_only':True},dst/(name+'.joblib'));out=[]
   for scope,rows in [('heldout',test),('public',public)]:
    z=rows[['id','sample_index','truth','vehicle']].copy();xx=np.stack([x[r.id][r.sample_index] for r in z.itertuples()]);p=m.predict(xx);prob=m.predict_proba(xx);z['prediction']=p
    for j in range(4):z[f'p_{j}']=prob[:,j]
    base=test if scope=='heldout' else basepub;assert list(zip(z.id,z.sample_index))==list(zip(base.id,base.sample_index));z['baseline']=base.mixed_budget_rav4.to_numpy()
    if name=='dis_base':assert np.array_equal(p,z.baseline),('baseline mismatch',fold,scope)
    z.to_csv(dst/(name+'_'+scope+'.csv'),index=False)
    for group,g in [(scope,z),*([(v,g) for v,g in z.groupby('vehicle')] if scope=='heldout' else [])]:
     r={'fold':fold,'variant':name,'scope':group,'features':train.shape[1],'train_n':len(sel),**metrics(g.truth,g.prediction)};old=((g.truth==0)&(g.baseline==1))|((g.truth==1)&(g.baseline==0));new=((g.truth==0)&(g.prediction==1))|((g.truth==1)&(g.prediction==0));r.update(new_opposites=int((new&~old).sum()),fixed_opposites=int((old&~new).sum()),new_from_correct=int((new&(g.baseline==g.truth)).sum()));results.append(r)
   print('fit',fold,name,train.shape,'seconds',round(time.perf_counter()-t,2),flush=True)
 pd.DataFrame(results).to_csv(O/(engine+'_metrics.csv'),index=False);write(O/(engine+'_metrics.json'),results)
 f=read(O/'freeze.json');assert all(sha(R/p)==h for p,h in {**f['inputs'],**f['production']}.items());write(O/(engine+'_checks.json'),{'input_and_production_hashes_unchanged':True,'baseline_prediction_exact_match':engine=='dis','completed':True})
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('mode',choices=['dis','raft']);a.add_argument('--evaluate-only',action='store_true');args=a.parse_args();freeze()
 if not args.evaluate_only:extract(args.mode)
 evaluate(args.mode)
