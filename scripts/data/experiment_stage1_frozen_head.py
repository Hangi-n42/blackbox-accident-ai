"""Single strongly regularized linear-head pilot. Frozen V7 visual, no tuning."""
import os
os.environ.setdefault('OMP_NUM_THREADS','2');os.environ.setdefault('OPENBLAS_NUM_THREADS','2')
import argparse,collections,json,time,hashlib,sys,importlib.util,warnings
from pathlib import Path
import cv2,numpy as np,torch
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.exceptions import ConvergenceWarning
from scipy.special import expit
from experiment_stage1_temporal23 import ROOT,OUT as DATA,write,digest
from evaluate_stage1_road_pairs import metrics,PRIMARY
from diagnose_uhdm_full_robustness import transform
OUT=ROOT/'artifacts/stage1_head_pilot_20260919'
OLD=ROOT/'artifacts/stage1_road_pairs_20260918'
RELEASE=ROOT/'releases/v7/source'
def init():
 OUT.mkdir(exist_ok=False)
 assert (DATA/'tcl_integrity.json').exists()
 pro={'candidate':'L2 logistic binary head on frozen V7 normalized visual features, no backbone or LoRA updates',
 'train_sources':json.load(open(OLD/'selection.json'))['selection']['development'],'development_sources':json.load(open(OLD/'selection.json'))['selection']['holdout'],
 'confirmation':'five new REDS val sources, iPhone and TCL separately. Source IDs never included in features. Same-region overlap caveat remains.',
 'C':1.0,'solver':'lbfgs','max_iter':2000,'tol':1e-8,'random_state':20260919,'frame_sample_weight':'1/(12 frames * 2 views * 3 conditions), total one weight per source-class clip',
 'input':'exact V7 12 full-frame original+hflip views, per-frame probabilities arithmetic mean, threshold 0.5',
 'modes':PRIMARY,'augmentations':'Use existing matching original/recapture base, JPEG75, blur1 conditions in training; no added candidate TTA beyond V7 hflip',
 'gate':'Strictly higher primary development Macro-F1 AND original FP no larger than frozen V7; otherwise stop without confirmation scoring. No coefficient/threshold/hyperparameter search.',
 'scope':'exploratory source-separated, device-reserved small-data pilot; not proof sufficient for target generalization','setup_before_new_source_scores':True,
 'frozen_files':{str(p.relative_to(ROOT)):digest(p) for p in [Path(__file__),RELEASE/'model/stage1/tpo/merged_visual.pt',RELEASE/'model/stage1/tpo/clip_source/clip/model.py',RELEASE/'model/stage2/code/solution/stage1_tpo_merged.py',DATA/'content_qa.json',DATA/'tcl_integrity.json']}}
 write(OUT/'protocol.json',pro)
def extract(split):
 pro=json.load(open(OUT/'protocol.json'))
 for p,h in pro['frozen_files'].items():assert digest(ROOT/p)==h,p
 if split in ['new_val','new_tcl']:assert json.load(open(OUT/'development_summary.json'))['gate_pass']
 cache=OUT/f'{split}_features.npz';assert not cache.exists()
 if split in ['train','development']:
  allrows=json.load(open(OLD/'prepared_frames.json'));oldsplit='development' if split=='train' else 'holdout';rows=[r for r in allrows if r['split']==oldsplit]
 else:
  allrows=json.load(open(DATA/('tcl_prepared_frames.json' if split=='new_tcl' else 'prepared_frames.json')));rows=[r for r in allrows if r['split']==split and r['baseline']]
 groups=collections.defaultdict(list)
 for r in rows:groups[r['source_group'],r['label']].append(r)
 spec=importlib.util.spec_from_file_location('head_pilot_tpo',RELEASE/'model/stage2/code/solution/stage1_tpo_merged.py');mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
 torch.set_num_threads(2);cv2.setNumThreads(2);m=mod.TPODetector(RELEASE/'model/stage1/tpo',device='cpu');X=[];Y=[];metadata=[];baselines=[];start=time.perf_counter()
 try:
  for (source,label),rs in sorted(groups.items()):
   rs.sort(key=lambda r:r.get('time',r.get('frame')));assert len(rs)==12
   images=[]
   for r in rs:assert digest(ROOT/r['path'])==r['sha256'];images.append(np.asarray(Image.open(ROOT/r['path']).convert('RGB')))
   for mode in PRIMARY:
    aa=[transform(a,mode) for a in images];aa=aa+[np.ascontiguousarray(a[:,::-1,:]) for a in aa];ff=[];ss=[]
    for i in range(0,24,8):
     batch=[]
     for a in aa[i:i+8]:
      im=cv2.resize(a,(m.size,m.size),interpolation=cv2.INTER_LINEAR).astype(np.float32)/255;batch.append(np.transpose((im-m.mean)/m.std,(2,0,1)))
     with torch.inference_mode():
      f=torch.nn.functional.normalize(m.visual(torch.from_numpy(np.stack(batch))).float(),dim=-1);s=m.head(f).softmax(1)[:,0]
     ff.append(f.numpy());ss.extend(s.tolist())
    X.append(np.concatenate(ff));Y.extend([int(label=='recapture')]*24);baselines.append(float(np.mean(ss)));metadata.append({'source_group':source,'source':rs[0]['source'],'label':label,'mode':mode,'offset':(len(X)-1)*24,'count':24})
   print('features',split,source,label,flush=True)
 finally:m.close()
 x=np.concatenate(X);y=np.asarray(Y);assert np.isfinite(x).all() and len(x)==len(y)==len(metadata)*24
 np.savez_compressed(cache,x=x,y=y,baseline=np.asarray(baselines));write(OUT/f'{split}_records.json',metadata)
 maxdiff=0
 if split in ['train','development']:
  prior={(r['source'],r['label'],r['mode']):r['candidate'] for r in json.load(open(ROOT/f'artifacts/stage1_tpo_hflip_20260918/{oldsplit}/predictions.json'))}
  maxdiff=max(abs(b-prior[r['source'],r['label'],r['mode']]) for r,b in zip(metadata,baselines));assert maxdiff<1e-6
 write(OUT/f'{split}_feature_validation.json',{'frames':len(x),'records':len(metadata),'features':x.shape[1],'seconds':time.perf_counter()-start,'max_v7_difference':maxdiff,'all_finite':True})
def evaluate(split):
 pro=json.load(open(OUT/'protocol.json'));z=np.load(OUT/f'{split}_features.npz');coef=np.load(OUT/'head.npz');rs=json.load(open(OUT/f'{split}_records.json'))
 pp=expit(z['x']@coef['coef'].reshape(-1)+float(coef['intercept'][0]));pred=[]
 for r,b in zip(rs,z['baseline']):pred.append(r|{'baseline':float(b),'candidate':float(pp[r['offset']:r['offset']+24].mean())})
 report={'split':split,'conditions':{}}
 for mode in ['primary_all',*PRIMARY]:
  rr=pred if mode=='primary_all' else [r for r in pred if r['mode']==mode]
  report['conditions'][mode]={m:metrics([dict(label=r['label'],tpo=r[m],forensic=0) for r in rr],0,.5) for m in ['baseline','candidate']}
 b=report['conditions']['primary_all']['baseline'];c=report['conditions']['primary_all']['candidate'];report['gate_pass']=c['macro_f1']>b['macro_f1'] and c['fp']<=b['fp']
 report['files_unchanged']=all(digest(ROOT/p)==h for p,h in pro['frozen_files'].items());assert report['files_unchanged']
 write(OUT/f'{split}_predictions.json',pred);write(OUT/f'{split}_summary.json',report);print(json.dumps(report,indent=2),flush=True)
def train():
 pro=json.load(open(OUT/'protocol.json'));z=np.load(OUT/'train_features.npz');start=time.perf_counter()
 m=LogisticRegression(C=pro['C'],solver=pro['solver'],max_iter=pro['max_iter'],tol=pro['tol'],random_state=pro['random_state'])
 with warnings.catch_warnings():
  warnings.simplefilter('error',ConvergenceWarning);m.fit(z['x'],z['y'],sample_weight=np.full(len(z['y']),1/72))
 assert m.classes_.tolist()==[0,1]
 np.savez(OUT/'head.npz',coef=m.coef_,intercept=m.intercept_,classes=m.classes_)
 exported=expit(z['x']@m.coef_.reshape(-1)+m.intercept_[0]);assert np.max(np.abs(exported-m.predict_proba(z['x'])[:,1]))<1e-12
 write(OUT/'training.json',{'seconds':time.perf_counter()-start,'iterations':m.n_iter_.tolist(),'frame_records':len(z['y']),'independent_train_sources':len(pro['train_sources']),'total_sample_weight':float(len(z['y'])/72),'parameters':int(m.coef_.size+m.intercept_.size),'train_source_ids':pro['train_sources'],'head_sha256':digest(OUT/'head.npz'),'candidate_has_been_fit':True,'export_probability_max_difference':float(np.max(np.abs(exported-m.predict_proba(z['x'])[:,1])))})
 evaluate('train')
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('action',choices=['init','extract_train','train','development','new_val','new_tcl']);a=p.parse_args().action
 if a=='init':init()
 elif a=='extract_train':extract('train')
 elif a=='train':train()
 else:extract(a);evaluate(a)
