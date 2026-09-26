"""Frozen 12 versus nested 23 time samples; no training or production edits."""
import argparse,collections,json,sys,time,hashlib,os
from pathlib import Path
import cv2,numpy as np
from PIL import Image,ImageDraw
import build_stage1_road_pairs as acq
from prepare_stage1_road_pairs import match,jpeg95,digest
ROOT=acq.ROOT; OLD=ROOT/'artifacts/stage1_road_pairs_20260918'; OUT=ROOT/'artifacts/stage1_temporal23_20260919'
BASE=[1,5,11,16,21,27,32,38,43,48,54,58]
TIMES=sorted(set(BASE+[(a+b)//2 for a,b in zip(BASE,BASE[1:])]))
VAL=[1,4,12,13,14]
cv2.setNumThreads(2)
def write(p,x):acq.write(p,x)
def setup():
 p=OUT/'protocol.json';assert not p.exists()
 sel=json.load(open(OLD/'selection.json'))
 protocol={'baseline_source_indices':BASE,'candidate_source_indices':TIMES,'candidate':'arithmetic mean of all original+hflip probabilities, threshold0.5, frozen V7 TPO',
 'gate':'Strict primary-development Macro-F1 improvement required before candidate new-val inference. No tuning. If failed, reserve new-val unscored.',
 'primary_modes':['full_frame','jpeg_75','blur_1'],'stress_modes':['jpeg_50','blur_2','resize_050'],
 'selection':sel['selection']|{'new_val':VAL},'new_val_selection_basis':'AI paired thumbnails before scoring: val1 plaza with bordering road; val4 station street; val12 stone road with vehicle; val13 parked-car street; val14 street with car. Candidate subset, final crop content QA before model inference.',
 'new_val_independence':'New split-qualified REDS clips, same camera/screen. Seoul and Turkey geography overlaps previous pool; val14 restaurant appears related to train142. NOT independent location/session/device holdout.',
 'proxy_time_contract':'Preserve prior 12 source indices in safe interior 1..58 (existing proxy endpoint exclusion). Add integer midpoints. This proxy uses 60 source frames with roughly 3 recapture frames per source, NOT long dashcam videos. Native MP4 runtime checked separately with exact V7 linspace sampler.',
 'acquisition_source':'Raw iPhone VDmoire plus official REDS camera sharp, no restored clean labels','training_gate':'No target training unless distinct source training/development/test AND held-out recapture device coverage are available. Do not train on new-val test.',
 'created_before_inference':True,'device':'iPhoneXR / MacBook Pro','original_weight':.5,'hflip_weight':.5,'threshold':.5,
 'frozen_files':{str(p.relative_to(ROOT)):digest(p) for p in [ROOT/'releases/v7/source/model/stage1/tpo/merged_visual.pt',ROOT/'releases/v7/source/model/stage1/config.json',ROOT/'releases/v7/source/model/stage2/code/solution/stage1_v7_tpo_hflip.py',Path(__file__)]}}
 assert len(TIMES)==23 and set(BASE)<=set(TIMES)
 write(p,protocol)
def acquire():
 pro=json.load(open(OUT/'protocol.json')); vd=json.load(open(OLD/'vd_index.json')); ti=json.load(open(OLD/'reds_index.json')); vi=json.load(open(OUT/'reds_val_index.json'))
 indexes={'vd':vd,'train':ti,'val':vi}; urls={'vd':acq.VD,'train':json.load(open(OLD/'reds_archive.json'))['url'],'val':json.load(open(OUT/'reds_val_archive.json'))['url']}
 jobs=collections.defaultdict(list); mapping=[]
 for split,ns in pro['selection'].items():
  dataset='val' if split=='new_val' else 'train'
  for n in ns:
   vn=sorted(k for k in vd if k.startswith(f'frames/{dataset}/Reds/video_{n}/') and k.endswith('.jpg'))
   numbers=[int(Path(k).stem) for k in vn];assert numbers==list(range(min(numbers),max(numbers)+1))
   rn=sorted(k for k in indexes[dataset] if f'/{n:03d}/' in k and k.endswith('.png'))
   for t in TIMES:
    # Old baseline bytes are kept exactly, including their prior geometric crop.
    for tag,key,member in [('original',dataset,rn[t]),('recapture','vd',vn[3*t+1])]:
     oldtag='reds' if tag=='original' else 'vd'
     oldpath=OLD/'source'/oldtag/f'{n:03d}'/Path(member).name
     path=oldpath if dataset=='train' and oldpath.exists() else OUT/'source'/dataset/f'{n:03d}'/tag/Path(member).name
     jobs[key].append((member,path));mapping.append({'split':split,'dataset':dataset,'source':n,'source_group':f'REDS_{dataset}_{n:03d}','time':t,'baseline':t in BASE,'label':tag,'path':str(path.relative_to(ROOT)),'member':member})
 write(OUT/'source_mapping.json',mapping)
 for key,jj in jobs.items():
  print(key,'missing_download_MB',sum(indexes[key][n]['size']+4096 for n,p in jj if not p.exists())/1e6,flush=True)
  acq.acquire(jj,urls[key],indexes[key],OUT/f'acquired_{key}.json',650_000_000)
def crop(a,b):
 h,m=match(a,b)
 if h is None or m['inliers']<12 or m.get('inlier_ratio',0)<.2 or m.get('median_error',99)>3:return None,m
 ah,aw=a.shape[:2];bh,bw=b.shape[:2]
 polygon=cv2.perspectiveTransform(np.float32([[[0,0],[aw-1,0],[aw-1,ah-1],[0,ah-1]]]),h)[0]
 cx,cy=map(float,cv2.perspectiveTransform(np.float32([[[aw/2,ah/2]]]),h)[0,0]);x,y=round(cx-320),round(cy-180)
 corners=np.float32([[x,y],[x+639,y],[x+639,y+359],[x,y+359]])
 safe=min(cv2.pointPolygonTest(polygon,tuple(map(float,p)),True) for p in corners)
 if safe<10 or x<0 or y<0 or x+640>bw or y+360>bh:return None,m|{'crop_fail':True}
 inv=cv2.perspectiveTransform(corners[None],np.linalg.inv(h))[0];ox,oy=np.floor(inv.min(0)).astype(int);ex,ey=np.ceil(inv.max(0)).astype(int)+1
 if ox<0 or oy<0 or ex>aw or ey>ah:return None,m|{'source_crop_fail':True}
 return {'original':jpeg95(cv2.resize(a[oy:ey,ox:ex],(640,360),interpolation=cv2.INTER_AREA)), 'recapture':jpeg95(b[y:y+360,x:x+640])},m|{'screen_margin':safe,'original_rect':[int(v) for v in [ox,oy,ex,ey]],'recapture_rect':[x,y,x+640,y+360]}
def prepare():
 mapping=json.load(open(OUT/'source_mapping.json'));old=json.load(open(OLD/'prepared_frames.json'));groups=collections.defaultdict(dict)
 for r in mapping:groups[r['source_group'],r['time']][r['label']]=r
 rows=[];qa=[]
 for (group,t),rr in groups.items():
  r=rr['original'];common={k:r[k] for k in ['split','dataset','source','source_group','time','baseline']}
  if r['dataset']=='train' and t in BASE:
   for label in ['original','recapture']:
    o=next(x for x in old if x['source']==r['source'] and x['label']==label and x['frame']==BASE.index(t))
    assert digest(ROOT/o['path'])==o['sha256'];rows.append(common|{k:o[k] for k in ['label','path','sha256','parent','parent_sha256']})
   continue
  a,b=[cv2.imread(str(ROOT/rr[label]['path'])) for label in ['original','recapture']];assert a is not None and b is not None
  crops,meta=crop(a,b);qa.append(common|meta|{'passed':crops is not None})
  if crops is None:continue
  for label,im in crops.items():
   p=OUT/'prepared'/r['dataset']/f'{r["source"]:03d}'/label/f'{t:02d}.png';p.parent.mkdir(parents=True,exist_ok=True);assert cv2.imwrite(str(p),im)
   parent=ROOT/rr[label]['path'];rows.append(common|{'label':label,'path':str(p.relative_to(ROOT)),'sha256':digest(p),'parent':str(parent.relative_to(ROOT)),'parent_sha256':digest(parent)})
  if len(qa)%20==0:print('prepared',len(qa),flush=True)
 write(OUT/'prepared_frames.json',rows);write(OUT/'crop_qa.json',qa)
 counts=collections.Counter((r['source_group'],r['label']) for r in rows)
 assert all(v==23 for v in counts.values()) and len(counts)==42,(counts,[x for x in qa if not x['passed']])
 hashes=collections.defaultdict(set)
 for r in rows:
  for k,d in [('path','sha256'),('parent','parent_sha256')]:assert digest(ROOT/r[k])==r[d]
  hashes[r['sha256']].add((r['source_group'],r['label']))
 assert all(len(v)==1 for v in hashes.values())
 write(OUT/'integrity.json',{'status':'PASS_SOURCE_PROXY_NOT_DEVICE_HOLDOUT','images':len(rows),'source_pairs':len(counts)//2,'old_baseline_images_preserved':384,'new_val_source_pairs':len(VAL),'min_inliers':min(x['inliers'] for x in qa),'min_screen_margin':min(x['screen_margin'] for x in qa),'max_median_error':max(x['median_error'] for x in qa),'exact_cross_group_label_duplicates':0,'device_holdout':False,'independent_location_confirmed':False})
 for page in range(2):
  im=Image.new('RGB',(1280,960),'#222');d=ImageDraw.Draw(im)
  for j,n in enumerate(VAL[page*3:(page+1)*3]):
   for k,label in enumerate(['original','recapture']):
    p=OUT/'prepared'/'val'/f'{n:03d}'/label/'32.png';a=Image.open(p);a.thumbnail((640,300));im.paste(a,(k*640,j*320+20));d.text((k*640+5,j*320),f'val{n:03d} {label}',fill='white')
  im.save(OUT/f'crop_val_{page}.jpg')
 print('INTEGRITY PASS',len(rows),flush=True)
def evaluate(split):
 # Gate and content approval precede candidate access to new-val scores.
 from evaluate_stage1_road_pairs import metrics,MODES,PRIMARY
 from diagnose_uhdm_full_robustness import transform
 import torch
 import importlib.util
 source=ROOT/'releases/v7/source/model/stage2/code/solution/stage1_tpo_merged.py'
 spec=importlib.util.spec_from_file_location('frozen_v7_tpo',source)
 module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);TPODetector=module.TPODetector
 pro=json.load(open(OUT/'protocol.json'));assert (OUT/'content_qa.json').exists()
 for p,h in pro['frozen_files'].items():assert digest(ROOT/p)==h,p
 if split!='development':assert json.load(open(OUT/'development/summary.json'))['development_gate_pass'],'No dev gain: do not open confirmation scores'
 dest=OUT/split;dest.mkdir(exist_ok=False)
 rows=json.load(open(OUT/'prepared_frames.json'));groups=collections.defaultdict(list)
 for r in rows:
  if r['split']==split:groups[r['source_group'],r['label']].append(r)
 torch.set_num_threads(2);cv2.setNumThreads(2);model=TPODetector(ROOT/'releases/v7/source/model/stage1/tpo',device='cpu');model.score([np.zeros((360,640,3),np.uint8)]*46)
 results=[];old={}
 if split!='new_val':old={(r['source'],r['label'],r['mode']):r for r in json.load(open(ROOT/f'artifacts/stage1_tpo_hflip_20260918/{split}/predictions.json'))}
 maxdiff=0.
 def score(arr):
  start=time.perf_counter();flip=[np.ascontiguousarray(a[:,::-1,:]) for a in arr];s=model.score(arr+flip);v=.5*float(s[:len(arr)].mean())+.5*float(s[len(arr):].mean());return v,s,time.perf_counter()-start
 try:
  for (group,label),items in sorted(groups.items()):
   items.sort(key=lambda r:r['time']);assert len(items)==23
   imgs=[]
   for r in items:assert digest(ROOT/r['path'])==r['sha256'];imgs.append(np.asarray(Image.open(ROOT/r['path']).convert('RGB')))
   for mode in MODES:
    aa=[transform(a,mode) for a in imgs];bb=[a for a,r in zip(aa,items) if r['baseline']];assert len(bb)==12
    if len(results)%2==0:b,bs,bt=score(bb);c,cs,ct=score(aa)
    else:c,cs,ct=score(aa);b,bs,bt=score(bb)
    if old:maxdiff=max(maxdiff,abs(b-old[items[0]['source'],label,mode]['candidate']))
    results.append({'source_group':group,'source':items[0]['source'],'label':label,'mode':mode,'baseline':b,'candidate':c,'baseline_seconds':bt,'candidate_seconds':ct,'baseline_frame_scores':bs.tolist(),'candidate_frame_scores':cs.tolist()})
   write(dest/'predictions.json',results);print(split,group,label,len(results),flush=True)
 finally:model.close()
 assert maxdiff<1e-6,maxdiff
 summary={'split':split,'conditions':{},'max_previous_v7_difference':maxdiff,'sources':len(groups)//2,'scope':'12 vs nested23 on frozen proxy, not official score; repeated conditions not independent'}
 for mode in ['primary_all',*MODES]:
  rr=[r for r in results if r['mode'] in PRIMARY] if mode=='primary_all' else [r for r in results if r['mode']==mode]
  summary['conditions'][mode]={method:metrics([dict(label=r['label'],tpo=r[method],forensic=0) for r in rr],0,.5) for method in ['baseline','candidate']}
  summary['conditions'][mode]['changes']={key:sum(r['label']==label and (r['baseline']>=.5)==b and (r['candidate']>=.5)==c for r in rr) for key,label,b,c in [('new_fp','original',False,True),('fixed_fp','original',True,False),('new_fn','recapture',True,False),('fixed_fn','recapture',False,True)]}
 summary['development_gate_pass']=summary['conditions']['primary_all']['candidate']['macro_f1']>summary['conditions']['primary_all']['baseline']['macro_f1'] if split=='development' else None
 summary['timing']={m:{'total_seconds':sum(r[m+'_seconds'] for r in results),'mean_seconds':float(np.mean([r[m+'_seconds'] for r in results]))} for m in ['baseline','candidate']}
 summary['files_unchanged']=all(digest(ROOT/p)==h for p,h in pro['frozen_files'].items());assert summary['files_unchanged']
 write(dest/'summary.json',summary);write(dest/'decision_changes.json',[r for r in results if (r['baseline']>=.5)!=(r['candidate']>=.5)])
 print(json.dumps(summary,indent=2),flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('action',choices=['setup','acquire','prepare','development','holdout','new_val']);args=p.parse_args()
 if args.action in ['setup','acquire','prepare']:globals()[args.action]()
 else:evaluate(args.action)
