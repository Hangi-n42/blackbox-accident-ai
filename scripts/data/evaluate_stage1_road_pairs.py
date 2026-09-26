"""Frozen video-aggregation comparison; development selection before holdout scoring."""
import argparse,collections,json,time
from pathlib import Path
from diagnose_uhdm_branches import ROOT,RELEASE,Image,TPODetector,cv2,np,torch,extract_features,feature_probability,sha,write
from diagnose_uhdm_full_robustness import transform
DATA=ROOT/'artifacts/stage1_road_pairs_20260918'
OUT=ROOT/'artifacts/stage1_road_comparison_20260918'
MODES=['full_frame','jpeg_75','blur_1','jpeg_50','blur_2','resize_050']
PRIMARY=['full_frame','jpeg_75','blur_1']
def metrics(rows,w,t):
 y=np.array([r['label']=='recapture' for r in rows]);p=np.array([w*r['forensic']+(1-w)*r['tpo']>=t for r in rows]);tp=int((y&p).sum());tn=int((~y&~p).sum());fp=int((~y&p).sum());fn=int((y&~p).sum())
 f1p=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.;f1n=2*tn/(2*tn+fp+fn) if 2*tn+fp+fn else 0.
 return {'macro_f1':(f1p+f1n)/2,'original_fpr':fp/(fp+tn),'recapture_fnr':fn/(fn+tp),'fp':fp,'fn':fn,'original_n':fp+tn,'recapture_n':fn+tp}
def init():
 OUT.mkdir(exist_ok=True)
 integrity=json.loads((DATA/'integrity.json').read_text());assert integrity['status']=='PASS_PROXY_INTEGRITY_NOT_TARGET_VALIDATION'
 cfg=RELEASE/'model/stage1/config.json';config=json.loads(cfg.read_text());artifact=RELEASE/'model/stage1'/config['forensic_artifact']
 paths=[cfg,artifact,RELEASE/'model/stage1/tpo/merged_visual.pt',RELEASE/'model/stage1/tpo/clip_source/clip/model.py',Path(__file__),DATA/'selection.json',DATA/'prepared_frames.json',DATA/'crop_qa.json',DATA/'integrity.json',ROOT/'scripts/data/build_stage1_road_pairs.py',ROOT/'scripts/data/prepare_stage1_road_pairs.py',ROOT/'scripts/data/verify_stage1_road_pairs.py']+[RELEASE/'model/stage2/code/solution'/f for f in ['stage1.py','stage1_v4.py','stage1_tpo_merged.py']]
 protocol={'scope':'16-source public road/street physical-recapture proxy; not official evaluation',
 'modes':MODES,'primary_selection_modes':PRIMARY,'stress_modes':[x for x in MODES if x not in PRIMARY],
 'forensic_weights':[0,.25,.5,.75,1],'thresholds':[round(x,2) for x in np.arange(.2,.901,.05)],
 'selection':'Max primary development macro-F1 subject to original FPR <= current; tie: lower FPR, lower FNR, closest current weight and threshold',
 'aggregation':'exact frozen V6 extract_features over 12 frames then one forensic probability; arithmetic mean of 12 TPO scores',
 'preprocessing':'SIFT homography only for geometry QA; raw axis-aligned crop640x360, corresponding original bounding crop resized640x360, both JPEG95 roundtrip stored PNG; no perspective warp of scoring images',
 'limitations':['one recapture device; not device holdout','handheld street source; not dashcam','approximate temporal correspondence only; source paired','original PNG and prior recapture JPEG histories differ despite common final encoding','visual location groups not verified independent recording sessions','small source count; repeated transformations not independent samples','source labels from provider and AI visual/geometric confirmation, not human per-file capture logs'],
 'files_sha256':{str(p.relative_to(ROOT)):sha(p) for p in paths},'config':config}
 assert config['forensic_weight']==config['threshold']==.5
 write(OUT/'protocol.json',protocol)
 return protocol

def score(split):
 protocol=json.loads((OUT/'protocol.json').read_text())
 for p,d in protocol['files_sha256'].items():assert sha(ROOT/p)==d,p
 if split=='holdout':assert (OUT/'selected.json').exists(),'Select and freeze candidate before holdout inference'
 groups=collections.defaultdict(list)
 for r in json.loads((DATA/'prepared_frames.json').read_text()):
  if r['split']==split:groups[r['source'],r['label']].append(r)
 assert len(groups)==16
 model=RELEASE/'model/stage1';artifact=json.loads((model/protocol['config']['forensic_artifact']).read_text())
 torch.set_num_threads(2);cv2.setNumThreads(2);detector=TPODetector(model/'tpo',device='cpu');rows=[];start=time.monotonic()
 try:
  for (source,label),frames in sorted(groups.items()):
   frames=sorted(frames,key=lambda r:r['frame']);assert len(frames)==12
   arrays=[]
   for r in frames:
    assert sha(ROOT/r['path'])==r['sha256'];assert sha(ROOT/r['parent'])==r['parent_sha256']
    arrays.append(np.asarray(Image.open(ROOT/r['path']).convert('RGB')))
   for mode in MODES:
    aa=[transform(a,mode) for a in arrays];f,_=extract_features(aa);fp=feature_probability(f,artifact);tp=detector.score(aa)
    row={'source':source,'cluster':frames[0]['cluster'],'split':split,'label':label,'mode':mode,'forensic':fp,'tpo':float(tp.mean()),'tpo_frame_scores':tp.tolist(),'ensemble':.5*fp+.5*float(tp.mean())};rows.append(row)
   write(OUT/f'{split}_scores.json',rows);print(split,source,label,'done',len(rows),flush=True)
 finally:detector.close()
 assert len(rows)==96
 write(OUT/f'{split}_verification.json',{'rows':len(rows),'seconds':time.monotonic()-start,'hashes_unchanged':all(sha(ROOT/p)==d for p,d in protocol['files_sha256'].items())})

def select():
 protocol=json.loads((OUT/'protocol.json').read_text());rows=json.loads((OUT/'development_scores.json').read_text());rows=[r for r in rows if r['mode'] in PRIMARY];base=metrics(rows,.5,.5);grid=[]
 for w in protocol['forensic_weights']:
  for t in protocol['thresholds']:
   m=metrics(rows,w,t);grid.append({'weight':w,'threshold':t,**m,'eligible':m['original_fpr']<=base['original_fpr']})
 eligible=[r for r in grid if r['eligible']];assert eligible
 best=max(eligible,key=lambda r:(r['macro_f1'],-r['original_fpr'],-r['recapture_fnr'],-abs(r['weight']-.5),-abs(r['threshold']-.5)))
 write(OUT/'development_grid.json',grid)
 assert not (OUT/'holdout_scores.json').exists(),'Candidate selection must precede holdout'
 write(OUT/'selected.json',{'selected':best,'baseline':base,'development_scores_sha256':sha(OUT/'development_scores.json'),'protocol_sha256':sha(OUT/'protocol.json'),'holdout_scores_existed':False})
 print('selected',json.dumps(best),flush=True)

def report():
 selected=json.loads((OUT/'selected.json').read_text());w=selected['selected']['weight'];t=selected['selected']['threshold'];allrows=[]
 candidates={'current':(.5,.5),'tpo_only':(0,.5),'fixed_tuned':(w,t),'forensic_only':(1,.5)};summary={'selected':selected,'splits':{}}
 for split in ['development','holdout']:
  rows=json.loads((OUT/f'{split}_scores.json').read_text());allrows+=rows;summary['splits'][split]={}
  for mode in ['primary_all',*MODES]:
   subset=[r for r in rows if r['mode'] in PRIMARY] if mode=='primary_all' else [r for r in rows if r['mode']==mode]
   summary['splits'][split][mode]={name:metrics(subset,*params) for name,params in candidates.items()}
 # Paired source-cluster bootstrap keeps labels and conditions together.
 rows=[r for r in allrows if r['split']=='holdout' and r['mode'] in PRIMARY];clusters=sorted({r['cluster'] for r in rows});rng=np.random.default_rng(20260918);deltas=[]
 for _ in range(2000):
  boot=[r for c in rng.choice(clusters,len(clusters),replace=True) for r in rows if r['cluster']==c];deltas.append(metrics(boot,w,t)['macro_f1']-metrics(boot,.5,.5)['macro_f1'])
 summary['holdout_cluster_bootstrap']={'clusters':clusters,'n_bootstrap':2000,'delta_macro_f1_95_percentile':np.quantile(deltas,[.025,.975]).tolist(),'caveat':'only four visually defined clusters; not population guarantee'}
 failures=[]
 for r in allrows:
  y=r['label']=='recapture';f=r['forensic']>=.5;tp=r['tpo']>=.5;cur=r['ensemble']>=.5;tuned=w*r['forensic']+(1-w)*r['tpo']>=t
  failures.append({k:r[k] for k in ['source','cluster','split','label','mode']}|{'forensic_wrong':bool(f!=y),'tpo_wrong':bool(tp!=y),'current_wrong':bool(cur!=y),'tuned_wrong':bool(tuned!=y),'tpo_correct_current_wrong':bool(tp==y and cur!=y),'both_branches_wrong':bool(f!=y and tp!=y)})
 write(OUT/'failures.json',failures);write(OUT/'summary.json',summary);print(json.dumps(summary,indent=2))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('action',choices=['init','development','select','holdout','report']);a=p.parse_args()
 if a.action=='init':init()
 elif a.action in ['development','holdout']:score(a.action)
 elif a.action=='select':select()
 else:report()
