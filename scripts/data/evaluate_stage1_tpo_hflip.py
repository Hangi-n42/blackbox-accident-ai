"""One fixed TPO candidate: original/horizontal flip 50:50, threshold .5; dev gate."""
import collections,json,time
from pathlib import Path
from diagnose_uhdm_branches import ROOT,RELEASE,Image,TPODetector,cv2,np,torch,sha,write
from diagnose_uhdm_full_robustness import transform
from evaluate_stage1_road_pairs import metrics,MODES,PRIMARY
DATA=ROOT/'artifacts/stage1_road_pairs_20260918'
PRIOR=ROOT/'artifacts/stage1_road_comparison_20260918'
BASE=ROOT/'artifacts/stage1_tpo_hflip_20260918'

def hflip(a):
 return np.ascontiguousarray(a[:, ::-1, :])

def main(split):
 OUT=BASE/split
 if split=='holdout':
  gate=json.loads((BASE/'development/summary.json').read_text())
  assert gate['development_gate_pass'], 'No primary development F1 gain: stop, no holdout evaluation'
 OUT.mkdir(parents=True,exist_ok=False)
 test=np.arange(360*640*3,dtype=np.int32).reshape(360,640,3)
 flipped=hflip(test)
 assert flipped.shape==test.shape and flipped.dtype==test.dtype
 assert np.array_equal(flipped[:,0,:],test[:,-1,:]) and np.array_equal(hflip(flipped),test)
 prior=json.loads((PRIOR/'protocol.json').read_text());frozen=dict(prior['files_sha256'])
 for split in [split]:frozen[str((PRIOR/f'{split}_scores.json').relative_to(ROOT))]=sha(PRIOR/f'{split}_scores.json')
 frozen[str(Path(__file__).relative_to(ROOT))]=sha(Path(__file__))
 for path,digest in frozen.items():assert sha(ROOT/path)==digest,path
 protocol={'scope':'one fixed candidate on previously exposed proxy; development gate before holdout, NOT a fresh independent test','evaluated_split':split,'development_gate':'Strictly greater macro-F1 on primary 3 conditions; no threshold/weight search',
 'baseline':'mean TPO of same 12 full frames >=0.5','candidate':'0.5*mean TPO(full12)+0.5*mean TPO(horizontal_flip12) >=0.5',
 'augmentation':'horizontal pixel-order reversal only, contiguous copy, no interpolation or compression; before existing TPO224 preprocessing',
 'modes':MODES,'primary_modes':PRIMARY,'no_tuning':True,
 'timing':'CPU 2 threads, warmed single loaded model. Baseline and candidate independently scored; order alternates per record. Includes TPO preprocessing, forward, means and candidate flip. Excludes common PNG decoding, condition transforms, disk I/O and shared model load. 96 paired measurements per split; not end-to-end MP4 timing.',
 'versions':{'torch':torch.__version__,'numpy':np.__version__,'opencv':cv2.__version__},'files_sha256':frozen}
 write(OUT/'protocol.json',protocol)
 records=json.loads((DATA/'prepared_frames.json').read_text());groups=collections.defaultdict(list)
 for r in records:
  if r['split']==split:groups[r['split'],r['source'],r['label']].append(r)
 previous={(r['split'],r['source'],r['label'],r['mode']):r for split in [split] for r in json.loads((PRIOR/f'{split}_scores.json').read_text())}
 torch.set_num_threads(2);cv2.setNumThreads(2);start=time.perf_counter();detector=TPODetector(RELEASE/'model/stage1/tpo',device='cpu');load_seconds=time.perf_counter()-start
 warm=np.zeros((360,640,3),np.uint8)
 detector.score([warm]*12);detector.score([warm]*24)
 rows=[];max_old_diff=0.;max_duplicate_diff=0.
 def baseline(aa):
  t=time.perf_counter();scores=detector.score(aa);mean=float(scores.mean());return mean,scores,time.perf_counter()-t
 def candidate(aa):
  t=time.perf_counter();cc=[hflip(a) for a in aa];scores=detector.score(aa+cc);full=float(scores[:12].mean());center=float(scores[12:].mean());value=.5*full+.5*center;return value,scores,time.perf_counter()-t
 try:
  for (split,source,label),items in sorted(groups.items()):
   items=sorted(items,key=lambda r:r['frame']);assert len(items)==12
   arrays=[]
   for r in items:
    assert sha(ROOT/r['path'])==r['sha256'];arrays.append(np.asarray(Image.open(ROOT/r['path']).convert('RGB')))
   for mode in MODES:
    aa=[transform(a,mode) for a in arrays]
    if len(rows)%2==0:b,bs,bt=baseline(aa);c,cs,ct=candidate(aa)
    else:c,cs,ct=candidate(aa);b,bs,bt=baseline(aa)
    old=previous[split,source,label,mode];max_old_diff=max(max_old_diff,abs(old['tpo']-b));max_duplicate_diff=max(max_duplicate_diff,float(np.max(np.abs(bs-cs[:12]))))
    assert all(0<=x<=1 for x in [b,c])
    rows.append({'split':split,'source':source,'cluster':items[0]['cluster'],'label':label,'mode':mode,'baseline':b,'candidate':c,'flipped':float(cs[12:].mean()),'full_frame_scores':bs.tolist(),'candidate_frame_scores':cs.tolist(),'baseline_seconds':bt,'candidate_seconds':ct,'baseline_first':len(rows)%2==0})
   write(OUT/'predictions.json',rows);print(split,source,label,'completed',len(rows),flush=True)
 finally:detector.close()
 assert len(rows)==96 and len({(r['split'],r['source'],r['label'],r['mode']) for r in rows})==96
 assert max_old_diff<1e-6 and max_duplicate_diff<1e-6
 summary={'splits':{},'max_previous_baseline_score_difference':max_old_diff,'max_repeated_full_frame_difference':max_duplicate_diff,'model_load_seconds':load_seconds,'timing':{}}
 for split in [split]:
  summary['splits'][split]={}
  for mode in ['primary_all',*MODES]:
   rr=[r for r in rows if r['split']==split and (r['mode'] in PRIMARY if mode=='primary_all' else r['mode']==mode)]
   result={}
   for method in ['baseline','candidate']:
    adapted=[dict(label=r['label'],tpo=r[method],forensic=0) for r in rr];result[method]=metrics(adapted,0,.5)
   result['original_new_fp']=sum(r['label']=='original' and r['baseline']<.5 and r['candidate']>=.5 for r in rr)
   result['original_fixed_fp']=sum(r['label']=='original' and r['baseline']>=.5 and r['candidate']<.5 for r in rr)
   result['recapture_new_fn']=sum(r['label']=='recapture' and r['baseline']>=.5 and r['candidate']<.5 for r in rr)
   result['recapture_fixed_fn']=sum(r['label']=='recapture' and r['baseline']<.5 and r['candidate']>=.5 for r in rr)
   summary['splits'][split][mode]=result
 for method in ['baseline','candidate']:
  v=np.array([r[method+'_seconds'] for r in rows]);summary['timing'][method]={'total_seconds':float(v.sum()),'mean_seconds_per_12_frame_record':float(v.mean()),'median_seconds':float(np.median(v)),'p95_seconds':float(np.quantile(v,.95))}
 summary['timing']['total_ratio']=summary['timing']['candidate']['total_seconds']/summary['timing']['baseline']['total_seconds']
 summary['files_unchanged']=all(sha(ROOT/path)==digest for path,digest in frozen.items());assert summary['files_unchanged']
 result=summary['splits'][split]['primary_all']
 summary['development_gate_pass']=(result['candidate']['macro_f1']>result['baseline']['macro_f1']) if split=='development' else None
 write(OUT/'summary.json',summary)
 flips=[r for r in rows if (r['baseline']>=.5)!=(r['candidate']>=.5)];write(OUT/'decision_changes.json',flips)
 print(json.dumps(summary,indent=2))
if __name__=='__main__':
 import argparse
 p=argparse.ArgumentParser();p.add_argument('split',choices=['development','holdout']);main(p.parse_args().split)
