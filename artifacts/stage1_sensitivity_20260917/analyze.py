import json,sys,hashlib,time
from pathlib import Path
import numpy as np,cv2,torch
from PIL import Image
ROOT=Path(__file__).resolve().parents[2]; OUT=Path(__file__).resolve().parent
release=ROOT/'artifacts/submissions/verify_v6'; sys.path.insert(0,str(release/'model/stage2/code'))
from solution.stage1 import extract_features,feature_probability
from solution.stage1_tpo_merged import TPODetector
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
model=release/'model/stage1'; cfg=json.loads((model/'config.json').read_text()); art=json.loads((model/cfg['forensic_artifact']).read_text())
old=json.loads((ROOT/'artifacts/data_pilot_20260916/vdmoire/baseline_comparison.json').read_text())
freeze=json.loads((ROOT/'artifacts/pipeline_diagnosis_20260917/freeze.json').read_text())['files']
paths=[release/'inference.py',ROOT/'scripts/data/score_vdmoire_baseline.py',ROOT/'scripts/migration/run_stage.py',model/'config.json',model/cfg['forensic_artifact'],model/'tpo/merged_visual.pt',model/'tpo/clip_source/clip/model.py']
for n in ['stage1.py','stage1_v4.py','stage1_tpo_merged.py']: paths.extend([ROOT/'solution'/n,release/'model/stage2/code/solution'/n])
provenance={str(p.relative_to(ROOT)):{'current_sha256':sha(p),'previous_diagnosis_sha256':freeze.get(str(p.relative_to(ROOT)))} for p in paths}
for v in provenance.values():v['matches_diagnosis']=v['current_sha256']==v['previous_diagnosis_sha256'] if v['previous_diagnosis_sha256'] else None
(OUT/'provenance.json').write_text(json.dumps({'files':provenance,'reason_for_small_rerun':'Original VD pilot lacks complete contemporaneous code/model hashes. Current frozen scoring path rerun only for original 5 source groups, not all 20.','config':cfg},indent=2))
torch.set_num_threads(2);cv2.setNumThreads(2); detector=TPODetector(model/'tpo',device='cpu');rows=[];start=time.monotonic()
try:
 for mode in ['full_frame','central_192']:
  for r in old[mode]['rows']:
   frames=[]
   for name in r['paths']:
    with Image.open(ROOT/name) as im:a=np.array(im.convert('RGB'))
    if mode=='central_192':h,w=a.shape[:2];a=a[h//2-96:h//2+96,w//2-96:w//2+96]
    frames.append(a)
   f=feature_probability(extract_features(frames)[0],art);t=float(detector.score(frames).mean());p=cfg['forensic_weight']*f+(1-cfg['forensic_weight'])*t
   rows.append({'source_group':r['source_group'],'provider_label':'screen_recapture' if r['label']==1 else 'provided_clean_reference','mode':mode,'paths':r['paths'],'forensic_probability':f,'tpo_probability':t,'probability':p,'recapture_decision':bool(p>=cfg['threshold']),'old_probability':r['probability'],'delta_old_probability':p-r['probability'],'official_original_ground_truth':None,'development_exposed':True})
finally:detector.close()
assert len(rows)==20 and len({r['source_group'] for r in rows})==5
(OUT/'predictions.json').write_text(json.dumps(rows,indent=2))
comparison=[]
for r in rows[:10]:
 c=next(x for x in rows[10:] if x['source_group']==r['source_group'] and x['provider_label']==r['provider_label'])
 comparison.append({'source_group':r['source_group'],'provider_label':r['provider_label'],'full':{k:r[k] for k in ['forensic_probability','tpo_probability','probability','recapture_decision']},'crop':{k:c[k] for k in ['forensic_probability','tpo_probability','probability','recapture_decision']},'delta_crop_minus_full':{k:c[k]-r[k] for k in ['forensic_probability','tpo_probability','probability']}})
summary={'source_groups':5,'unique_frames':len({p for r in rows for p in r['paths']}),'scoring_rows':20,'seconds':time.monotonic()-start,'max_abs_delta_from_old':max(abs(r['delta_old_probability']) for r in rows),'by_mode':{}}
for mode in ['full_frame','central_192']:
 rr=[r for r in rows if r['mode']==mode and r['provider_label']=='screen_recapture'];summary['by_mode'][mode]={'screen_recapture_cases':len(rr),'ensemble_misses':sum(not r['recapture_decision'] for r in rr),'forensic_below_point5':sum(r['forensic_probability']<.5 for r in rr),'tpo_below_point5':sum(r['tpo_probability']<.5 for r in rr),'mean_probability':float(np.mean([r['probability'] for r in rr]))}
(OUT/'branch_comparison.json').write_text(json.dumps(comparison,indent=2));(OUT/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))
