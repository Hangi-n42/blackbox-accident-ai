"""Post-hoc feature neutralization probe, not fitted or deployed correction."""
import run
import numpy as np,joblib
O,R,old=run.O,run.R,run.old
assert not (O/'sensitivity_freeze.json').exists()
old.write(O/'sensitivity_freeze.json',{'scope':'Exploratory diagnosis after primary outputs. Not a score-gain experiment or mask selection. All12 ROIs treated identically; never choose best ROI per example.',
 'probe':'For each frozen standardized feature vector, set one ROI across all6 temporal blocks,4channels,3stats (72 of864features) to0=training mean. Predict fixed classifier without refit. Compare central11samples; count output changes, wrong-to-correct, correct-to-wrong. Absent feature means do not necessarily represent physically possible video.',
 'candidate_if_supported':'One predefined training-only ROI masking augmentation, uniform across12ROIs, existing864 representation and linear classifier; not an inference mask.',
 'script_sha256':old.sha(__import__('pathlib').Path(__file__))})
models={'comma2395':joblib.load(old.Z/'models/comma_only.joblib'),'production':joblib.load(R/'releases/v7/source/model/stage3/motion_model.joblib')['accel']}
cs=old.load();out=[]
for p in old.read(O/'matched_manifest.json'):
 for name,m in models.items():
  for ds in ['comma','zod']:
   q=p[ds];x=old.feature(cs[q['key']])[q['i']-5:q['i']+6];z=m[0].transform(x);base=m[-1].predict(z);truth=p['truth']
   for roi in range(12):
    masked=z.copy();masked.reshape(-1,6,12,4,3)[:,:,roi]=0
    pred=m[-1].predict(masked)
    out.append({'pair':p['pair_id'],'dataset':ds,'model':name,'roi':roi,'base_correct':int((base==truth).sum()),'prediction_changes':int((pred!=base).sum()),'wrong_to_correct':int(((base!=truth)&(pred==truth)).sum()),'correct_to_wrong':int(((base==truth)&(pred!=truth)).sum()),'base':base.tolist(),'probe':pred.tolist(),'truth':truth})
old.write(O/'roi_sensitivity.json',out)
summary=[]
for p in old.read(O/'matched_manifest.json'):
 for name in models:
  for ds in ['comma','zod']:
   rr=[r for r in out if r['pair']==p['pair_id'] and r['model']==name and r['dataset']==ds]
   summary.append({'pair':p['pair_id'],'model':name,'dataset':ds,'base_correct_of11':rr[0]['base_correct'],'rois_changing_any_prediction':sum(r['prediction_changes']>0 for r in rr),'rois_restoring_any_correct':sum(r['wrong_to_correct']>0 for r in rr),'rois_breaking_any_correct':sum(r['correct_to_wrong']>0 for r in rr),'probe_outcomes_of132':{'changed':sum(r['prediction_changes'] for r in rr),'wrong_to_correct':sum(r['wrong_to_correct'] for r in rr),'correct_to_wrong':sum(r['correct_to_wrong'] for r in rr)}})
old.write(O/'roi_sensitivity_summary.json',summary)
print(__import__('json').dumps(summary,indent=2))
