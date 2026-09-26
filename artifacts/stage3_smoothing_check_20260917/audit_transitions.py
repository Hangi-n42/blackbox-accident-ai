"""Correct event association, not labels/candidate: prevent matching neighboring transitions."""
import json
from pathlib import Path
import numpy as np
O=Path(__file__).resolve().parent;R=O.parents[1];D=R/'artifacts/pipeline_diagnosis_20260917';OLD=R/'artifacts/stage3_smoothing_20260917'
def read(p):return json.loads(p.read_text())
def write(p,d):p.write_text(json.dumps(d,ensure_ascii=False,indent=2))
def detect(pred,e):
 lo=max(e['old_run_start'],e['left_index']-20);hi=min(e['new_run_end']-2,e['right_index']+20,len(pred)-3);seen=False
 for k in range(lo,hi+1):
  if np.all(pred[k:k+3]==e['old_class']):seen=True
  if seen and np.all(pred[k:k+3]==e['new_class']):return {'index':k,'relative_s':k/10,'distance_outside_interval_s':max(e['left_index']-k,0,k-e['right_index'])/10}
 return None
# Prior true event at5.5 must not replace crossing7.9 when old-state anchor begins6.3.
p=np.full(100,2);p[63:79]=0;e={'old_run_start':63,'new_run_end':90,'left_index':72,'right_index':77,'old_class':0,'new_class':2};assert detect(p,e)['index']==79
assert detect(np.zeros(100,int),e) is None
rows=[]
for e in read(O/'transition_manifest.json'):
 root=O if e['prior_group']=='extra' else D;d=np.load(root/(e['id']+'_labels.npz'));y=d['accel_candidate'];left=e['left_index'];right=e['right_index'];start=left;end=right
 while start>0 and y[start-1]==e['old_class']:start-=1
 while end+1<len(y) and y[end+1]==e['new_class']:end+=1
 event=dict(e,old_run_start=start,new_run_end=end);p=np.load((O if e['prior_group']=='extra' else OLD)/(e['id']+'_probabilities.npz'));base=p['classes'][p['probabilities'].argmax(1)];cand=p['classes'][p['smoothed_probabilities'].argmax(1)];a=detect(base,event);b=detect(cand,event)
 rows.append(dict(event,baseline=a,candidate=b,candidate_minus_baseline_s=(b['index']-a['index'])/10 if a and b else None))
summaries={}
for group in ['all','extra','previous']:
 ts=rows if group=='all' else [t for t in rows if t['prior_group']==group];both=[t for t in ts if t['baseline'] and t['candidate']]
 summaries[group]={'n':len(ts),'both_detected':len(both),'baseline_missing':sum(t['baseline'] is None for t in ts),'candidate_missing':sum(t['candidate'] is None for t in ts),'baseline_matched_mean_distance_s':float(np.mean([t['baseline']['distance_outside_interval_s'] for t in both])) if both else None,'candidate_matched_mean_distance_s':float(np.mean([t['candidate']['distance_outside_interval_s'] for t in both])) if both else None,'later_count':sum(t['candidate_minus_baseline_s']>0 for t in both),'earlier_count':sum(t['candidate_minus_baseline_s']<0 for t in both),'mean_signed_shift_s':float(np.mean([t['candidate_minus_baseline_s'] for t in both])) if both else None}
write(O/'transitions_corrected.json',rows)
r=read(O/'report.json');s=summaries['all'];gates=dict(r['gates']);gates['matched_transition_distance_no_worse']=s['both_detected']>0 and s['candidate_matched_mean_distance_s']<=s['baseline_matched_mean_distance_s'];gates['missing_crossings_no_increase']=s['candidate_missing']<=s['baseline_missing']
write(O/'transition_scoring_audit.json',{'supersedes':'report.json transition metrics and its decision; stable metrics unaffected','problem':'Search +/-2s matched a prior real CONSTANT period5.5s to later7.2-7.7s transition. This is event-association error, not evidence of2.5s model shift.','correction':'Restrict search to known old/new contiguous proxy runs and original +/-2s bound; labels/probabilities/candidate unchanged. Keep original output for audit.','regression_test':'prior neighboring constant run not matched; absent crossing retained missing','summaries':summaries,'gates':gates,'decision':'keep_provisional_not_production' if all(gates.values()) else 'do_not_promote_keep_production_baseline'})
print(json.dumps({'summaries':summaries,'gates':gates},indent=2))
