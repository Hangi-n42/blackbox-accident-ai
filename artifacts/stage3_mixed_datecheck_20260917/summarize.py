import json
from pathlib import Path
import numpy as np,pandas as pd
O=Path(__file__).resolve().parent; rows=[];changes=[]
for i in range(1,4):
 p=O/f'fold_{i}';report=json.loads((p/'report.json').read_text());s=pd.read_csv(p/'heldout_predictions.csv');fits=json.loads((p/'fit.json').read_text());a=next(x for x in fits if x['model']=='rav4');b=next(x for x in fits if x['model']=='mixed_budget_rav4');assert a['n']==b['n'] and a['class_counts']==b['class_counts']
 for scope,g in [('pooled',s),*list(s.groupby('vehicle'))]:
  out={'fold':i,'scope':scope,'n':len(g),'moving_gt_n':int(g.truth.isin([0,1]).sum()),'hold_dates':report['held_dates'],'training_samples':a['n']}
  masks={}
  for name in ['rav4','mixed_budget_rav4']:
   accel=(g.truth==0)&(g[name]==1);decel=(g.truth==1)&(g[name]==0);mask=accel|decel;masks[name]=mask
   out[name]={'f1':report['results'][scope][name]['macro_f1'],'opposite':int(mask.sum()),'rate':float(mask.sum()/out['moving_gt_n']) if out['moving_gt_n'] else None,'accel_to_decel':int(accel.sum()),'decel_to_accel':int(decel.sum()),'constant_errors':report['results'][scope][name]['constant_errors']}
  out['new_opposites']=int((~masks['rav4']&masks['mixed_budget_rav4']).sum());out['fixed_opposites']=int((masks['rav4']&~masks['mixed_budget_rav4']).sum());out['retained_opposites']=int((masks['rav4']&masks['mixed_budget_rav4']).sum());out['new_opposites_from_correct']=int(((g.rav4==g.truth)&masks['mixed_budget_rav4']).sum());out['new_opposites_from_other_error']=out['new_opposites']-out['new_opposites_from_correct'];rows.append(out)
  if scope=='pooled':
   c=g[masks['rav4']!=masks['mixed_budget_rav4']].copy();c['fold']=i;c['effect']=np.where(masks['mixed_budget_rav4'][c.index],'new_opposite','fixed_opposite');changes.append(c)
result={'rows':rows,'public':{str(i):{n:json.loads((O/f'fold_{i}/report.json').read_text())['results']['public'][n] for n in ['rav4','mixed_budget_rav4']} for i in range(1,4)},'no_pooled_cross_fold_score':'RAV4 test date repeated; Civic different dates; results are dependent development checks'}
(O/'summary.json').write_text(json.dumps(result,indent=2));pd.concat(changes).to_csv(O/'opposite_changes.csv',index=False)
for x in rows:print(x)
print('PUBLIC', {k:{n:round(m['macro_f1'],6) for n,m in v.items()} for k,v in result['public'].items()})
