"""Independent arithmetic and artifact audit, no refit or label changes."""
from qa import O,R,read,write
import hashlib,json
import numpy as np,pandas as pd
from PIL import Image,ImageDraw
df=pd.read_csv(O/'comparison_predictions.csv');scores=read(O/'comparison_metrics.json');checks=[]
def calc(g,name):
 cm=np.zeros((4,4),int)
 for y,p in zip(g.truth,g[name]):cm[int(y),int(p)]+=1
 den=cm.sum(0)+cm.sum(1);f=np.divide(2*cm.diagonal(),den,out=np.zeros(4,float),where=den>0)
 return {'n':len(g),'macro_f1':float(f.mean()),'observed_macro_f1':float(f[cm.sum(1)>0].mean()),'class_f1':f.tolist(),'confusion':cm.tolist(),'opposite':int(cm[0,1]+cm[1,0])}
for s in scores:
 g=df[df.scope==s['scope']]
 if s['group']!='all':g=g[g.group==s['group']]
 m=calc(g,s['variant']);assert abs(m['macro_f1']-s['macro_f1'])<1e-12 and m['confusion']==s['confusion']
 checks.append([s['scope'],s['group'],s['variant']])
z=df[df.scope=='zod_heldout'];gp=read(O/'geographic_stratum_pre_results.json');ids=[r['heldout'] for r in gp['rows'] if r['geographically_separated']]
# pandas integer inference strips zeroes from numeric IDs only when the entire column is numeric; mixed OPEN IDs keep strings.
ids=set(ids);zg=z[z.id.isin(ids)];geo={n:calc(zg,n) for n in ['baseline','candidate']}
op=lambda y,p:((y==0)&(p==1))|((y==1)&(p==0))
resolved=op(z.truth,z.baseline)&~op(z.truth,z.candidate)
zt=z[z.truth.isin([0,1])]
summary={'metrics_independently_recomputed':len(checks),'zod_class_counts':np.bincount(z.truth,minlength=4).tolist(),
 'zod_contributing_videos':int(z.id.nunique()),'reserved_but_no_strict_rows':['000039'],
 'zod_accel_decel_correct_baseline':int((zt.baseline==zt.truth).sum()),'zod_accel_decel_correct_candidate':int((zt.candidate==zt.truth).sum()),
 'zod_resolved_opposite_to_correct':int((resolved&(z.candidate==z.truth)).sum()),'zod_resolved_opposite_to_other_wrong':int((resolved&(z.candidate!=z.truth)).sum()),
 'zod_corrected_by_class':z[z.corrected].groupby('truth').size().to_dict(),
 'geographically_separated_ids':sorted(ids),'geographically_separated_metrics':geo,'production_zod':calc(z,'production'),
 'temporal_dependence':'472 samples are not472 independent experiments;7 contributing20s clips and4 date/vehicle groups. No statistical generalization claim.'}
freeze=read(O/'freeze.json');sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert all(sha(R/p)==h for p,h in {**freeze['protected'],**freeze['label_files'],**read(O/'comparison_inputs.json')}.items())
for sid in freeze['training_ids']+freeze['heldout_ids']:
 x=np.load(O/'features'/(sid+'.npz'))['base'];d=np.load(O/'labels'/(sid+'.npz'));assert x.shape==(len(d['time']),864) and np.isfinite(x).all();assert not np.isfinite(d['speed'][~d['speed_valid']]).any()
summary.update(protected_and_original_inputs_unchanged=True,all33_feature_arrays_finite=True,masked_speed_not_zero_imputed=True)
write(O/'audit.json',summary)
# Evidence for the main failure: A/D becomes CONSTANT; labels stay frozen.
vis=O/'visual';ev=[]
for sid in ['000014','000036']:
 d=np.load(O/'labels'/(sid+'.npz'));c=z[z.id==sid];indices=[]
 for label in [0,1]:
  g=c[c.truth==label]
  if len(g):indices.append(int(g.iloc[len(g)//2].sample_index))
 sheet=Image.new('RGB',(500*len(indices),350),'white');draw=ImageDraw.Draw(sheet);fr=read(O/'raw/sequences'/sid/'info.json')['camera_frames']['front_blur']
 for col,i in enumerate(indices):
  row=c[c.sample_index==i].iloc[0];p=O/'raw'/fr[int(d['frame_index'][i])]['filepath'];im=Image.open(p).resize((500,282));sheet.paste(im,(500*col,68))
  draw.text((500*col+4,4),f"{sid} t={i/10:.1f}s speed={d['speed_smoothed'][i]:.2f}\na={d['acceleration_proxy'][i]:.3f} truth={row.truth}\nbaseline={row.baseline} candidate={row.candidate}",fill='black');ev.append({'id':sid,'sample_index':i,'image':str(p.relative_to(R)),'truth':int(row.truth),'baseline':int(row.baseline),'candidate':int(row.candidate)})
 sheet.save(vis/(sid+'_errors.jpg'))
write(vis/'error_examples.json',ev)
print(json.dumps(summary,indent=2))
