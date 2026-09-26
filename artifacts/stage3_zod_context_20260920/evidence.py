"""Reuse existing plots; export complete6s windows and exact classifier margins."""
import run,runpy
from pathlib import Path
import numpy as np,av,joblib
from PIL import Image,ImageDraw
O,R,old=run.O,run.R,run.old
runpy.run_path(str(R/'artifacts/stage3_zod_matched_20260920/evidence.py'))
cs=old.load();pairs=old.read(O/'matched_manifest.json');images={};requests={};sources=[]
for p in pairs:
 for name in ['comma','zod']:
  q=p[name];requests.setdefault(q['key'],set()).update(q['i']+j for j in range(-30,31,10))
for key,indices in requests.items():
 c=cs[key];d=c['d']
 if c['dataset']=='comma':
  frame_map={int(d['frame_index'][i]):i for i in indices}
  with av.open(str(R/c['row']['raw_path'])) as con:
   for j,f in enumerate(con.decode(video=0)):
    if j in frame_map:
     i=frame_map[j];images[key,i]=f.to_image().resize((320,180));sources.append({'key':key,'index':i,'native_frame':j,'frame_time':float(d['frame_time'][i]),'pts':f.pts,'path':c['row']['raw_path']})
    if j>=max(frame_map):break
 else:
  frames=old.read(old.Z/'raw/sequences'/c['row']['id']/'info.json')['camera_frames']['front_blur']
  for i in indices:
   f=frames[int(d['frame_index'][i])];path=old.Z/'raw'/f['filepath'];images[key,i]=Image.open(path).convert('RGB').resize((320,180));sources.append({'key':key,'index':i,'native_frame':int(d['frame_index'][i]),'frame_time':float(d['frame_time'][i]),'path':str(path.relative_to(R))})
for p in pairs:
 im=Image.new('RGB',(2240,452),'white');draw=ImageDraw.Draw(im)
 for row,name in enumerate(['comma','zod']):
  q=p[name];draw.text((4,row*226+4),f"{p['pair_id']} {q['key']} full6s, center={q['i']/10:.1f}s",fill='black')
  for j,offset in enumerate(range(-30,31,10)):
   i=q['i']+offset;im.paste(images[q['key'],i],(320*j,row*226+40));draw.text((320*j+4,row*226+23),f'{i/10:.1f}s',fill='black')
 im.save(O/'evidence'/f"{p['pair_id']}_full6s.jpg")
old.write(O/'evidence/full6s_sources.json',sources)
models={'comma2395':joblib.load(old.Z/'models/comma_only.joblib'),'production':joblib.load(R/'releases/v7/source/model/stage3/motion_model.joblib')['accel']}
out=[]
for p in pairs:
 for name,m in models.items():
  scaled={};scores={}
  for ds in ['comma','zod']:
   q=p[ds];x=old.feature(cs[q['key']])[q['i']-5:q['i']+6];scaled[ds]=m[0].transform(x);scores[ds]=m[-1].decision_function(scaled[ds])
  truth=p['truth'];rival=max((k for k in range(4) if k!=truth),key=lambda k:scores['zod'][:,k].mean())
  coef=m[-1].coef_[truth]-m[-1].coef_[rival];contrib=(scaled['zod'].mean(0)-scaled['comma'].mean(0))*coef
  margin={ds:float((s[:,truth]-s[:,rival]).mean()) for ds,s in scores.items()}
  assert np.isclose(contrib.sum(),margin['zod']-margin['comma'],atol=1e-6)
  x=contrib.reshape(6,12,4,3)
  out.append({'pair':p['pair_id'],'model':name,'truth':truth,'rival':rival,'margins':margin,'delta':float(contrib.sum()),'by_block':x.sum((1,2,3)).tolist(),'by_roi':x.sum((0,2,3)).tolist(),'sum_checked':True})
old.write(O/'margin_attribution.json',out)
print('complete-window frames',len(sources),'attribution comparisons',len(out))
