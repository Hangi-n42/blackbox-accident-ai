import json
from pathlib import Path
import numpy as np,pandas as pd,av
from PIL import Image,ImageDraw
O=Path(__file__).resolve().parent;R=O.parents[1];B=R/'artifacts/stage3_training_basis_20260917';cases=json.load(open(B/'cases.json'));s=pd.read_csv(O/'full_context.csv');episodes=[('expanded_14',16.2,19.0),('extra_01',24.8,29.4),('extra_01',43.7,45.2)];sheet=Image.new('RGB',(1200,3*190),'white');draw=ImageDraw.Draw(sheet);manifest=[]
for id in ['expanded_14','extra_01']:
 c=next(c for c in cases if c['id']==id);d=np.load(B/c['labels_npz']);plans={}
 for row,(eid,a,b) in enumerate(episodes):
  if eid==id:plans[row]=np.array([max(0,int(a*10)-23),int(a*10),int((a+b)*5),int(b*10),min(len(d['time'])-1,int(b*10)+23)])
 targets={int(d['frame_index'][i]) for ix in plans.values() for i in ix};frames={}
 with av.open(str(R/c['raw_path'])) as container:
  for k,f in enumerate(container.decode(video=0)):
   if k in targets:frames[k]=f.to_image()
   if k>=max(targets):break
 for row,ix in plans.items():
  draw.text((3,row*190),id+' | native index frames; relative sample times | speed, proxy a, steering',fill='black')
  for col,i in enumerate(ix):
   k=int(d['frame_index'][i]);im=frames[k].copy();im.thumbnail((238,135));sheet.paste(im,(240*col,190*row+20));draw.text((240*col,190*row+158),f'{i/10:.1f}s v{d["speed_smoothed"][i]:.1f} a{d["acceleration_proxy"][i]:+.2f}',fill='black');draw.text((240*col,190*row+172),f'steer {d["steering_smoothed"][i]:+.1f} deg',fill='black');manifest.append({'id':id,'sample_index':int(i),'native_frame_index':k,'source_path':c['raw_path'],'frame_time':float(d['frame_time'][i])})
sheet.save(O/'review.jpg');(O/'visual_manifest.json').write_text(json.dumps(manifest,indent=2))
print('15 native frames saved; sensor/probability curves retained in full_context.csv')
