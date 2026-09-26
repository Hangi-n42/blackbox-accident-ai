from pathlib import Path
import json,av,numpy as np
from PIL import Image,ImageDraw
R=Path.cwd();O=R/'artifacts/stage3_model_difference_20260917';B=R/'artifacts/stage3_training_basis_20260917';rows=json.loads((O/'selected_cases.json').read_text());manifest=[]
for vehicle in ['rav4','civic']:
 es=[e for e in rows if e['vehicle']==vehicle]; sheet=Image.new('RGB',(1200,len(es)*186),'white');draw=ImageDraw.Draw(sheet)
 for j,e in enumerate(es):
  d=np.load(B/(e['id']+'_labels.npz'));mid=e['midpoint'];ix=np.clip(np.array([mid-23,mid-10,mid,mid+10,mid+23]),0,len(d['time'])-1);targets={int(d['frame_index'][i]):int(i) for i in ix};frames={}
  with av.open(str(R/e['original_path'])) as container:
   for k,f in enumerate(container.decode(video=0)):
    if k in targets:frames[k]=(f.to_image(),float(f.pts*f.time_base) if f.pts is not None else None)
    if k>=max(targets):break
  draw.text((3,j*186),f"{e['id']} {e['truth']} {e['category']} run {e['start']/10:.1f}-{e['end']/10:.1f}s",fill='black')
  for col,i in enumerate(ix):
   k=int(d['frame_index'][i]);im,pts=frames[k];im.thumbnail((238,136));sheet.paste(im,(col*240,j*186+20));draw.text((col*240,j*186+157),f"{i/10:.1f}s v={d['speed_smoothed'][i]:.2f} a={d['acceleration_proxy'][i]:+.2f}",fill='black');manifest.append({'id':e['id'],'sample_index':int(i),'native_index':k,'actual_native_pts':pts,'saved_frame_time':float(d['frame_time'][i])})
 sheet.save(O/(vehicle+'_review.jpg'))
(O/'visual_frame_manifest.json').write_text(json.dumps(manifest,indent=2))
print('8 contexts x 5 native frames saved; raw videos unchanged')
