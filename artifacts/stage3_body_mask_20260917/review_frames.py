import json,cv2
from pathlib import Path
from PIL import Image,ImageDraw
O=Path(__file__).resolve().parent;R=O.parents[1];rows=json.loads((R/'artifacts/stage3_lower_roi_ablation_20260917/cases.json').read_text())
rows += [dict(id=f'OPEN_{i:03d}',input_path=f'artifacts/public_eval_10hz/stage3/videos/OPEN_{i:03d}.mp4',role='public') for i in range(1,6)]
manifest=[]
for page in range(5):
 sheet=Image.new('RGB',(1350,840),'white');draw=ImageDraw.Draw(sheet)
 for rowno,r in enumerate(rows[page*3:page*3+3]):
  cap=cv2.VideoCapture(str(R/r['input_path']));count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT));indices=[min(20,count-1),count//2,max(0,count-21)]
  for col,i in enumerate(indices):
   cap.set(cv2.CAP_PROP_POS_FRAMES,i);ok,f=cap.read();assert ok;im=Image.fromarray(cv2.cvtColor(f,cv2.COLOR_BGR2RGB)).resize((450,245));dr=ImageDraw.Draw(im)
   for y in [.72,.80,.85,.90]:dr.line((0,y*245,450,y*245),fill='yellow',width=1);dr.text((2,y*245-12),str(y),fill='yellow')
   sheet.paste(im,(col*450,rowno*280+30));draw.text((col*450+4,rowno*280+3),f'{r["id"]} frame {i} / {count}',fill='black')
  cap.release();manifest.append(dict(r,review_frames=indices))
 sheet.save(O/f'review_{page}.jpg')
(O/'review_manifest.json').write_text(json.dumps(manifest,indent=2))
