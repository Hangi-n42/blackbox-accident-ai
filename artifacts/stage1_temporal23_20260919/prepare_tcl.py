import sys,json,collections
from pathlib import Path
sys.path.insert(0,'scripts/data')
import experiment_stage1_temporal23 as e
from PIL import Image,ImageDraw
import cv2
O=e.OUT;idx=json.load(open(O/'tcl_index.json'));u=json.load(open(O/'tcl_archive.json'))['url'];jobs=[];mapping=[]
for n in e.VAL:
 names=sorted(k for k in idx if k.startswith(f'frames/val/Reds/video_{n}/') and k.endswith('.jpg'))
 nums=[int(Path(k).stem) for k in names];assert nums==list(range(min(nums),max(nums)+1)),n
 for t in e.TIMES:
  name=names[3*t+1];p=O/'source/tcl_val'/f'{n:03d}'/Path(name).name;jobs.append((name,p));mapping.append({'source':n,'time':t,'member':name,'path':str(p.relative_to(e.ROOT))})
print('TCL additional MB',sum(idx[n]['size']+4096 for n,p in jobs if not p.exists())/1e6,flush=True)
e.acq.acquire(jobs,u,idx,O/'acquired_tcl.json',80_000_000)
base=json.load(open(O/'source_mapping.json'));rows=[];qa=[]
for r in mapping:
 source=next(x for x in base if x['split']=='new_val' and x['source']==r['source'] and x['time']==r['time'] and x['label']=='original')
 op=e.ROOT/source['path'];rp=e.ROOT/r['path'];crops,m=e.crop(cv2.imread(str(op)),cv2.imread(str(rp)));qa.append(r|m|{'passed':crops is not None})
 if crops is None:continue
 for label,im in crops.items():
  p=O/'prepared/tcl_val'/f'{r["source"]:03d}'/label/f'{r["time"]:02d}.png';p.parent.mkdir(parents=True,exist_ok=True);assert cv2.imwrite(str(p),im)
  parent=op if label=='original' else rp
  rows.append({'split':'new_tcl','dataset':'tcl_val','source':r['source'],'source_group':f'REDS_val_{r["source"]:03d}','time':r['time'],'baseline':r['time'] in e.BASE,'label':label,'path':str(p.relative_to(e.ROOT)),'sha256':e.digest(p),'parent':str(parent.relative_to(e.ROOT)),'parent_sha256':e.digest(parent)})
e.write(O/'tcl_prepared_frames.json',rows);e.write(O/'tcl_crop_qa.json',qa)
assert len(rows)==230 and all(r['passed'] for r in qa),[r for r in qa if not r['passed']]
for page in range(2):
 im=Image.new('RGB',(1280,960),'#222');d=ImageDraw.Draw(im)
 for j,n in enumerate(e.VAL[page*3:(page+1)*3]):
  for k,label in enumerate(['original','recapture']):
   a=Image.open(O/'prepared/tcl_val'/f'{n:03d}'/label/'32.png');a.thumbnail((640,300));im.paste(a,(k*640,j*320+20));d.text((k*640+5,j*320),f'TCL val{n:03d} {label}',fill='white')
 im.save(O/f'crop_tcl_{page}.jpg')
print('TCL DONE',len(rows),flush=True)
