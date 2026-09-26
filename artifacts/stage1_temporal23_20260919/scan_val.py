import sys,json,collections
from pathlib import Path
sys.path.insert(0,'scripts/data')
import build_stage1_road_pairs as b
from PIL import Image,ImageDraw
b.OUT=b.ROOT/'artifacts/stage1_temporal23_20260919'
idx=json.load(open('artifacts/stage1_road_pairs_20260918/vd_index.json'));groups=collections.defaultdict(list)
for n in idx:
 if n.startswith('frames/val/Reds/') and n.endswith('.jpg'):groups[int(n.split('/')[3].split('_')[1])].append(n)
jobs=[(sorted(ns)[len(ns)//2],b.OUT/'scan'/'vd'/f'{n:03d}.jpg') for n,ns in sorted(groups.items())]
print('VD MB',sum(idx[n]['size'] for n,p in jobs)/1e6,flush=True)
b.acquire(jobs,b.VD,idx,b.OUT/'scan_vd_manifest.json',30_000_000)
url='https://huggingface.co/datasets/snah/REDS/resolve/main/val_sharp.zip?download=true&temporal=20260919'
ridx=b.index(url,'reds_val')
jobs2=[(next(k for k in ridx if f'/{n:03d}/' in k and k.endswith('00000030.png')),b.OUT/'scan'/'reds'/f'{n:03d}.png') for n in sorted(groups)]
print('REDS MB',sum(ridx[n]['size'] for n,p in jobs2)/1e6,flush=True)
b.acquire(jobs2,url,ridx,b.OUT/'scan_reds_manifest.json',50_000_000)
for page in range(3):
 im=Image.new('RGB',(1280,960),'#222');d=ImageDraw.Draw(im)
 for j,n in enumerate(sorted(groups)[page*8:(page+1)*8]):
  for k,tag in enumerate(['reds','vd']):
   p=b.OUT/'scan'/tag/f'{n:03d}.{ "png" if tag=="reds" else "jpg"}'
   a=Image.open(p);a.thumbnail((312,205));x=(j%2)*640+k*320;y=(j//2)*240
   im.paste(a,(x,y+25));d.text((x+5,y+5),f'val {n:03d} {tag}',fill='white')
 im.save(b.OUT/f'scan_val_{page}.jpg')
print('DONE',flush=True)
