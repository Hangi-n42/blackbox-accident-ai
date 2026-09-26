import collections,json
from pathlib import Path
from PIL import Image,ImageDraw
O=Path(__file__).resolve().parent;ROOT=O.parents[1]
rows=json.load(open(O/'dlc_confirmation_acquired.json'));g=collections.defaultdict(list)
for r in rows:g[r['document_id'],r['label'],r['camera']].append(r)
for typ in ['fin_id','grc_passport','svk_id']:
 docs=sorted({r['document_id'] for r in rows if r['document_type']==typ})
 if len(docs)!=2 or any(len(g[d,l,c])!=12 for d in docs for l in ['or','re'] for c in ['iphone','android']):continue
 im=Image.new('RGB',(1000,620),'#202020');draw=ImageDraw.Draw(im)
 for j,d in enumerate(docs):
  for k,(l,c) in enumerate([('or','iphone'),('re','iphone'),('or','android'),('re','android')]):
   rr=sorted(g[d,l,c],key=lambda r:int(Path(r['archive_path']).stem));a=Image.open(ROOT/rr[6]['path']);a.thumbnail((244,278));im.paste(a,(k*250,j*310+28));draw.text((k*250+3,j*310+5),f'{d} {l} {c}',fill='white')
 p=O/f'confirmation_contact_{typ}.jpg';im.save(p);print(p)
