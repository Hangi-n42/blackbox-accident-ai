"""Bounded public ZIP acquisition; source-linked physical recapture evaluation."""
import concurrent.futures,hashlib,json,struct,sys,time,zipfile,zlib
from pathlib import Path
import requests
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'research'))
from acquire_comma_subset import RemoteZipFile
OUT=ROOT/'artifacts/stage1_road_pairs_20260918'
VD=json.loads((ROOT/'artifacts/data_pilot_20260916/vdmoire/manifest.json').read_text())['source_url']
REDS='https://huggingface.co/datasets/snah/REDS/resolve/main/train_sharp.zip?download=true&road=20260918'
def write(path,obj):path.write_text(json.dumps(obj,indent=2,ensure_ascii=False))
def index(url,tag):
 p=OUT/f'{tag}_index.json'
 if p.exists():return json.loads(p.read_text())
 r=RemoteZipFile(url,budget=30_000_000)
 with zipfile.ZipFile(r) as z:
  records={i.filename:{'offset':i.header_offset,'size':i.compress_size,'uncompressed':i.file_size,'crc':i.CRC,'method':i.compress_type} for i in z.infolist() if not i.is_dir()}
 write(p,records);write(OUT/f'{tag}_archive.json',{'url':url,'archive_bytes':r.size,'index_transfer_bytes':r.used+128});return records

def fetch(url,name,info,target):
 if target.exists():
  data=target.read_bytes();assert len(data)==info['uncompressed'] and zlib.crc32(data)==info['crc']
  return {'member':name,'path':str(target.relative_to(ROOT)),'sha256':hashlib.sha256(data).hexdigest(),'downloaded_bytes':0}
 start=info['offset'];end=start+info['size']+4095
 for attempt in range(3):
  try:
   with requests.get(url,params={'road_start':start,'road_end':end,'retry':str(time.time_ns()) if attempt else 'initial'},headers={'Range':f'bytes={start}-{end}'},timeout=60,stream=True) as r:
    r.raise_for_status();assert r.status_code==206,(r.status_code,name)
    assert r.headers['Content-Range'].startswith(f'bytes {start}-')
    blob=r.raw.read(end-start+2);assert len(blob)<=end-start+1
   assert blob[:4]==b'PK\x03\x04'
   header=struct.unpack('<4s5H3I2H',blob[:30]);a,b=header[-2:];payload=blob[30+a+b:30+a+b+info['size']]
   assert len(payload)==info['size']
   data=zlib.decompress(payload,-15) if info['method']==8 else payload
   assert info['method'] in (0,8) and len(data)==info['uncompressed'] and zlib.crc32(data)==info['crc']
   target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(data)
   return {'member':name,'path':str(target.relative_to(ROOT)),'sha256':hashlib.sha256(data).hexdigest(),'downloaded_bytes':len(blob)}
  except Exception:
   if attempt==2:raise
   time.sleep(attempt+1)

def acquire(jobs,url,idx,manifest,budget):
 assert sum(idx[n]['size']+4096 for n,p in jobs if not p.exists())<=budget
 rows=[]
 with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
  tasks={pool.submit(fetch,url,n,idx[n],p):n for n,p in jobs}
  for future in concurrent.futures.as_completed(tasks):
   rows.append(future.result());write(manifest,rows)
   if len(rows)%10==0:print('acquired',len(rows),'/',len(jobs),flush=True)
 return rows

def scan():
 from PIL import Image,ImageDraw
 idx=index(VD,'vd');groups={}
 for name in idx:
  if name.startswith('frames/train/Reds/') and name.endswith('.jpg'):groups.setdefault(int(name.split('/')[3].split('_')[1]),[]).append(name)
 jobs=[(sorted(names)[len(names)//2],OUT/'scan'/f'{n:03d}.jpg') for n,names in sorted(groups.items())]
 rows=acquire(jobs,VD,idx,OUT/'scan_manifest.json',100_000_000)
 for page in range((len(jobs)+29)//30):
  im=Image.new('RGB',(1500,1200),'#222');d=ImageDraw.Draw(im)
  for i,(name,p) in enumerate(jobs[page*30:(page+1)*30]):
   x=(i%5)*300;y=(i//5)*200;pic=Image.open(p);pic.thumbnail((296,170));im.paste(pic,(x,y+25));d.text((x+3,y+3),p.stem,fill='white')
  im.save(OUT/f'scan_{page}.jpg')
 print('scan complete',len(rows),'bytes',sum(r['downloaded_bytes'] for r in rows),flush=True)

def sequences():
 selection={'development':[0,2,13,16,38,39,51,67], 'holdout':[103,104,109,141,142,152,154,155]}
 protocol={'selection_basis':'AI inspection of source thumbnails before model inference: visible road and vehicles; Seoul development, distinct Turkey scenes holdout',
 'selection':selection,'previously_exposed_source_ids':[2,16], 'device':'iPhoneXR / MacBook Pro; same device both splits',
 'frames_per_class_per_source':12,'source_frames':'REDS indices [1,5,11,16,21,27,32,38,43,48,54,58]; raw middle of corresponding 3-frame groups, index 3*k+1; exclude presentation endpoints uniformly',
 'scope':'public physical street recapture proxy; not dashboard-camera or device independent evaluation',
 'location_clusters':{'0':'seoul_campus','2':'seoul_campus','13':'seoul_old_street','16':'seoul_main_roads','38':'seoul_main_roads','39':'seoul_main_roads','51':'seoul_main_roads','67':'seoul_campus', '103':'turkey_old_street','104':'turkey_old_street','109':'turkey_rock_road','141':'turkey_market','142':'turkey_market','152':'turkey_city_roads','154':'turkey_city_roads','155':'turkey_city_roads'},
 'cluster_basis':'conservative visual grouping; exact capture-session metadata unavailable',
 'fixed_before_inference':True}
 write(OUT/'selection.json',protocol)
 vd=index(VD,'vd');reds=index(REDS,'reds')
 for tag,url,idx in [('vd',VD,vd),('reds',REDS,reds)]:
  jobs=[]
  for split,numbers in selection.items():
   for n in numbers:
    names=sorted(k for k in idx if (k.startswith(f'frames/train/Reds/video_{n}/') and k.endswith('.jpg')) if tag=='vd') if tag=='vd' else sorted(k for k in idx if f'/{n:03d}/' in k and k.endswith('.png'))[:60]
    assert len(names)>=60,(tag,n,len(names))
    for i in range(12):
     k=[1,5,11,16,21,27,32,38,43,48,54,58][i];member=names[3*k+1 if tag=='vd' else k];jobs.append((member,OUT/'source'/tag/f'{n:03d}'/Path(member).name))
  print(tag,'expected compressed MB',sum(idx[n]['size'] for n,p in jobs)/1e6,flush=True)
  acquire(jobs,url,idx,OUT/f'{tag}_frames.json',450_000_000)

if __name__=='__main__':
 import argparse
 p=argparse.ArgumentParser();p.add_argument('action',choices=['scan','sequences']);args=p.parse_args()
 OUT.mkdir(exist_ok=True,parents=True)
 (scan if args.action=='scan' else sequences)()
