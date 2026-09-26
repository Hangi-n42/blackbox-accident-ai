from pathlib import Path
import tarfile,shutil,json,sys
O=Path(__file__).resolve().parent
name=sys.argv[1];out=O/'raw';out.mkdir(exist_ok=True);rows=[]
with tarfile.open(O/'archives'/name,'r:gz') as tar:
 for m in tar:
  if not m.isfile():continue
  p=Path(m.name)
  assert not p.is_absolute() and '..' not in p.parts
  # Keep only video/images and required state/metadata; never execute archive members.
  if any(k in m.name.lower() for k in ['lidar','radar']):continue
  dst=out/p;dst.parent.mkdir(parents=True,exist_ok=True)
  if not dst.exists():
   with tar.extractfile(m) as src,dst.open('xb') as f:shutil.copyfileobj(src,f)
  rows.append({'path':str(dst.relative_to(O)),'bytes':m.size})
(O/(name+'.members.json')).write_text(json.dumps(rows,indent=2));print(len(rows),'files');print('\n'.join(r['path'] for r in rows[:15]))
