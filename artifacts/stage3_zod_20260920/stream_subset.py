"""Resume a gzip prefix, extracting only the selected leading sequence IDs."""
from intake import O,URL,session
from pathlib import Path
import json,tarfile,shutil,sys,time,hashlib
class PrefixReader:
 def __init__(self,name):
  self.path=O/'archives'/(name+'.prefix');self.size=self.path.stat().st_size if self.path.exists() else 0;self.old=self.path.open('rb') if self.size else None;self.left=self.size;self.pos=0;self.new=0;self.name=name;self.t=time.monotonic();self.last=self.t
  self.write=self.path.open('ab');self.response=None
 def read(self,n):
  if self.left:
   b=self.old.read(min(n,self.left));self.left-=len(b);self.pos+=len(b);return b
  if self.response is None:
   s=session();self.response=s.post('https://content.dropboxapi.com/2/sharing/get_shared_link_file',headers={'Dropbox-API-Arg':json.dumps({'url':URL,'path':'/sequences/'+self.name}),'Range':f'bytes={self.size}-'},stream=True,timeout=(30,90));self.response.raise_for_status();assert self.response.status_code==206 and self.response.headers['Content-Range'].startswith(f'bytes {self.size}-')
  b=self.response.raw.read(n);self.write.write(b);self.pos+=len(b);self.new+=len(b)
  if time.monotonic()-self.last>20:print(self.name,'new_bytes',self.new,'total_prefix',self.pos,flush=True);self.last=time.monotonic()
  assert self.pos<6_000_000_000,'bounded prefix size cap6GB'
  return b
 def close(self):
  self.write.close()
  if self.old:self.old.close()
  if self.response:self.response.close()
def main(name,end):
 reader=PrefixReader(name);rows=[];ids=set();stopped=False
 wanted={r['id'] for r in json.loads((O/'split_manifest.json').read_text()) if r['role']!='excluded'} if '--selected' in sys.argv else None
 try:
  with tarfile.open(fileobj=reader,mode='r|gz',bufsize=1024*1024) as tar:
   for m in tar:
    if not m.isfile():continue
    p=Path(m.name);assert not p.is_absolute() and '..' not in p.parts
    sid=next((x for x in p.parts if len(x)==6 and x.isdigit()),None)
    if sid is None:continue
    if int(sid)>end:stopped=True;break
    if wanted is not None and sid not in wanted:continue
    dst=O/'raw'/p;dst.parent.mkdir(parents=True,exist_ok=True)
    with tar.extractfile(m) as f:
     b=f.read();assert len(b)==m.size
    if dst.exists():assert hashlib.sha256(dst.read_bytes()).digest()==hashlib.sha256(b).digest(),dst
    else:dst.write_bytes(b)
    ids.add(sid);rows.append({'path':str(p),'bytes':m.size,'sha256':hashlib.sha256(b).hexdigest()})
 finally:reader.close()
 report={'archive':name,'end_id':end,'complete_selected_sequences_through_end':stopped,'selection':sorted(wanted) if wanted else None,'ids':sorted(ids),'files':rows,'new_network_bytes':reader.new,'prefix_bytes':reader.pos,'archive_full_hash_verified':False,'integrity':'provider full archive hash unavailable for prefix; each extracted file hashed, structural/decode checks separate','seconds':time.monotonic()-reader.t}
 (O/f'{name}.through_{end:06}.json').write_text(json.dumps(report,indent=2));print('finished',name,len(ids),'sequences',reader.new,'newbytes',flush=True)
if __name__=='__main__':main(sys.argv[1],int(sys.argv[2]))
