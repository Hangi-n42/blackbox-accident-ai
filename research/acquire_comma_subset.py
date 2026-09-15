"""Read selected public CAN/video members from remote ZIP with HTTP ranges."""
import argparse, io, json, time, zipfile
from pathlib import Path
import requests
import numpy as np

class RemoteZipFile(io.RawIOBase):
    def __init__(self,url,budget=4_000_000_000):
        self.url=url; self.pos=0; self.used=0; self.budget=budget; self.session=requests.Session(); self.cache={}
        r=self.session.get(url,headers={'Range':'bytes=-128'},timeout=90); r.raise_for_status()
        if r.status_code!=206: raise RuntimeError('Server does not honor ranges')
        self.size=int(r.headers['Content-Range'].split('/')[-1]); self.cache[(self.size-len(r.content),len(r.content))]=r.content
    def seekable(self): return True
    def readable(self): return True
    def tell(self): return self.pos
    def seek(self,offset,whence=0):
        self.pos=offset if whence==0 else self.pos+offset if whence==1 else self.size+offset
        return self.pos
    def read(self,n=-1):
        if n<0:n=self.size-self.pos
        n=min(n,self.size-self.pos)
        if n<=0:return b''
        for (start,length),data in self.cache.items():
            if start<=self.pos and self.pos+n<=start+length:
                result=data[self.pos-start:self.pos-start+n];self.pos+=n;return result
        if self.used+n>self.budget:raise RuntimeError('Download budget exceeded')
        start=self.pos;end=start+n-1
        for attempt in range(4):
            try:
                r=self.session.get(self.url,params={'range_start':start,'range_end':end},headers={'Range':f'bytes={start}-{end}'},timeout=120)
                r.raise_for_status()
                if r.status_code!=206 or len(r.content)!=n:raise RuntimeError(f'Bad range: {r.status_code} {len(r.content)} expected {n}')
                result=r.content;break
            except Exception:
                if attempt==3:raise
                time.sleep(2**attempt)
        self.used+=len(result)
        if n<3_000_000:self.cache[(start,n)]=result
        self.pos+=n;return result

def main():
    p=argparse.ArgumentParser();p.add_argument('--chunk',type=int,default=1);p.add_argument('--segments',type=int,default=16);p.add_argument('--out',default='external_data/comma2k19');args=p.parse_args()
    dest=Path(args.out);dest.mkdir(parents=True,exist_ok=True)
    remote=RemoteZipFile(f'https://huggingface.co/datasets/commaai/comma2k19/resolve/main/raw_data/Chunk_{args.chunk}.zip')
    z=zipfile.ZipFile(remote); names=z.namelist()
    (dest/f'chunk{args.chunk}_members.json').write_text(json.dumps(names),encoding='utf-8')
    videos=[n for n in names if n.endswith('/video.hevc')]
    # Sample across distinct routes, then evenly within routes, without private data.
    routes={}
    for v in videos:routes.setdefault(v.rsplit('/',2)[0],[]).append(v)
    chosen=[]
    for layer in range(100):
        for route in sorted(routes):
            values=sorted(routes[route]); ix=np.linspace(0,len(values)-1,min(len(values),args.segments),dtype=int)
            if layer<len(ix) and len(chosen)<args.segments:chosen.append(values[ix[layer]])
        if len(chosen)>=args.segments:break
    manifest=[]
    for video in chosen:
        base=video.rsplit('/',1)[0]; route=base.rsplit('/',1)[0]; safe=base.replace('|','_').replace(':','_')
        members=[n for n in names if not n.endswith('/') and n.startswith(base+'/') and (n==video or '/CAN/speed/' in n or '/CAN/car_speed/' in n or '/CAN/steering_angle/' in n or '/frame_times' in n)]
        for name in members:
            relative=name[len(base)+1:];target=dest/safe/relative
            if not target.resolve().is_relative_to(dest.resolve()):raise ValueError('Unsafe ZIP member path')
            target.parent.mkdir(parents=True,exist_ok=True)
            if not target.exists():target.write_bytes(z.read(name))
        manifest.append({'route':route,'segment':base,'local':str(dest/safe),'source':remote.url,'members':members})
        (dest/f'chunk{args.chunk}_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
        print(f'Completed {len(manifest)}/{len(chosen)}; downloaded {remote.used/1e6:.1f} MB; {base}',flush=True)
    print('done',flush=True)
if __name__=='__main__':main()
