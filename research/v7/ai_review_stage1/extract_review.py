"""Read-only native video extraction for independent AI review; no model or labels."""
from pathlib import Path
import argparse,json,hashlib
import av,numpy as np
from PIL import Image,ImageDraw

ROOT=Path(__file__).resolve().parents[3]
SOURCE=ROOT/'research/v7/fresh_sources_retry1'
OUT=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    p=argparse.ArgumentParser();p.add_argument('ID',choices=['00014','00015','00016']);p.add_argument('--start',type=int,default=0);p.add_argument('--end',type=int);p.add_argument('--count',type=int,default=36);p.add_argument('--tag',default='overview');p.add_argument('--width',type=int,default=320);args=p.parse_args()
    acquisition=json.loads((SOURCE/'acquisition.json').read_text(encoding='utf8'));record=next(r for r in acquisition['records'] if r['ID']==args.ID);source=SOURCE/(args.ID+'.mp4');source_sha=sha(source);assert source_sha==record['sha256']
    folder=OUT/args.ID;folder.mkdir(exist_ok=True);meta=folder/'native_mapping.json'
    if meta.exists():mapping=json.loads(meta.read_text(encoding='utf8'))['frames']
    else:
        mapping=[]
        with av.open(str(source)) as c:
            c.streams.video[0].thread_count=2
            for i,f in enumerate(c.decode(video=0)):
                mapping.append(dict(frame=i,pts=f.pts,time_base=str(f.time_base),seconds=float(f.pts*f.time_base)))
        meta.write_text(json.dumps(dict(source=record,source_sha256=source_sha,frames=mapping),indent=2)+'\n',encoding='utf8')
    end=len(mapping)-1 if args.end is None else min(args.end,len(mapping)-1);indices=np.unique(np.linspace(args.start,end,args.count).round().astype(int)).tolist();wanted=set(indices);tiles={};selected=[]
    with av.open(str(source)) as c:
        c.streams.video[0].thread_count=2
        for i,f in enumerate(c.decode(video=0)):
            if i in wanted:
                im=f.to_image().convert('RGB');ip=folder/f'frame_{i:06d}.png'
                if not ip.exists():im.save(ip)
                else:assert np.array_equal(np.asarray(Image.open(ip).convert('RGB')),np.asarray(im))
                h=round(args.width*im.height/im.width);tile=Image.new('RGB',(args.width,h+28),'#111111');tile.paste(im.resize((args.width,h)),(0,28));ImageDraw.Draw(tile).text((5,7),f'{args.ID} f={i} t={mapping[i]["seconds"]:.6f}s',fill='white');tiles[i]=tile
                selected.append(dict(**mapping[i],path=str(ip),sha256=sha(ip),RGB_sha256=hashlib.sha256(np.asarray(im).tobytes()).hexdigest()))
            if i>=end:break
    columns=4 if args.width<=320 else 3;tw,th=next(iter(tiles.values())).size;sheet=Image.new('RGB',(tw*columns,th*((len(indices)+columns-1)//columns)),'#111111')
    for j,i in enumerate(indices):sheet.paste(tiles[i],((j%columns)*tw,(j//columns)*th))
    sp=folder/(args.tag+'.jpg');assert not sp.exists();sheet.save(sp,quality=93)
    assert sha(source)==source_sha
    (folder/(args.tag+'_manifest.json')).write_text(json.dumps(dict(source=record,selection='Uniform original decoded indices within requested inclusive range; actual native PTS labels',indices=indices,selected=selected,sheet_sha256=sha(sp)),indent=2)+'\n',encoding='utf8')
    print(json.dumps(dict(ID=args.ID,count=len(mapping),first=mapping[0],last=mapping[-1],sheet=str(sp))))
if __name__=='__main__':main()
