import argparse,json
from pathlib import Path
import av
from PIL import Image, ImageDraw
p=argparse.ArgumentParser();p.add_argument('clip');p.add_argument('start',type=int);p.add_argument('end',type=int);p.add_argument('--step',type=int,default=1);a=p.parse_args()
base=Path(__file__).resolve().parent
src=base.parents[1]/'acquisition'/f'{a.clip}.mp4'
out=base/f'{a.clip}_{a.start:04}_{a.end:04}_s{a.step}'
out.mkdir(exist_ok=True)
wanted=set(range(a.start,a.end+1,a.step)); frames=[]
with av.open(str(src)) as c:
 for i,f in enumerate(c.decode(video=0)):
  if i>a.end: break
  if i in wanted:
   im=f.to_image(); path=out/f'f{i:04}.png';im.save(path)
   frames.append((i,f.pts,float(f.time),im))
for start in range(0,len(frames),8):
 chunk=frames[start:start+8]; sheet=Image.new('RGB',(960,296*((len(chunk)+1)//2)),(15,15,15));d=ImageDraw.Draw(sheet)
 for k,(i,pts,t,im) in enumerate(chunk):
  x=(k%2)*480;y=(k//2)*296
  d.text((x+5,y+5),f'{a.clip} f={i} pts={pts} t={t:.6f}',fill='white')
  im=im.copy();im.thumbnail((480,270));sheet.paste(im,(x,y+26))
 path=out/f'sheet_{start//8:02}.jpg';sheet.save(path,quality=95);print(path)
(out/'manifest.json').write_text(json.dumps([{'frame_id':i,'native_pts':pts,'time_s':t,'image':str(out/f'f{i:04}.png')} for i,pts,t,im in frames],indent=2))
