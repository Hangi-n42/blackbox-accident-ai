import argparse,json
from pathlib import Path
import av
from PIL import Image,ImageDraw
base=Path('/Users/hyeongi/projects/blackbox-accident-ai/artifacts/stage2_goal_20260919')
a=argparse.ArgumentParser(); a.add_argument('video'); a.add_argument('--frames'); a.add_argument('--tag',default='full_2hz'); a.add_argument('--width',type=int,default=480); a.add_argument('--cols',type=int,default=3); a.add_argument('--per-page',type=int,default=18); args=a.parse_args()
m=json.loads((base/'acquisition'/f'{args.video}.pts.json').read_text())['mapping']
if args.frames:
 ids=set()
 for part in args.frames.split(','):
  if ':' in part:
   vals=[int(x) for x in part.split(':')]; ids.update(range(vals[0],vals[1]+1,vals[2] if len(vals)>2 else 1))
  else: ids.add(int(part))
else:
 ids={min(range(len(m)),key=lambda k:abs(m[k]['time_s']-t/2)) for t in range(int(m[-1]['time_s']*2)+1)}; ids.add(len(m)-1)
out=base/'expert_mac_review'/'reviewer_a'/args.video/args.tag; out.mkdir(parents=True,exist_ok=True)
cap=av.open(str(base/'acquisition'/f'{args.video}.mp4'))
imgs=[]
for i,frm in enumerate(cap.decode(video=0)):
 assert frm.pts == m[i]["native_pts"], (i,frm.pts,m[i]["native_pts"])
 if i not in ids: continue
 im=frm.to_image(); im.thumbnail((args.width,1000))
 cell=Image.new('RGB',(args.width,im.height+28),'#fff'); cell.paste(im,(0,28)); ImageDraw.Draw(cell).text((7,7),f'{args.video} f{i} PTS={m[i]["native_pts"]} t={m[i]["time_s"]:.6f}',fill='#000')
 imgs.append((i,cell))
 if args.frames: frm.to_image().save(out/f'f{i:04}.png')
cap.close()
for n in range(0,len(imgs),args.per_page):
 batch=imgs[n:n+args.per_page]; w=args.width; h=batch[0][1].height
 sheet=Image.new('RGB',(w*args.cols,h*((len(batch)+args.cols-1)//args.cols)),'#eee')
 for k,(_,im) in enumerate(batch): sheet.paste(im,((k%args.cols)*w,(k//args.cols)*h))
 path=out/f'sheet_{n//args.per_page+1:02}.jpg';sheet.save(path,quality=94);print(path)
(out/'frames.json').write_text(json.dumps([m[i] for i,_ in imgs],indent=2))
