"""Read-only audit of released videos; write diagnostics outside data."""
from pathlib import Path
import json, hashlib, time
import av
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'Baseline/data'
OUT = ROOT / 'artifacts/data_audit'

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    records=[]
    for path in sorted(DATA.rglob('*.mp4')):
        start=time.perf_counter()
        frames=[]; pts=[]; count=0; errors=[]
        try:
            with av.open(str(path)) as container:
                stream=container.streams.video[0]
                meta={'width':stream.width,'height':stream.height,'reported_frames':stream.frames,
                      'average_rate':str(stream.average_rate),'time_base':str(stream.time_base)}
                for frame in container.decode(video=0):
                    pts.append(float(frame.time) if frame.time is not None else None)
                    if count in ([0,120,240,360,480,600,720,840,960,1080] if 'stage3' in path.parts else [0,5,10,15,20,25,30,35,40,45]):
                        im=frame.to_image();im.thumbnail((384,240))
                        frames.append((count,im))
                    count+=1
        except Exception as exc: errors.append(repr(exc));meta={}
        canvas=Image.new('RGB',(384*5,270*2),'#171717');draw=ImageDraw.Draw(canvas)
        for i,(idx,im) in enumerate(frames):
            x=(i%5)*384;y=(i//5)*270
            canvas.paste(im,(x,y+25));draw.text((x+8,y+5),f'frame {idx}',fill='white')
        key='_'.join(path.relative_to(DATA).parts).replace('.mp4','.jpg')
        canvas.save(OUT/key,quality=90)
        diff=np.diff([x for x in pts if x is not None])
        record={'path':path.relative_to(DATA).as_posix(),**meta,'decoded_frames':count,
                'pts_first':pts[0] if pts else None,'pts_last':pts[-1] if pts else None,
                'nonpositive_pts_steps':int((diff<=0).sum()),
                'delta_time_quantiles':np.quantile(diff,[0,.5,1]).tolist() if len(diff) else [],
                'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
                'elapsed_seconds':round(time.perf_counter()-start,2),'errors':errors}
        records.append(record);print(json.dumps(record),flush=True)
    (OUT/'audit.json').write_text(json.dumps(records,indent=2),encoding='utf-8')

if __name__=='__main__':main()
