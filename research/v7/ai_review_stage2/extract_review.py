"""CPU-only image evidence extraction. No predictions or human labels are read."""
import os
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'): os.environ[k]='2'
import argparse, hashlib, json, math
from pathlib import Path
import av
from PIL import Image, ImageDraw, ImageOps, ImageFont

HERE=Path(__file__).resolve().parent
SOURCE=HERE.parent/'fresh_sources_retry1'

def sha(path):
    with Path(path).open('rb') as f: return hashlib.file_digest(f,'sha256').hexdigest()

def put(path,data):
    with path.open('x',encoding='utf-8') as f: json.dump(data,f,ensure_ascii=False,indent=2)

def sheet(rows,folder,prefix,columns=4):
    for batch in range(math.ceil(len(rows)/20)):
        group=rows[batch*20:(batch+1)*20]
        canvas=Image.new('RGB',(columns*384,math.ceil(len(group)/columns)*244),'#171717')
        draw=ImageDraw.Draw(canvas)
        for j,row in enumerate(group):
            with Image.open(folder/row['path']) as image:
                im=ImageOps.contain(image.convert('RGB'),(384,216))
            x,y=j%columns*384,j//columns*244
            canvas.paste(im,(x+(384-im.width)//2,y+28))
            draw.text((x+3,y+3),f"frame {row['frame']} | {row['pts_seconds']:.3f}s",font=ImageFont.load_default(size=18),fill='white')
        canvas.save(folder/f'{prefix}_{batch:02d}.png',compress_level=1)

def overview(ID):
    folder=HERE/ID;folder.mkdir()
    frames=[];selected=[];target=0.0
    with av.open(SOURCE/f'{ID}.mp4') as container:
        stream=container.streams.video[0];stream.codec_context.thread_count=2
        for i,frame in enumerate(container.decode(stream)):
            row=dict(frame=i,native_pts=frame.pts,time_base_num=frame.time_base.numerator,
                     time_base_den=frame.time_base.denominator,pts_seconds=float(frame.pts*frame.time_base))
            frames.append(row)
            if row['pts_seconds']+1e-9>=target:
                name=f'overview_frame{i:06d}.png';frame.to_image().save(folder/name,compress_level=1)
                selected.append(dict(row,path=name));target=math.floor(row['pts_seconds'])+1.0
    put(folder/'native_pts.json',dict(ID=ID,source_path=str(SOURCE/f'{ID}.mp4'),source_sha256=sha(SOURCE/f'{ID}.mp4'),
                                    frame_count=len(frames),frames=frames,selected=selected))
    sheet(selected,folder,'overview')
    print(json.dumps(dict(ID=ID,count=len(frames),last_pts=frames[-1]['pts_seconds'],overview=len(selected))),flush=True)

def detail(ID,start,end,step,label):
    folder=HERE/ID;mapping=json.loads((folder/'native_pts.json').read_text(encoding='utf-8'))
    lookup={r['native_pts']:r for r in mapping['frames']}
    candidates=[r for r in mapping['frames'] if start<=r['frame']<=end]
    chosen={r['native_pts'] for r in candidates[::step]}
    if candidates: chosen.add(candidates[-1]['native_pts'])
    selected=[]
    with av.open(SOURCE/f'{ID}.mp4') as container:
        stream=container.streams.video[0];stream.codec_context.thread_count=2
        container.seek(candidates[0]['native_pts'],stream=stream,backward=True,any_frame=False)
        for frame in container.decode(stream):
            if frame.pts>candidates[-1]['native_pts']:break
            if frame.pts in chosen:
                row=lookup[frame.pts];name=f'native_frame{row["frame"]:06d}.png'
                if not (folder/name).exists():frame.to_image().save(folder/name,compress_level=1)
                selected.append(dict(row,path=name,sha256=sha(folder/name)))
    assert {r['native_pts'] for r in selected}==chosen
    put(folder/f'{label}_manifest.json',dict(source_sha256=mapping['source_sha256'],start_frame=start,end_frame=end,step=step,selected=selected))
    sheet(selected,folder,label)
    print(json.dumps(dict(ID=ID,label=label,selected=len(selected))),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['overview','detail']);p.add_argument('--ids',nargs='+',required=True)
    p.add_argument('--start',type=int);p.add_argument('--end',type=int);p.add_argument('--step',type=int,default=3);p.add_argument('--label',default='detail')
    a=p.parse_args()
    for ID in a.ids:
        assert ID in {'00014','00015','00016'}
        overview(ID) if a.mode=='overview' else detail(ID,a.start,a.end,a.step,a.label)
