"""Create a public-only 10Hz validation copy using the CSV's frame/time mapping.

The released MP4 timestamps are inconsistent. Frame identity is preserved in a
sidecar mapping; original assets are never modified. This is development only.
"""
from pathlib import Path
from fractions import Fraction
import json
import av

ROOT=Path(__file__).resolve().parents[1]
def main():
    out=ROOT/'artifacts/public_eval_10hz/stage3/videos';out.mkdir(parents=True,exist_ok=True)
    records=[]
    for source in sorted((ROOT/'Baseline/data/stage3/videos').glob('*.mp4')):
        target=out/source.name
        mapping=[]
        with av.open(str(source)) as reader, av.open(str(target),'w') as writer:
            original=reader.streams.video[0]
            stream=writer.add_stream('libx264',rate=10)
            stream.width=original.width;stream.height=original.height
            stream.pix_fmt='yuv420p';stream.options={'crf':'18','preset':'fast'}
            for idx,frame in enumerate(reader.decode(video=0)):
                if idx%2:continue
                sample=len(mapping);mapping.append(idx)
                frame.pts=sample;frame.time_base=Fraction(1,10)
                for packet in stream.encode(frame):writer.mux(packet)
            for packet in stream.encode():writer.mux(packet)
        records.append({'source':str(source),'target':str(target),'source_fps_basis':'labels.csv frame_index/time_seconds=20',
                        'target_fps':10,'target_to_source_frame':mapping})
        print(source.name,len(mapping),flush=True)
    (out.parent/'frame_mapping.json').write_text(json.dumps(records,indent=2),encoding='utf-8')

if __name__=='__main__':main()
