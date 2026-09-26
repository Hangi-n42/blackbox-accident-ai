"""Native decoded pixels selected by existing 10Hz frame map; lossless FFV1."""
from pathlib import Path
import json
import av,numpy as np
O=Path(__file__).resolve().parent;R=O.parents[1];V=O/'comma_input/videos';V.mkdir(parents=True,exist_ok=True)
for row in json.loads((O/'comma_cases.json').read_text()):
 d=np.load(O/row['labels_npz']);indices=d['frame_index'];assert np.all(np.diff(indices)>0)
 p=V/(row['id']+'.mkv');assert not p.exists();wanted=set(map(int,indices));count=0
 with av.open(str(R/row['raw_path'])) as src,av.open(str(p),'w') as out:
  stream=None
  for i,f in enumerate(src.decode(video=0)):
   if i not in wanted:continue
   a=f.to_ndarray(format='bgr24')
   if stream is None:stream=out.add_stream('ffv1',rate=10);stream.width=f.width;stream.height=f.height;stream.pix_fmt='bgr0'
   frame=av.VideoFrame.from_ndarray(a,format='bgr24');frame.pts=count
   for packet in stream.encode(frame):out.mux(packet)
   count+=1
   if count==len(indices):break
  for packet in stream.encode():out.mux(packet)
 assert count==len(indices)
 print(row['id'],count,flush=True)
