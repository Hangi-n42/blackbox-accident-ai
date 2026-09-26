"""Receive only the authorized six clips and inspect selected native frames."""
from pathlib import Path
import json,sys
from concurrent.futures import ThreadPoolExecutor
import numpy as np,av
from PIL import Image,ImageDraw
O=Path(__file__).resolve().parent;A=O/'acquired';R=O.parents[2];sys.path.insert(0,str(O))
from acquire_reserved import allsegs,download_member
read=lambda p:json.loads(p.read_text());write=lambda p,d:p.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n')
plan=read(A/'triplet_selection_plan.json');chosen=[r for r in plan['all_route_results'] if r['route'] in ['Chunk_1/b0c9d2329ad1606b|2018-08-10--22-42-26','Chunk_3/99c94dc769b5d96e|2018-05-18--16-00-42']]
segments=[s for r in chosen for s in r['selected_segments']];meta={s['segment']:s for s in allsegs};assert len(chosen)==2 and len(segments)==6
payload=sum(r['video_compressed_bytes'] for r in chosen);assert payload==225031110 and payload<=225100000
write(A/'budget_amendment.json',{'original_root_estimate_video_cap_bytes':200000000,'revised_authorized_video_cap_bytes':225100000,'selected_video_compressed_bytes':payload,'reason':'Root approved +25.1MB because minimum complete two-vehicle/two-route sensor-matched18 windows require six clips. Original200MB was root planning budget, not user cap. Sensor labels/selected windows unchanged; no model prediction selection.','authorization':'Parent agent message 2026-09-20'})
write(A/'reserved_manifest.json',{'split':'reserved_new_routes_same_known_vehicles','source_url':'https://huggingface.co/datasets/commaai/comma2k19','license_file':str((A/'SOURCE_LICENSE').relative_to(R)),'routes':chosen,'model_predictions_inspected':False,'official_truth':False,'video_review_status':'pending','sensor_hardware_latency_verified':False,'road_independence_verified':False,'other_public4_source_overlap':'unverified','video_compressed_bytes':payload})
def receive(segment):
 s=meta[segment];m=next(m for m in s['members'] if m['name'].endswith('/video.hevc'));d=download_member(s,m);print('received',segment,m['compressed_bytes'],flush=True);return d
with ThreadPoolExecutor(max_workers=3) as pool:receipts=list(pool.map(receive,segments))
write(A/'video_receipts.json',receipts)
windows=[dict(w,vehicle_model='RAV4' if r['vehicle'].startswith('b0') else 'Civic') for r in chosen for w in r['selected_windows']]
reviewdir=A/'video_review';reviewdir.mkdir(exist_ok=True);frames={};checks=[]
for seg in segments:
 base=A/'raw'/seg.replace('|','_');key=seg.replace('/','_').replace('|','_');z=np.load(A/'aligned'/(key+'.npz'));ft=np.load(base/'global_pose/frame_times').ravel();ws=[w for w in windows if w['segment']==seg];native_set={int(z['frame_index'][w['center_index']+di]) for w in ws for di in [-10,0,10]};count=0;times={};dims=None
 with av.open(str(base/'video.hevc')) as c:
  for i,f in enumerate(c.decode(video=0)):
   count+=1
   if i in native_set:
    im=f.to_image();dims=im.size;im.thumbnail((320,240));frames[(seg,i)]=im;times[i]=float(f.time) if f.time is not None else None
 assert count==len(ft),f'{seg}: decoded/clock count mismatch {count}/{len(ft)}'
 assert native_set=={i for s,i in frames if s==seg}
 checks.append({'segment':seg,'decoded_frames':count,'frame_times_count':len(ft),'width_height':dims,'selected_native_frames':sorted(native_set),'decoder_pts_seconds':times,'frame_times_strictly_increasing':bool(np.all(np.diff(ft)>0)),'max_query_frame_error_s':float(abs(z['time']-z['frame_time']).max()),'hardware_sensor_delay_verified':False})
 print('decoded',seg,count,len(native_set),flush=True)
panels=[]
for j,w in enumerate(windows):
 key=w['segment'].replace('/','_').replace('|','_');z=np.load(A/'aligned'/(key+'.npz'));im=Image.new('RGB',(960,282),'white');dr=ImageDraw.Draw(im)
 dr.text((4,3),f"{j:02d} {w['vehicle_model']} clip{w['segment'].rsplit('/',1)[1]} t={w['center_seconds']:.1f}s triplet{w['triplet_index']} class{w['label']} v={w['speed_m_s']:.2f} a={w['acceleration_proxy']:.3f}",fill='black')
 native=[]
 for k,delta in enumerate([-10,0,10]):
  fi=int(z['frame_index'][w['center_index']+delta]);im.paste(frames[(w['segment'],fi)],(k*320,42));native.append(fi)
 panels.append(im);w.update(review_row=j,review_page=f'contact_{j//6}.jpg',native_frame_indices=native)
for first in range(0,len(panels),6):
 page=panels[first:first+6];im=Image.new('RGB',(960,282*len(page)),'white')
 for row,p in enumerate(page):im.paste(p,(0,row*282))
 im.save(reviewdir/f'contact_{first//6}.jpg')
write(A/'video_decode_checks.json',checks);write(A/'reserved_windows.json',windows)
print('COMPLETE',len(segments),'clips',len(windows),'sensor_proxy_windows',flush=True)
