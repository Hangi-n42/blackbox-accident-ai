"""Bounded new-source intake. Sensor-first, no model predictions or prior split edits."""
from pathlib import Path
import sys, json, struct, zlib, time, hashlib, shutil, argparse
from concurrent.futures import ThreadPoolExecutor
import numpy as np, requests
from numpy.lib.stride_tricks import sliding_window_view
O=Path(__file__).resolve().parent; R=O.parents[2]; A=O/'acquired'; A.mkdir(exist_ok=True)
sys.path.insert(0,str(R/'scripts/data'))
from clean_comma_pilot import align
read=lambda p:json.loads(p.read_text())
write=lambda p,d:p.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n')
allroutes=read(O/'remote_segment_sizes.json'); allsegs=[dict(s,route=r['route'],vehicle=r['vehicle'],source_url=r['source_url']) for r in allroutes for s in r['segments']]
initial={r['segment'] for r in read(O/'minimal_two_clip_plan.json')['selected_segments']}
required=['global_pose/frame_times','processed_log/CAN/speed/t','processed_log/CAN/speed/value','processed_log/CAN/steering_angle/t','processed_log/CAN/steering_angle/value']
def fetch_range(url,start,end):
 for attempt in range(3):
  try:
   with requests.get(url,params={'s3reserved_start':start,'s3reserved_end':end,'retry':attempt},headers={'Range':f'bytes={start}-{end}'},stream=True,timeout=40) as q:
    if q.status_code!=206 or not q.headers.get('Content-Range','').startswith(f'bytes {start}-{end}/'):raise RuntimeError('nonexact range response')
    b=q.raw.read(end-start+2)
    if len(b)!=end-start+1: raise RuntimeError('range length mismatch')
    return b
  except Exception:
   if attempt==2: raise
   time.sleep(1)
def download_member(s,m):
 target=A/'raw'/m['name'].replace('|','_');target.parent.mkdir(parents=True,exist_ok=True)
 if target.exists():
  b=target.read_bytes();assert len(b)==m['uncompressed_bytes'] and zlib.crc32(b)==m['crc32'];return {'name':m['name'],'path':str(target.relative_to(R)),'cache_reused':True,'crc32':m['crc32']}
 off=m['header_offset'];h=fetch_range(s['source_url'],off,off+29);v=struct.unpack('<4s5H3I2H',h);assert v[0]==b'PK\x03\x04' and v[3]==m['compression_method']
 name_n,extra_n=v[-2:];body=fetch_range(s['source_url'],off+30,off+30+name_n+extra_n+m['compressed_bytes']-1)
 assert body[:name_n].decode('utf-8')==m['name']
 compressed=body[name_n+extra_n:];b=zlib.decompress(compressed,-15) if m['compression_method']==8 else compressed if m['compression_method']==0 else None
 assert b is not None and len(b)==m['uncompressed_bytes'] and zlib.crc32(b)==m['crc32']
 target.write_bytes(b)
 return {'name':m['name'],'path':str(target.relative_to(R)),'compressed_payload_bytes':m['compressed_bytes'],'transfer_bytes':30+len(body),'uncompressed_bytes':len(b),'crc32':m['crc32'],'sha256':hashlib.sha256(b).hexdigest(),'status':206,'cache_reused':False}
def process(s):
 base=A/'raw'/s['segment'].replace('|','_'); info={'segment':s['segment'],'route':s['route'],'vehicle':s['vehicle'],'source_url':s['source_url'],'classification':'reserved_new_route_same_vehicle_proxy_only','model_predictions_inspected':False}
 try:
  members=[m for m in s['members'] if not m['name'].endswith('/video.hevc')]
  records=[download_member(s,m) for m in members]
  ft=np.load(base/'global_pose/frame_times').ravel(); assert np.isfinite(ft).all() and np.all(np.diff(ft)>0)
  t=ft[0]+np.arange(int(np.floor((ft[-1]-ft[0])*10))+1)/10;right=np.searchsorted(ft,t).clip(1,len(ft)-1);frame=np.where(t-ft[right-1]<=ft[right]-t,right-1,right)
  d={'time':t,'frame_index':frame,'frame_time':ft[frame]}
  for name in ['speed','steering_angle']:
   x,mask,removed=align(np.load(base/f'processed_log/CAN/{name}/t'),np.load(base/f'processed_log/CAN/{name}/value'),t);d[name]=x;d[name+'_valid']=mask
  v=d['speed'];sangle=d['steering_angle'];a=np.full(len(t),np.nan);xx=np.arange(-5,6)*.1;a[5:-5]=(sliding_window_view(v,11)*xx).sum(axis=1)/(xx@xx)
  vs=np.full(len(t),np.nan);vs[5:-5]=sliding_window_view(v,11).mean(axis=1);ss=np.full(len(t),np.nan);ss[2:-2]=sliding_window_view(sangle,5).mean(axis=1)
  d.update(acceleration_proxy=a,speed_smoothed=vs,steering_smoothed=ss)
  lab=np.full(len(t),-1,np.int8);lab[(vs<.2)&np.isfinite(a)]=3;mv=vs>.4;lab[mv&(a>.3)]=0;lab[mv&(a<-.3)]=1;lab[mv&(abs(a)<.2)]=2
  changes=np.flatnonzero((lab[1:]!=lab[:-1])&(lab[1:]>=0)&(lab[:-1]>=0))+1
  for k in changes:lab[max(0,k-3):min(len(t),k+4)]=-1
  d['accel_candidate']=lab;d['accel_use_mask']=lab>=0
  name=s['segment'].replace('/','_').replace('|','_');dest=A/'aligned';dest.mkdir(exist_ok=True);np.savez_compressed(dest/(name+'.npz'),**d)
  windows=[]
  for i in range(15,len(t)-15,10):
   w=slice(i-10,i+11);av=a[w];vv=vs[w];sv=ss[w]
   if not all(np.isfinite(x).all() for x in [av,vv,sv]) or min(vv)<5 or max(abs(sv))>5:continue
   k=0 if min(av)>.5 else 1 if max(av)<-.5 else 2 if max(abs(av))<.1 else -1
   if k>=0: windows.append({'segment':s['segment'],'route':s['route'],'vehicle':s['vehicle'],'center_index':i,'center_seconds':i/10,'start_boot_s':float(t[i-10]),'end_boot_s':float(t[i+10]),'label':k,'speed_m_s':float(vs[i]),'acceleration_proxy':float(a[i]),'video_path':str((base/'video.hevc').relative_to(R)),'truth_type':'sensor_speed_derivative_proxy_trial_threshold_not_official','split':'reserved_no_model_prediction','video_reviewed':False})
  info.update(status='aligned_sensor_only',members=records,aligned_npz=str((dest/(name+'.npz')).relative_to(R)),window_counts={str(k):sum(w['label']==k for w in windows) for k in range(3)},windows=windows,native_frames=len(ft),sample_count=len(t),max_frame_time_error_s=float(abs(t-ft[frame]).max()),invalid_speed=int((~d['speed_valid']).sum()),invalid_steering=int((~d['steering_angle_valid']).sum()),sensor_hardware_latency_verified=False)
 except Exception as e: info.update(status='error',error=str(e))
 result=A/'records';result.mkdir(exist_ok=True);write(result/(s['segment'].replace('/','_').replace('|','_')+'.json'),info);return info
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--phase',choices=['initial','all'],required=True);args=p.parse_args()
 segs=[s for s in allsegs if args.phase=='all' or s['segment'] in initial]
 assert sum(s['sensor_only_compressed_bytes'] for s in allsegs)<=12_200_000
 if not (A/'freeze.json').exists():write(A/'freeze.json',{'sensor_payload_cap':12_200_000,'video_payload_cap':200_000_000,'selection':'same point-motion sensor conditions; fixed 2sec windows/1sec candidate stride; same route, all chosen speeds within2m/s; centers >=3sec apart; 3each A/D/C; minimize distinct video segments; no model predictions','new_assets_only':True,'official_labels':False,'source_url':'https://huggingface.co/datasets/commaai/comma2k19','license':'MIT comma.ai2018'})
 shutil.copyfile(R/'external_data/comma2k19/LICENSE',A/'SOURCE_LICENSE')
 with ThreadPoolExecutor(max_workers=10) as pool:
  for i,res in enumerate(pool.map(process,segs)):
   print(i+1,len(segs),res['segment'],res['status'],res.get('window_counts'),flush=True)
