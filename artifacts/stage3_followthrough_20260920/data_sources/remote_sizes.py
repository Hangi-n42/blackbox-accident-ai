"""Fetch ZIP central-directory metadata only, with bounded streamed ranges."""
import io, json, zipfile
from pathlib import Path
import requests
OUT=Path(__file__).resolve().parent
heads=json.loads((OUT/'remote_head_checks.json').read_text())
candidates=json.loads((OUT/'unacquired_route_candidates.json').read_text())
required=['video.hevc','global_pose/frame_times','processed_log/CAN/speed/t','processed_log/CAN/speed/value','processed_log/CAN/steering_angle/t','processed_log/CAN/steering_angle/value']
class BoundedMetadata(io.RawIOBase):
 def __init__(self,url,size): self.url=url; self.size=size; self.pos=0; self.used=0; self.log=[]; self.cache={}
 def seekable(self): return True
 def readable(self): return True
 def tell(self): return self.pos
 def seek(self,offset,whence=0):
  self.pos=offset if whence==0 else self.pos+offset if whence==1 else self.size+offset
  return self.pos
 def read(self,n=-1):
  n=min(n if n>=0 else self.size-self.pos,self.size-self.pos)
  if n<=0:return b''
  start=self.pos; end=start+n-1
  if (start,n) not in self.cache:
   if n>4_000_000 or self.used+n>8_000_000: raise RuntimeError('Metadata request exceeds byte cap')
   with requests.get(self.url,params={'stage3_metadata_start':start,'stage3_metadata_end':end},headers={'Range':f'bytes={start}-{end}'},stream=True,timeout=45) as r:
    if r.status_code!=206 or r.headers.get('Content-Range')!=f'bytes {start}-{end}/{self.size}': raise RuntimeError(f'Refuse nonexact range status {r.status_code}')
    data=r.raw.read(n+1)
    if len(data)!=n: raise RuntimeError('Range byte length mismatch')
   self.used+=len(data); self.log.append({'start':start,'end':end,'bytes':len(data),'status':206}); self.cache[start,n]=data
  self.pos+=n
  return self.cache[start,n]
results=[]; logs=[]
for h in heads:
 remote=BoundedMetadata(h['url'],int(h['content_length']))
 with zipfile.ZipFile(remote) as z:
  infos={x.filename:x for x in z.infolist()}
  for c in candidates:
   if c['source_url']!=h['url']:continue
   segrows=[]
   for s in c['segments_with_required_members']:
    members=[{'name':s+'/'+p,'compressed_bytes':infos[s+'/'+p].compress_size,'uncompressed_bytes':infos[s+'/'+p].file_size,'compression_method':infos[s+'/'+p].compress_type,'header_offset':infos[s+'/'+p].header_offset,'crc32':infos[s+'/'+p].CRC} for p in required]
    segrows.append({'segment':s,'members':members,'compressed_bytes':sum(x['compressed_bytes'] for x in members),'sensor_only_compressed_bytes':sum(x['compressed_bytes'] for x in members if not x['name'].endswith('/video.hevc'))})
   results.append({'route':c['route'],'vehicle':c['vehicle'],'source_url':c['source_url'],'segments':segrows,'known_route_overlap':False,'known_date_overlap':False,'known_vehicle_overlap':True,'public_other4_overlap':'unverified'})
 logs.append({'url':h['url'],'metadata_bytes_read':remote.used,'requests':remote.log,'video_sensor_payload_bytes_read':0})
 print(h['url'],remote.used,flush=True)
chosen=[]
for v in sorted({r['vehicle'] for r in results}):
 route=sorted([r for r in results if r['vehicle']==v],key=lambda r:r['route'])[0]
 seg=sorted(route['segments'],key=lambda x:int(x['segment'].rsplit('/',1)[1]))[0]
 chosen.append({**{k:val for k,val in route.items() if k!='segments'},**seg})
plan={'selection_rule':'Earliest cached unexposed route per each existing vehicle, lowest numeric segment. No sensor/label suitability claim.','selected_segments':chosen,'total_compressed_payload_bytes':sum(r['compressed_bytes'] for r in chosen),'headers_transport_extra_bytes':'Not included; small ZIP local headers plus directory metadata','ADC_coverage_guaranteed':False,'source_independence_guaranteed':False,'read_scope':'Only ZIP index metadata; no video/sensor payload'}
for name,data in [('remote_segment_sizes.json',results),('remote_metadata_checks.json',logs),('minimal_two_clip_plan.json',plan)]: (OUT/name).write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')
assert len(chosen)==2 and len({c['vehicle'] for c in chosen})==2
print(json.dumps({k:v for k,v in plan.items() if k!='selected_segments'},indent=2),flush=True)
