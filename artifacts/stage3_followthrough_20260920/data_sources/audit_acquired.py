from pathlib import Path
import json,itertools,zlib,hashlib,collections
import numpy as np
O=Path(__file__).resolve().parent;R=O.parents[2];A=O/'acquired';read=lambda p:json.loads(p.read_text());write=lambda p,d:p.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n')
windows=read(A/'reserved_windows.json');manifest=read(A/'reserved_manifest.json');known=read(O/'local_sources.json');knownroutes={r['route'] for r in known if r['dataset']=='comma2k19'};knowndates={r.split('|')[1][:10] for r in knownroutes};records=[read(p) for p in (A/'records').glob('*.json')];metadata=read(O/'remote_segment_sizes.json');byseg={s['segment']:s for r in metadata for s in r['segments']};decode=read(A/'video_decode_checks.json')
assert len(records)==176 and all(r['status']=='aligned_sensor_only' for r in records)
assert len(windows)==18 and len(manifest['routes'])==2
for r in manifest['routes']:
 assert r['route'] not in knownroutes and r['route'].split('|')[1][:10] not in knowndates
 ww=[w for w in windows if w['route']==r['route']]
 assert len(ww)==9
 for triplet in range(3):
  t=[w for w in ww if w['triplet_index']==triplet];assert sorted(w['label'] for w in t)==[0,1,2]
  assert max(w['speed_m_s'] for w in t)-min(w['speed_m_s'] for w in t)<=2
 assert all(abs((x['start_boot_s']+x['end_boot_s']-y['start_boot_s']-y['end_boot_s'])/2)>=3-1e-7 for x,y in itertools.combinations(ww,2))
 for w in ww:
  z=np.load(A/'aligned'/(w['segment'].replace('/','_').replace('|','_')+'.npz'));i=w['center_index'];sl=slice(i-10,i+11);v=z['speed_smoothed'][sl];a=z['acceleration_proxy'][sl];s=z['steering_smoothed'][sl]
  assert all(np.isfinite(x).all() for x in [v,a,s]) and min(v)>=5 and max(abs(s))<=5
  assert min(a)>.5 if w['label']==0 else max(a)<-.5 if w['label']==1 else max(abs(a))<.1
 # Check freshly acquired members only; no old integrity/decode repeat.
 for seg in r['selected_segments']:
  for m in byseg[seg]['members']:
   p=A/'raw'/m['name'].replace('|','_');b=p.read_bytes();assert len(b)==m['uncompressed_bytes'] and zlib.crc32(b)==m['crc32']
assert all(r['decoded_frames']==r['frame_times_count']==1200 for r in decode)
review=[]
for w in windows:
 j=w['review_row'];night=j<9;close=j in [9,10,11,14,15]
 review.append({'review_row':j,'segment':w['segment'],'center_seconds':w['center_seconds'],'reviewer':'AI root+data expert; not human or official truth','review_scope':'start/middle/end native frames only','illumination':'night' if night else 'daylight_overcast','glare_reflection':'visible, particularly row3 and row7' if night else 'no strong glare evident in displayed samples','dynamic_object_occlusion':'nearby vehicles occupy substantial side/foreground area' if close else 'other moving vehicles present; static road/guardrail/trees partly visible','static_background_support':'road marks and some guardrail visible, weak distant texture' if night else 'trees, guardrail and road marks visible; masks/tracking not validated','visual_acceleration_sign_verified':False,'sensor_label_visual_contradiction':'no demonstrable contradiction; actual acceleration cannot be confirmed from three frames','proxy_generalization_role':'reserved sensor-proxy challenge, new route/date same vehicle','clean_geometry_readiness':'hold until automatic static-support and motion quality verified','action':'retain; do not relabel, discard or replace based on appearance'})
write(A/'visual_review.json',{'rows':review,'reviewed_windows':18,'displayed_frames':54,'independent_unique_frames':sum(len(r['selected_native_frames']) for r in decode),'limitations':['RAV4 all night vs Civic all cloudy daylight: vehicle and illumination confounded','Selected speed ranges differ by vehicle; no cross-vehicle speed matching','Sensor hardware latency and true CAN longitudinal acceleration equivalence unverified','Static background visible does not establish tracker/pose/geometry reliability','Three sampled frames cannot verify acceleration sign or precise sensor timing']})
manifest.update(video_review_status='AI sampled54frames; proxy use reserved, clean geometry readiness held',actual_video_clips=6,selected_windows=18,classes_per_route={'ACCELERATING':3,'DECELERATING':3,'CONSTANT':3},actual_acceleration_visually_verified=False,training_permitted_this_turn=False)
write(A/'reserved_manifest.json',manifest)
summary={'sensor_segments_acquired':176,'sensor_routes':18,'aligned_timestamps':sum(r['sample_count'] for r in records),'eligible_sensor_windows':sum(len(r['windows']) for r in records),'sensor_compressed_payload_bytes':sum(s['sensor_only_compressed_bytes'] for r in metadata for s in r['segments']),'video_segments_acquired':6,'video_routes':2,'video_compressed_payload_bytes':225031110,'video_decoded_frames':sum(r['decoded_frames'] for r in decode),'reserved_windows':18,'matched_triplets':6,'A_D_C_counts':[6,6,6],'new_route_and_date_ids':True,'new_vehicles':False,'public_remaining4_overlap_verified':False,'geo_road_independence_verified':False,'prior_model_predictions_on_new_sources':0,'training_on_new_sources':0,'new_official_labels':0,'clean_static_geometry_passed_windows':0,'clean_static_geometry_status':'Not tested; do not interpret0 as18 failed windows','visual_review_frames':54,'max_nearest_frame_time_error_s':max(r['max_query_frame_error_s'] for r in decode),'all_6_video_CRC_and_decode_counts_pass':True,'sensor_windows_constraints_pass':True,'final_audit_exit_status':0}
write(A/'summary.json',summary);write(A/'checks.json',{'passed':True,'new_source_crc_framecounts_and_constraints_checked':True,'old_data_full_checks_repeated':False,'source_policy':'reserved; not inserted into existing23 training splits','source_license_preserved':(A/'SOURCE_LICENSE').read_bytes()==(R/'external_data/comma2k19/LICENSE').read_bytes()});print(json.dumps(summary,ensure_ascii=False,indent=2))
