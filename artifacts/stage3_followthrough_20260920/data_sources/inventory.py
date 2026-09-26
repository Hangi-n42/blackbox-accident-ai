"""Metadata-only source/exposure inventory. No decode, full hashing or downloads."""
from pathlib import Path
import json, collections
ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
read = lambda p: json.loads((ROOT / p).read_text())
cases = read('artifacts/stage3_training_basis_20260917/cases.json')
by_path = {x['raw_path']: x for x in cases}
overlap = read('research/v4_stage3/overlap_audit.json')
matched = overlap['matched_raw_segment']
required = ['video.hevc', 'global_pose/frame_times', 'processed_log/CAN/speed/t', 'processed_log/CAN/speed/value', 'processed_log/CAN/steering_angle/t', 'processed_log/CAN/steering_angle/value']
local = []
for manifest in ['external_data/comma2k19/chunk1_manifest.json','external_data/comma2k19/chunk3_manifest.json']:
 for row in read(manifest):
  base = row['local'].replace('\\', '/')
  video = base + '/video.hevc'
  case = by_path.get(video)
  status = 'development_exposed' if case else 'public_OPEN_001_duplicate' if row['segment'] == matched else 'unresolved'
  local.append({'dataset':'comma2k19','video':video,'segment':row['segment'],'route':row['route'], 'bytes':(ROOT/video).stat().st_size if (ROOT/video).exists() else None,'required_files_present':{p:(ROOT/base/p).is_file() for p in required},'classification':status,'case_id':case['id'] if case else None,'exposure_evidence':'artifacts/stage3_training_basis_20260917/cases.json' if case else 'research/v4_stage3/overlap_audit.json','source_url':row['source'],'license_file':'external_data/comma2k19/LICENSE','unused_independent':False})
actual = {str(p.relative_to(ROOT)) for p in (ROOT/'external_data/comma2k19').rglob('video.hevc')}
assert actual == {x['video'] for x in local}, 'Local manifest misses files; inspect before proceeding'
assert len(by_path)==23 and set(by_path)<=actual
assert collections.Counter(x['classification'] for x in local)=={'development_exposed':23,'public_OPEN_001_duplicate':1}
anaid=[]
for r in read('artifacts/data_readiness_20260917/anaid_qa.json'):
 video=ROOT/r['path']; csv=video.with_suffix('.csv')
 anaid.append({'dataset':'ANAID','video':r['path'],'video_exists':video.is_file(),'sensor_csv':str(csv.relative_to(ROOT)),'csv_exists':csv.is_file(),'frame_count_previous_qa':r['frames'],'sensor_rows_previous_qa':r['csv_rows'],'csv_has_timestamp':r['csv_has_timestamp'],'training_status':r['training_status'],'incomplete_required_rows':r['incomplete_required_rows'],'classification':'alignment_unproven_prior_QA_exposed','unused_independent':False,'evidence':'artifacts/data_readiness_20260917/anaid_qa.json'})
assert len(anaid)==5 and all(x['video_exists'] and x['csv_exists'] for x in anaid)
seen_routes={x['route'] for x in local}
remote=[]
for chunk in [1,3]:
 members=read(f'external_data/comma2k19/chunk{chunk}_members.json'); ms=set(members)
 segments=[p.rsplit('/',1)[0] for p in members if p.endswith('/video.hevc')]
 groups=collections.defaultdict(list)
 for segment in segments: groups[segment.rsplit('/',1)[0]].append(segment)
 for route, segments in sorted(groups.items()):
  if route in seen_routes: continue
  complete=[s for s in sorted(segments) if all(s+'/'+p in ms for p in required)]
  remote.append({'route':route,'vehicle':route.split('/')[1].split('|')[0],'source_url':f'https://huggingface.co/datasets/commaai/comma2k19/resolve/main/raw_data/Chunk_{chunk}.zip','cached_members_evidence':f'external_data/comma2k19/chunk{chunk}_members.json','segments_in_cached_catalog':len(segments),'segments_with_required_members':complete,'local_acquired':False,'known_route_id_overlap':False,'independence_verified':False,'ADC_speed_coverage':'unknown_until_sensor_retrieval','note':'Unacquired catalog candidate, not usable/local or proven independent. Geographic/continuous-drive overlap and other public videos remain unverified.'})
other={}
for name in ['Baseline/data/stage3','research','artifacts/data_pilot_20260916']:
 vids=[str(p.relative_to(ROOT)) for p in (ROOT/name).rglob('*') if p.is_file() and p.suffix.lower() in {'.mp4','.hevc','.mov','.avi','.mkv'}]
 other[name]={'video_count':len(vids),'videos':vids,'note':'Public/development/derived or non-sensor dataset files; no new paired timestamped sensor source identified in current dataset inventories.'}
summary={'local_comma_raw_sources':len(local),'already_development_used':23,'known_public_duplicate':1,'local_anaid_pairs':len(anaid),'anaid_alignment_hold':len(anaid),'new_usable_sensor_video_sources':0,'new_independent_matched_windows':0,'unacquired_route_candidates_from_cached_catalog':len(remote),'unacquired_segment_candidates_with_required_members':sum(len(x['segments_with_required_members']) for x in remote),'inspection':'File existence/stat, source IDs, prior exposure and QA manifests. No decoding or full integrity checks repeated. No new sensor/label generation, training, downloads.'}
for name,data in [('local_sources.json',local+anaid),('unacquired_route_candidates.json',remote),('other_video_inventory.json',other),('summary.json',summary),('matched_windows.json',{'windows':[],'status':'blocked_no_new_usable_sensor_video_source','criteria':'Different from all known exposed routes, synchronized sensors/video, same-speed A/D/C nonoverlap; not filled from exposed or unsynchronized cases.'})]:
 (OUT/name).write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(summary,ensure_ascii=False,indent=2))
