import json,hashlib
from pathlib import Path
root=Path.cwd();out=root/'artifacts/stage1_road_data_search_20260918'
orig=json.loads((out/'reds_source_probe.json').read_text())['frames']
vd=json.loads((root/'artifacts/data_pilot_20260916/vdmoire/manifest.json').read_text())
rows=[]
for o in orig:
 n=o['sequence'];g=f'frames/train/Reds/video_{n}';rec=[x for x in vd['frames'] if x['source_group']==g]
 for f in [o]+rec:
  assert hashlib.sha256((root/f['path']).read_bytes()).hexdigest()==f['sha256']
 rows.append({'source_id':f'REDS/train/{n:03d}','domain':{2:'street_with_vehicle',16:'urban_road_buses',45:'pedestrian_shopping_street'}[n],
 'original':o,'physical_recapture_frames':rec,'recapture_archive':vd['source_url'],
 'capture_setup_provider':'iPhoneXR + MacBook Pro','pair_status':'same scene visually confirmed; exact temporal correspondence NOT verified',
 'visual_review':'assistant image inspection, 2026-09-18; not human annotation',
 'split':'development_only','previously_exposed':True,'dashcam_mount_verified':False,
 'borderless_recapture_ready':False,'ready_for_final_evaluation':False})
(out/'candidate_pairs.json').write_text(json.dumps({'status':'source-linkage pilot, NOT completed evaluation dataset','groups':rows},indent=2))
print('verified',len(rows),'source groups;',sum(len(x['physical_recapture_frames']) for x in rows),'existing recapture frames; 3 new original frames')
