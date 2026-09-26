"""Freeze label-only split and single data contrast before first prediction."""
from qa import O,R,read,write
from pathlib import Path
import hashlib,json
import numpy as np
from PIL import Image,ImageDraw
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
qa=read(O/'labels/qa.json')
held_groups=['uniform_2022-06-15','uniform_2022-06-17','quebec_2022-04-28','quebec_2022-05-31']
rows=[]
for r in qa:
 role='heldout' if r['group'] in held_groups else 'train'
 if not r['structural_qa_pass']:role='excluded'
 rows.append({**r,'role':role,'labels_path':str((O/'labels'/(r['id']+'.npz')).relative_to(R)),
 'development_exposure_before_experiment':r['id'] in ['000000','000001','000002','000004','000005'],
 'exclusion_reason':None if role!='excluded' else 'failed sensor availability, minimum20 usable samples, raw interpolation reproduction, or pose-speed consistency; see qa fields'})
assert sum(r['role']=='train' for r in rows)==25
assert sum(r['role']=='heldout' for r in rows)==8
assert {r['group'] for r in rows if r['role']=='train'}.isdisjoint({r['group'] for r in rows if r['role']=='heldout'})
assert not (O/'freeze.json').exists()
protected=[R/'artifacts/submissions/verify_v6/model/stage3/motion_model.joblib',R/'releases/v7/source/model/stage3/motion_model.joblib',R/'releases/v7/source/model/stage2/code/solution/stage3.py']
write(O/'split_manifest.json',rows)
write(O/'freeze.json',{
 'scope':'Stage3 acceleration only; one data-only contrast; no production edits; no heldout tuning',
 'pilot_ids':['000000','000001','000002','000004','000005'],
 'pilot_initial_exclusions':['000006','000007'],
 'split':'25 training sequences,8 heldout sequences,17 rejected from first50 metadata/sensor screen; heldout four whole vehicle/date groups selected for A/D/C support before model predictions. New dates, same dataset camera domain; road independence not established.',
 'heldout_groups':held_groups,'training_ids':[r['id'] for r in rows if r['role']=='train'],'heldout_ids':[r['id'] for r in rows if r['role']=='heldout'],
 'labels':'Same comma centered11point1s speed slope and mean; A>.3,D<-.3,Cabs<.2 m/s² at speed>.4m/s; STOPspeed<.2;±.3s known-class boundary excluded. Evalstrict A>.5,D<-.5,Cabs<.1 atspeed>1,STOPspeed<.1, stable11point window. Thresholds are trial sensor proxies, not official labels.',
 'sensor_quality':'UTC GPSepoch+time+leapSeconds, explicit validNorth/validEast/validXY both bracket points, finite stdNorth/East<=.5m/s both points, gap<=.03s raw, <=.15s published ego; no extrapolation. Generic isValid semantics unknown, not reported as valid. Pose-v p99<.5m/s; raw interpolation at native metadata points difference<.001m/s; at least20 usable points.',
 'qa_draft_changes_before_predictions':'First draft compared twice-interpolated10Hz data instead of native metadata times; corrected. Draft std<=.1m/s rejected000002 despite CAN label agreement and speed discrepancy<=.115m/s; revised trial gross-error gate<=.5. Draft artifacts retained. This gate is not a sensor accuracy guarantee.',
 'camera':'Provider blur front camera, original FOV; nearest native frame to fixed10Hz grid, error<=.056s; no rectification or feature change. No sensor input to classifier.',
 'features':'Exactly existing DISFAST*10; grayscalewidth256;144 spatial features + existing5/15/31 averages and5/15 lag derivatives =864; cache only newZOD.',
 'model':'Same StandardScaler + LogisticRegression C=.03,class_weight=balanced,max_iter=2000,random_state=42; refit scaler on training only using unchanged recipe. No weight or hyperparameter search.',
 'contrast':'comma2395 vs samecomma+ZOD usable2Hz rows25videos. PublicLOVO diagnostic additionally same40 official labels both arms; old3date folds reused. Primary newZOD8 heldout evaluated once after fits. Existing productionc52ecd15 separate reference, not data-only baseline.',
 'metrics':'Public50 OOF S3=.7accelF1+.3fixedsteerF1; other sets acceleration proxyF1 only. Report 4classF1 incl absentSTOP zero, also observed3classF1, pergroup, confusion, new/fixed opposites and newwrong/corrected. No private or officialZOD S3 claim.',
 'decision':'One candidate only. Improvement claim requires positive publicS3 and newZOD proxyF1, no newpublic inversion, no newZOD opposite-rate increase; report date/vehicle regressions separately. No automatic adoption.',
 'protected':{str(p.relative_to(R)):sha(p) for p in protected},
 'label_files':{str((O/'labels'/(r['id']+'.npz')).relative_to(R)):sha(O/'labels'/(r['id']+'.npz')) for r in rows},
 'sources':['https://zod.zenseact.com/sequences/','https://github.com/zenseact/zod','https://github.com/commaai/comma2k19'],
 'license':'ZOD CC BY-SA4.0; Zenseact Open Dataset creators. Derived labels/manifests retain attribution and same dataset license. SDK MIT. Download authorization received by user; accesslink excluded from report.',
})
vis=O/'visual';vis.mkdir(exist_ok=True);evidence=[]
for sid in ['000000','000001','000002','000004','000005']:
 d=np.load(O/'labels'/(sid+'.npz'));info=read(O/'raw/sequences'/sid/'info.json');frames=info['camera_frames']['front_blur']
 indices=sorted(set([10,len(d['time'])//2,len(d['time'])-11,int(np.nanargmax(d['acceleration_proxy'])),int(np.nanargmin(d['acceleration_proxy']))]))
 sheet=Image.new('RGB',(400*len(indices),278),'white');draw=ImageDraw.Draw(sheet);rr=[]
 for j,k in enumerate(indices):
  path=O/'raw'/frames[int(d['frame_index'][k])]['filepath'];im=Image.open(path);assert im.size==(3848,2168);im=im.resize((400,225));sheet.paste(im,(j*400,53))
  draw.text((j*400+4,3),f"{sid} t={k/10:.1f}s v={d['speed'][k]:.2f}\na={d['acceleration_proxy'][k]:.3f} y={d['accel_candidate'][k]}",fill='black')
  rr.append({'sample_index':k,'native_frame_index':int(d['frame_index'][k]),'image':str(path.relative_to(R)),'time_utc':float(d['frame_time'][k])})
 sheet.save(vis/(sid+'.jpg'));evidence.append({'id':sid,'file':str((vis/(sid+'.jpg')).relative_to(R)),'frames':rr})
write(vis/'manifest.json',evidence)
print('frozen:train25,heldout8; pilot5 sheets')
