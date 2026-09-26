"""Freeze eight development windows before new depth/flow measurements."""
from pathlib import Path
import json,hashlib,sys,platform
import numpy as np
from PIL import Image
O=Path(__file__).resolve().parent;R=O.parents[1];P=R/'artifacts/stage3_point_motion_20260919';D=R/'artifacts/stage3_followthrough_20260920/da3';Z=R/'artifacts/stage3_zod_20260920'
read=lambda p:json.loads(p.read_text())
write=lambda p,v:p.write_text(json.dumps(v,ensure_ascii=False,indent=2))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()

def main():
 assert not (O/'freeze.json').exists()
 keys=['expanded_11_225','expanded_14_215','expanded_19_275','expanded_14_555','expanded_11_525','expanded_09_335']
 source={r['key']:r for r in read(P/'windows.json')};manifest=[];protected=read(P/'freeze.json')['source_inputs_protected'].copy()
 for name in ['prepare.py','infer.py','geometry.py']:protected[str((O/name).relative_to(R))]=sha(O/name)
 for p in [D/'weights/model.safetensors',D/'weights/config.json',D/'smoke.py',P/'windows.json']:
  protected[str(p.relative_to(R))]=sha(p)
 for key in keys:
  r=source[key].copy();path=P/(key+'_frames.npz');data=np.load(path);lp=R/'artifacts/stage3_training_basis_20260917'/r['labels_npz'];lab=np.load(lp);ix=np.arange(r['index']-10,r['index']+11);v=lab['speed_smoothed'][ix]
  assert len(data['rgb'])==21 and np.isfinite(v).all() and np.all(v>1)
  r.update(dataset='comma',frames_path=str(path.relative_to(R)),sensor_logspeed_slope=float(np.polyfit(data['time']-data['time'][10],np.log(v),1)[0]),sensor_speed_series=v.tolist(),time=data['time'].tolist(),sensor_source=str(lp.relative_to(R)))
  manifest.append(r);protected[str(path.relative_to(R))]=sha(path);protected[str(lp.relative_to(R))]=sha(lp)
 for sid,index in [('000002',140),('000026',161)]:
  lp=Z/'labels'/(sid+'.npz');lab=np.load(lp);info=read(Z/'raw/sequences'/sid/'info.json');frames=info['camera_frames']['front_blur'];ix=np.arange(index-10,index+11);rgb=[];paths=[]
  for i in ix:
   path=Z/'raw'/frames[int(lab['frame_index'][i])]['filepath'];im=Image.open(path).convert('RGB');h=round(im.height*640/im.width);rgb.append(np.array(im.resize((640,h))));paths.append(str(path.relative_to(R)))
   protected[str(path.relative_to(R))]=sha(path)
  key=f'zod_{sid}_{index}';path=O/(key+'_frames.npz');np.savez_compressed(path,rgb=np.array(rgb),time=lab['frame_time'][ix],frame_index=lab['frame_index'][ix])
  v=lab['speed_smoothed'][ix];assert np.isfinite(v).all() and np.all(v>1)
  manifest.append({'key':key,'id':sid,'dataset':'zod','index':index,'label':int(lab['accel_candidate'][index]),'role':'development_diagnostic','frames_path':str(path.relative_to(R)),'sensor_source':str(lp.relative_to(R)),'sensor_ratio_a_v':float(lab['acceleration_proxy'][index]/lab['speed_smoothed'][index]),'sensor_logspeed_slope':float(np.polyfit(lab['frame_time'][ix]-lab['frame_time'][index],np.log(v),1)[0]),'sensor_speed_series':v.tolist(),'time':lab['frame_time'][ix].tolist(),'source_images':paths,'source_url':'https://zod.zenseact.com/sequences/','license':'CC BY-SA4.0'})
  protected[str(lp.relative_to(R))]=sha(lp);protected[str(path.relative_to(R))]=sha(path)
 write(O/'manifest.json',manifest)
 write(O/'freeze.json',{'selection':'First train and first held A/D from old sensor-selected windows; first train C and first held RAV4 C for vehicle diversity. Plus previous ZOD C01/C02 failure windows. All8 development exposed, not independent evaluation.',
 'inference':'Frozen existing DA3-SMALL revision/provenance, CPU FP32, threads2, official upper_bound_resize504.21frames/2s together. First chronological, reverse first restored, chronological middle. No framewise depth normalization, no supplied ground-truth pose/K/sensors. No extrinsic outputs consumed.',
 'depth_stability':'Compare same pixels across variants after ONE global positive depth scale per whole window (diagnostic only). Base confidence>=frame median, x.08-.92,y.25-.80. Per-frame log scale drift slope<=.02/s; max global-adjusted frame median abs<=.05; spatial residual abslog p90<=.15. Estimated fx/fy variation across base frames<=5percent and variant median difference<=5percent; principalpoint difference<=3px. Gates provisional engineering limits, not official labels.',
 'geometry':'DIS_FAST adjacent forward/backward flow on official processed RGB. Grid8px in x.08-.92/y.25-.80;7px patch std>=4;FB<=1px;positivefinite depth and base-depth-confidence>=median. Fixed clip K=median chronological21 K, also fixed for reverse/middle depth controls. Predicted lens distortion unavailable: distortion0 pinhole approximation, not verified calibration.',
 'pnp':'EPNP RANSAC300 iterations,2px threshold,.999 confidence,seed42;LM refine inliers. Forward and reverse PnP eachpair. Quality>=30inliers,>=.5ratio,median reproj<=1px,xspan>=.4w,yspan>=.2h;forward inlier depthconsistency median abslog<=.10;roundtrip translation norm/forwardnorm<=.25,rotation<=.5deg. Base accepted point identities fixed across depth variants; no threshold tuning.',
 'feature':'Relative step speed=norm(t)/native_frame_dt. q=OLS slope(log(relative_speed),midpoint_time) across passed pairs;>=16/20passed and>=3 in first/last4 pairs. Window scale cancels; per-frame scale drift does not. Sensor comparator same log-speed slope across existing21 smoothed sensor samples; center a/v also retained. No velocity labels changed.',
 'gate':'At least6/8 windows pass basegeometry,depth/K stability and all3 q within.02/s, with>=1A,>=1D,>=3C. All valid A/D q direction correct and abs error<=.03/s; all valid C absq<=.02/s; valid q MAE below zero-q baseline. Failure stops classifier expansion; success only permits broader source-separated feature validation, not adoption.',
 'mechanism_controls':'Depth-order/ref tests change depth only with sameK and same DIS matches. Backprojection/PnP synthetic tests with known geometry check formula, scale invariance, rotation, and injected depth-scale drift before interpreting images.',
 'limits':'No metric-speed claim, no proven static segmentation, unknown calibration/distortion, pretrained-depth prior and sensorproxy error. Short-correspondence quality is not true pose accuracy. No grid search/new model/download/production changes.',
 'sources':{'DA3':'https://github.com/ByteDance-Seed/Depth-Anything-3','PnP':'https://docs.opencv.org/4.10.0/d5/d1f/calib3d_solvePnP.html'},'protected':protected,'python':sys.version,'platform':platform.platform()})
 print('frozen8windows',[(r['key'],r['label'],round(r['sensor_logspeed_slope'],5)) for r in manifest],flush=True)
if __name__=='__main__':main()
