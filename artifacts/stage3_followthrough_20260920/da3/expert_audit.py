"""Independent saved-pose arithmetic audit; never runs a neural model."""
from pathlib import Path
import json
import numpy as np
O=Path(__file__).resolve().parent
R=O.parents[2]
P=R/'artifacts/stage3_point_motion_20260919'
B=R/'artifacts/stage3_training_basis_20260917'
wins=json.loads((P/'windows.json').read_text())
rows=[]
for w in wins:
    z=np.load(B/w['labels_npz']); i=w['index']; truth=float(z['acceleration_proxy'][i]/z['speed_smoothed'][i])
    assert np.isclose(truth,w['sensor_ratio_a_v'],atol=1e-15,rtol=0)
    rows.append({'key':w['key'],'saved_sensor_q':w['sensor_ratio_a_v'],'center_sensor_q':truth,'window_mean_a_over_mean_v':float(np.mean(z['acceleration_proxy'][i-10:i+11])/np.mean(z['speed_smoothed'][i-10:i+11]))})
poses=[]
pose_files=list((O/'poses').glob('*.npz'))+list((O/'cpu_control').glob('*.npz'))
if (O/'cpu_crosscheck.npz').exists(): pose_files.append(O/'cpu_crosscheck.npz')
for p in sorted(pose_files):
    z=np.load(p); e=z['extrinsics']; h=np.broadcast_to(np.eye(4),(len(e),4,4)).copy();h[:,:3]=e;c=np.linalg.inv(h)[:,:3,3]
    err=float(abs(c-z['centers']).max()); assert err<1e-4
    rot=e[:,:3,:3]; ortho=float(abs(rot.transpose(0,2,1)@rot-np.eye(3)).max());assert ortho<1e-4
    t=z['time']-z['time'][len(e)//2];coef=np.polynomial.polynomial.polyfit(t,c,2);v=coef[1];a=2*coef[2]
    q=float(v@a/(v@v)) if v@v>1e-12 else None
    poses.append({'file':str(p.relative_to(O)),'center_inverse_maxerror':err,'rotation_orthogonality_maxerror':ortho,'q_independent':q,'midpoint_time':float(z['time'][len(e)//2])})
# Exact straight motion and a proper rigid similarity check, no sensor values fed.
t=np.linspace(-1,1,21); c=np.stack([4*t+.5*.8*t*t, np.full(21,2.), np.full(21,-3.)],axis=1)
def qfit(c):
    co=np.polynomial.polynomial.polyfit(t,c,2);return float(co[1]@(2*co[2])/(co[1]@co[1]))
rot=np.array([[0.,-1.,0.],[1.,0.,0.],[0.,0.,1.]])
q=qfit(c);assert abs(q-.2)<1e-10
assert abs(qfit(2.7*c@rot+np.array([11,13,-5]))-q)<1e-10
assert abs(qfit(c[::-1])+q)<1e-10
result={'sensor_center_verified':len(rows),'sensor_definition':'saved value equals central smoothed acceleration_proxy/speed_smoothed, not window means; no labels or gates changed','sensor_rows':rows,'saved_poses_audited':len(poses),'pose_rows':poses,'synthetic_q':q,'synthetic_similarity_and_reverse_passed':True,'snapshot_may_be_incomplete':not (O/'pilot_summary.json').exists()}
(O/'expert_arithmetic_audit.json').write_text(json.dumps(result,indent=2))
print(json.dumps({k:v for k,v in result.items() if k not in ['sensor_rows','pose_rows']},indent=2))
