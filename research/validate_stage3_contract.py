import sys,json,time,argparse
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import cv2,numpy as np
from solution.stage3 import predict_stage3
root=Path(__file__).resolve().parents[1];base=root/'research/stage3_contract'
parser=argparse.ArgumentParser();parser.add_argument('--checkpoint');args=parser.parse_args()
model_dir=root/'solution/model/stage3'
if args.checkpoint:
    model_dir=base/'model';model_dir.mkdir(parents=True,exist_ok=True)
    (model_dir/'motion_model.joblib').write_bytes((root/args.checkpoint).read_bytes())
for group in ('alone','combined'):(base/group/'videos').mkdir(parents=True,exist_ok=True)
def create(target,source,limit):
    cap=cv2.VideoCapture(str(source));out=cv2.VideoWriter(str(target),cv2.VideoWriter_fourcc(*'mp4v'),10.,(256,192));count=0
    while count<limit*2:
        ok,frame=cap.read()
        if not ok:break
        if count%2==0:out.write(cv2.resize(frame,(256,192)))
        count+=1
    cap.release();out.release()
    return (count+1)//2
n=create(base/'alone/videos/Z.mp4',root/'Baseline/data/stage3/videos/OPEN_001.mp4',31)
(base/'combined/videos/A.mp4').write_bytes((base/'alone/videos/Z.mp4').read_bytes())
create(base/'combined/videos/Q.mp4',root/'Baseline/data/stage3/videos/OPEN_004.mp4',17)
t=time.perf_counter();a=predict_stage3(base/'alone',model_dir);b=predict_stage3(base/'combined',model_dir)
checks={'expected_frames':n,'actual_frames':len(a),'all_indices':a.sample_index.tolist()==list(range(n)),'finite_allowed_accel':a.accel_label.isin(['ACCELERATING','DECELERATING','CONSTANT','STOPPED']).all().item(),'finite_allowed_steer':a.steer_label.isin(['LEFT','STRAIGHT','RIGHT']).all().item(),'independent_of_filename_and_other_files':a[['sample_index','accel_label','steer_label']].reset_index(drop=True).equals(b[b.ID=='A'][['sample_index','accel_label','steer_label']].reset_index(drop=True)),'elapsed_seconds':time.perf_counter()-t}
assert all(checks[k] for k in ('all_indices','finite_allowed_accel','finite_allowed_steer','independent_of_filename_and_other_files'))
(base/'result.json').write_text(json.dumps(checks,indent=2),encoding='utf-8');print(json.dumps(checks))
