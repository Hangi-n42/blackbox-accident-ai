import sys,json,time,hashlib
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import av
from threadpoolctl import threadpool_limits
from solution.stage3 import predict_stage3
root=Path(__file__).resolve().parents[1];data=root/'artifacts/public_eval_10hz/stage3';model=root/'model/stage3'
expected={}
for path in sorted((data/'videos').glob('*.mp4')):
    with av.open(str(path)) as container:expected[path.stem]=sum(1 for _ in container.decode(video=0))
t=time.perf_counter()
with threadpool_limits(limits=2):result=predict_stage3(data,model)
elapsed=time.perf_counter()-t;counts=result.groupby('ID').size().to_dict()
checks={'expected_decoded_counts':expected,'actual_counts':counts,'all_decoded_frames_returned':expected==counts,'all_indices_contiguous':all(rows.sample_index.tolist()==list(range(len(rows))) for _,rows in result.groupby('ID')),'allowed_labels':bool(result.accel_label.isin(['ACCELERATING','DECELERATING','CONSTANT','STOPPED']).all() and result.steer_label.isin(['LEFT','STRAIGHT','RIGHT']).all()),'elapsed_seconds':elapsed,'model_sha256':hashlib.file_digest((model/'motion_model.joblib').open('rb'),'sha256').hexdigest(),'scope':'Actual predict_stage3 on public videos converted to 10Hz; these are training-fit predictions, not OOF accuracy.'}
assert all(checks[k] for k in ('all_decoded_frames_returned','all_indices_contiguous','allowed_labels'))
result.to_csv(root/'research/stage3_full10hz_predictions.csv',index=False)
(root/'research/stage3_full10hz_contract.json').write_text(json.dumps(checks,indent=2),encoding='utf-8');print(json.dumps(checks),flush=True)
