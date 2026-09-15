"""Capture unchanged V5 motion scan components on three development clips only."""
import os,sys
sys.dont_write_bytecode=True
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='2'
from pathlib import Path
import json,hashlib,datetime,time,importlib.util,csv
import numpy as np
import cv2
ROOT=Path(__file__).resolve().parents[3];OUT=Path(__file__).resolve().parent;HERE=OUT.parent
BASE=ROOT/'artifacts/submissions/verify_v5/model/stage2/code/solution/stage2.py'
IDS=('00000','00003','00004')
EXPECTED={'00000':584,'00003':582,'00004':579}
REVIEWS={'00000':'NEXAR_REVIEW_00000_review_1789400071868.json','00003':'NEXAR_REVIEW_00003_review_1789444125537.json','00004':'NEXAR_REVIEW_00004_review_1789444066492.json'}
cv2.setNumThreads(2)
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def save(n,x):(OUT/n).write_text(json.dumps(x,indent=2,allow_nan=False),encoding='utf8')
def main():
    start=time.perf_counter();files={str(BASE):sha(BASE),str(Path(__file__)):sha(__file__)};records=[]
    for ID in IDS:
        folder=HERE/('nexar_dev00004_baseline_run' if ID=='00004' else 'nexar_baseline_run')
        fp=folder/'freeze.json';rp=folder/'report.json';reviewp=HERE/'user_reviews'/REVIEWS[ID]
        f=read(fp);r=read(rp);assert r['status']=='complete' and r['freeze_sha256']==sha(fp)
        video=next(v for v in f['videos'] if v['ID']==ID);trace=next(v for v in r['videos'] if v['ID']==ID)
        assert f['binding']['files']['model/stage2/code/solution/stage2.py']==sha(BASE)
        assert video['input_manifest_sha256']==trace['input_manifest_sha256']
        review=read(reviewp);contact=review['review']['contact']
        assert contact['status']=='observed' and contact['frame']==EXPECTED[ID]
        assert review['source_video_sha256']==video['source_sha256']==sha(video['source_path'])
        times={x['frame']:x['pts_seconds'] for x in video['source_frame_pts']}
        assert times[contact['frame']]==contact['pts_seconds']
        for p in (fp,rp,reviewp,Path(video['source_path'])):files[str(p)]=sha(p)
        for item in video['input_images']:assert sha(folder/item['path'])==item['file_sha256']
        records.append((ID,folder,video,trace,contact,times))
    save('plan_frozen.json',dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),files=files,ids=IDS,
      method='Unchanged extracted V5 _motion_scan called directly. Wrap _robust_scale to capture input/output; wrap Farneback to collect observational statistics without changing returned flow.',
      comparison='Exact float32 values against saved motion_scores before contribution interpretation',
      selection='Global argmax; top10 rows by score descending, earlier index on ties, no NMS; human contact nearest input and all native-PTS rows within +/-1 second.',
      limitations='Development diagnosis only. No candidate scores, parameter search, reserved data, model/GPU invocation or physical object identification.',
      diagnostics='Median core flow vector and magnitude; signed mean gray change, mean absolute change, fraction abs gray change>20. These do not identify wiper or camera cause.'))
    spec=importlib.util.spec_from_file_location('frozen_motion_base',BASE);base=importlib.util.module_from_spec(spec);spec.loader.exec_module(base)
    original_scale=base._robust_scale;original_flow=cv2.calcOpticalFlowFarneback
    outputs=[]
    for ID,folder,video,trace,contact,times in records:
        captures=[];observations=[]
        def scale(x):
            y=original_scale(x);captures.append((x.copy(),y.copy()));return y
        def flow(prev,gray,*args,**kwargs):
            ans=original_flow(prev,gray,*args,**kwargs)
            core=ans[8:76,8:152];shift=np.median(core.reshape(-1,2),axis=0)
            diff=gray.astype(np.float32)-prev
            observations.append(dict(shift_x=float(shift[0]),shift_y=float(shift[1]),shift_magnitude=float(np.linalg.norm(shift)),
              gray_mean=float(np.mean(gray)),signed_gray_mean_change=float(np.mean(diff)),gray_abs_mean_change=float(np.mean(np.abs(diff))),gray_abs_change_gt20_fraction=float(np.mean(np.abs(diff)>20))))
            return ans
        base._robust_scale=scale;cv2.calcOpticalFlowFarneback=flow
        paths=[folder/x['path'] for x in video['input_images']]
        t=time.perf_counter()
        try:valid,scores,side=base._motion_scan(paths)
        finally:base._robust_scale=original_scale;cv2.calcOpticalFlowFarneback=original_flow
        nums=[base._frame_number(p) for p in valid];assert nums==trace['frame_numbers']
        old=np.asarray(trace['motion_scores'],dtype=np.float32)
        assert len(captures)==3 and scores.dtype==old.dtype and np.array_equal(scores.view(np.uint32),old.view(np.uint32)),ID
        raw=np.stack([x[0] for x in captures],axis=1);scaled=np.stack([x[1] for x in captures],axis=1)
        contributions=np.stack((scaled[:,0],.6*scaled[:,1],.25*scaled[:,2]),axis=1)
        reconstructed=contributions[:,0]+contributions[:,1]+contributions[:,2];reconstructed[0]=0
        assert np.array_equal(reconstructed.view(np.uint32),scores.view(np.uint32))
        names=['jerk','residual90','appearance'];stats={}
        for j,n in enumerate(names):
            med=np.median(raw[:,j]);mad=np.median(np.abs(raw[:,j]-med));s=float(mad*1.4826)
            stats[n]=dict(median=float(med),mad=float(mad),raw_scale=s,denominator=max(s,1e-3),clipped_to10_count=int(np.sum(scaled[:,j]==10)),zero_count=int(np.sum(scaled[:,j]==0)))
        zero={k:0. for k in observations[0]};obs=[zero]+observations;assert len(obs)==len(nums)
        rows=[]
        order=sorted(range(len(nums)),key=lambda i:(-float(scores[i]),i));ranks={i:k+1 for k,i in enumerate(order)}
        for i,num in enumerate(nums):
            row=dict(frame=num,pts_seconds=times[num],delta_contact_seconds=times[num]-contact['pts_seconds'],score=float(scores[i]),rank=ranks[i],**obs[i])
            for j,n in enumerate(names):row[n+'_raw']=float(raw[i,j]);row[n+'_scaled']=float(scaled[i,j]);row[n+'_contribution']=float(contributions[i,j])
            rows.append(row)
        window=[r for r in rows if abs(r['delta_contact_seconds'])<=1.]
        nearest=min(rows,key=lambda r:(abs(r['delta_contact_seconds']),r['frame']))
        with (OUT/f'{ID}_components.csv').open('w',newline='',encoding='utf8') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
        result=dict(ID=ID,rows=len(rows),scores_bit_exact=True,score_sha256=hashlib.sha256(scores.tobytes()).hexdigest(),max_abs_saved_score_error=float(np.max(np.abs(scores-old))),
          robust_scaling=stats,human_contact=contact,global_max=rows[int(np.argmax(scores))],nearest_contact_input=nearest,
          contact_window_count=len(window),contact_window_max=max(window,key=lambda r:r['score']),contact_window_rows=window,
          top10_score_rows=[rows[i] for i in order[:10]],elapsed_seconds=time.perf_counter()-t)
        outputs.append(result);print(json.dumps(dict(ID=ID,exact=True,globalmax=result['global_max'],contact=nearest)),flush=True)
    assert all(sha(p)==h for p,h in files.items())
    save('report.json',dict(status='complete_exact_motion_component_audit',videos=outputs,files_unchanged=True,seconds=time.perf_counter()-start,training=0,model_calls=0,reserved_videos=0))
if __name__=='__main__':main()
