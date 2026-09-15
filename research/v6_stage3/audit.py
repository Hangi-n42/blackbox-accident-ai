"""Direct fixed CAN-proxy/public-GT audit only. No models, training or label search."""
from pathlib import Path
import os,sys
sys.dont_write_bytecode=True
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[key]='2'
import json,time,hashlib,traceback
import numpy as np,pandas as pd,cv2,scipy
from scipy.ndimage import uniform_filter1d
from threadpoolctl import threadpool_limits
ROOT=Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parent
SEGMENT='Chunk_1/b0c9d2329ad1606b|2018-07-27--06-03-57/10'
ACCEL=['ACCELERATING','DECELERATING','CONSTANT','STOPPED']
STEER=['LEFT','STRAIGHT','RIGHT']

def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def save(name,value):(OUT/name).write_text(json.dumps(value,indent=2,default=str),encoding='utf8')

def freeze():
    if (OUT/'plan_frozen.json').exists():raise FileExistsError('Preserve the completed or started audit')
    manifest=ROOT/'external_data/comma2k19/chunk1_manifest.json'
    record=next(r for r in json.loads(manifest.read_text()) if r['segment']==SEGMENT)
    base=ROOT/record['local'];video=ROOT/'Baseline/data/stage3/videos/OPEN_001.mp4'
    files=[video,ROOT/'Baseline/data/stage3/labels.csv',*sorted(base.rglob('*')),
      manifest,ROOT/'external_data/comma2k19/chunk1_members.json',ROOT/'external_data/comma2k19/inventory.json',
      ROOT/'external_data/comma2k19/LICENSE',ROOT/'research/v4_stage3/audit_report.json',ROOT/'research/v4_stage3/audit_plan_frozen.json',
      ROOT/'research/v4_stage3/overlap_audit.json',ROOT/'research/v4_stage3/public_can_proxy_comparison.csv',
      ROOT/'research/stage3_external_overlap.json',ROOT/'research/stage3_temporal_experiment.py',ROOT/'대회_통합_정보.md',Path(__file__)]
    files=[p for p in files if p.is_file()]
    protected=[ROOT/'solution/stage3.py',ROOT/'solution/stage3_fast.py',ROOT/'solution/stage3_v5_compatible.py',ROOT/'model/stage3/motion_model.joblib']
    plan={'timestamp_unix':time.time(),'segment':SEGMENT,'public_id':'OPEN_001','source_manifest_record':record,
      'inputs_and_code_sha256':{str(p.relative_to(ROOT)):sha(p) for p in files},
      'protected_stage3_sha256':{str(p.relative_to(ROOT)):sha(p) for p in protected},
      'existing_zip_stat':{str(p.relative_to(ROOT)):{'bytes':p.stat().st_size,'mtime_ns':p.stat().st_mtime_ns} for p in (ROOT/'artifacts/submissions').glob('*.zip')},
      'fixed_policy':{'speed_stop_m_s':.3,'acceleration_m_s2':.25,'steering_degrees':2.,
        'speed_filter_samples':21,'steering_filter_samples':11,'filter_mode':'reflect','filter_origin':0,
        'interpolation':'numpy.interp at actual frame_times, endpoint hold default',
        'acceleration':'numpy.gradient(filtered_speed, frame_times, edge_order=1)',
        'label_join':'public frame_index directly into matching original20Hz frame_times; no timestamp shift or search'},
      'reuse':'V4 full decoded pixel match and time/overlap audit are prior known evidence. Reconfirm only the two matching raw-file decoded hashes, not five-video feature extraction.',
      'prior_results_already_known':True,'not_holdout':True,'public_scope':'Only10 sparse official labels of OPEN_001; all other public files excluded without exact source proof.',
      'comparison':'Canonical proxy versus existing public GT. Numeric interpolated-versus-filtered values only; no alternative proxy labels or new thresholds.',
      'no_model_load_or_inference':True,'no_training':True,'no_candidate_search':True,'no_download':True,'gpu':False,'cpu_threads':2,
      'versions':{'numpy':np.__version__,'scipy':scipy.__version__,'opencv':cv2.__version__,'python':sys.version}}
    save('plan_frozen.json',plan);return plan,base,video

def decoded_hash(path):
    cap=cv2.VideoCapture(str(path));digest=hashlib.sha256();n=0;shape=None
    try:
        while True:
            ok,frame=cap.read()
            if not ok:break
            digest.update(frame.data);n+=1;shape=frame.shape
    finally:cap.release()
    return {'rows':n,'shape':shape,'decoded_bgr_sha256':digest.hexdigest()}

def signal_inventory(base):
    index=ROOT/'external_data/comma2k19/chunk1_members.json'
    members=[p for p in json.loads(index.read_text()) if p.startswith(SEGMENT+'/')]
    local=[str(p.relative_to(base)) for p in base.rglob('*') if p.is_file()]
    can=[p[len(SEGMENT)+1:] for p in members if '/CAN/' in p]
    accel=[p[len(SEGMENT)+1:] for p in members if 'accel' in p.lower()]
    return {'local_files':local,'cached_archive_CAN_members':can,'cached_archive_acceleration_named_members':accel,
      'explicit_longitudinal_CAN_acceleration_file_local':False,
      'explicit_longitudinal_CAN_acceleration_named_member':any('/CAN/' in p and 'accel' in p.lower() for p in members),
      'IMU_accelerometer_listed':any('/IMU/accelerometer/' in p for p in members),
      'IMU_accelerometer_downloaded':(base/'processed_log/IMU/accelerometer/value').exists(),
      'limits':'Archive member names prove listed paths only. Raw CAN contents were not decoded; presence or derivability of an acceleration signal inside raw_can/raw_log is unverified. IMU acceleration is not declared equivalent to official CAN longitudinal acceleration.',
      'source':'https://huggingface.co/datasets/commaai/comma2k19/resolve/main/raw_data/Chunk_1.zip',
      'cached_member_index_sha256':sha(index),'license':'MIT; local LICENSE hash frozen'}

def confusion(gt,pred,names):
    matrix=np.zeros((len(names),len(names)),int)
    for a,b in zip(gt,pred):matrix[names.index(a),names.index(b)]+=1
    return {'order':names,'rows_GT_columns_proxy':matrix.tolist(),'n':len(gt),'agreements':int(np.sum(gt==pred)),
      'disagreements':int(np.sum(gt!=pred)),'agreement_fraction':float(np.mean(gt==pred)),
      'GT_support':{v:int(np.sum(gt==v)) for v in names},'proxy_support':{v:int(np.sum(pred==v)) for v in names}}

def nearest_distance(t,q):
    i=np.searchsorted(t,q);left=np.maximum(i-1,0);right=np.minimum(i,len(t)-1)
    return np.minimum(np.abs(t[left]-q),np.abs(t[right]-q))

def main():
    plan,base,video=freeze();started=time.perf_counter();cv2.setNumThreads(2)
    try:
        prior=json.loads((ROOT/'research/v4_stage3/audit_report.json').read_text())
        overlap=json.loads((ROOT/'research/v4_stage3/overlap_audit.json').read_text())
        assert prior['public_can']['segment']==SEGMENT and prior['public_can']['full_decoded_pixel_sequence_matches_public']
        assert overlap['matched_raw_segment']==SEGMENT and not overlap['known_match_in_training'] and not overlap['known_match_route_in_training']
        expected=prior['public_can']['external_decoding']['decoded_bgr_sequence_sha256']
        pubhash=decoded_hash(video);exthash=decoded_hash(base/'video.hevc')
        assert pubhash==exthash and pubhash['decoded_bgr_sha256']==expected and pubhash['rows']==1200
        save('exact_source_match.json',{'public':pubhash,'external':exthash,'matches_prior_V4_pixel_sha':True,'known_segment_and_route_excluded_from_external_training':True})
        inventory=signal_inventory(base);save('signal_inventory.json',inventory)
        ft=np.load(base/'global_pose/frame_times').ravel()
        st=np.load(base/'processed_log/CAN/speed/t').ravel();sv=np.load(base/'processed_log/CAN/speed/value').ravel()
        at=np.load(base/'processed_log/CAN/steering_angle/t').ravel();av=np.load(base/'processed_log/CAN/steering_angle/value').ravel()
        assert len(ft)==1200 and np.all(np.diff(ft)>0) and np.all(np.diff(st)>=0) and np.all(np.diff(at)>=0)
        assert all(np.isfinite(v).all() for v in (ft,st,sv,at,av))
        speed_raw=np.interp(ft,st,sv);angle_raw=np.interp(ft,at,av)
        speed=uniform_filter1d(speed_raw,size=21,mode='reflect',origin=0)
        acceleration=np.gradient(speed,ft,edge_order=1)
        angle=uniform_filter1d(angle_raw,size=11,mode='reflect',origin=0)
        proxy_a=np.array(ACCEL)[np.where(speed<.3,3,np.where(acceleration>.25,0,np.where(acceleration<-.25,1,2)))]
        proxy_s=np.array(STEER)[np.where(angle>2.,0,np.where(angle<-2.,2,1))]
        frame=pd.read_csv(ROOT/'Baseline/data/stage3/labels.csv');frame=frame.loc[frame.ID=='OPEN_001'].copy()
        assert len(frame)==10 and frame.frame_index.is_unique
        ix=frame.frame_index.to_numpy();assert np.all(ix>=0) and np.all(ix<len(ft))
        assert np.array_equal(frame.sample_index.to_numpy()*2,ix)
        frame['can_boot_time']=ft[ix];frame['relative_can_time']=ft[ix]-ft[0]
        frame['nominal_minus_CAN_seconds']=frame.time_seconds.to_numpy()-(ft[ix]-ft[0])
        frame['speed_interpolated_raw_m_s']=speed_raw[ix];frame['speed_filtered_m_s']=speed[ix]
        frame['speed_filter_change_m_s']=speed[ix]-speed_raw[ix];frame['acceleration_proxy_m_s2']=acceleration[ix]
        frame['steering_interpolated_raw_degrees']=angle_raw[ix];frame['steering_filtered_degrees']=angle[ix]
        frame['steering_filter_change_degrees']=angle[ix]-angle_raw[ix]
        frame['speed_nearest_CAN_sample_seconds']=nearest_distance(st,ft[ix]);frame['steer_nearest_CAN_sample_seconds']=nearest_distance(at,ft[ix])
        frame['speed_endpoint_hold']=((ft[ix]<st[0])|(ft[ix]>st[-1]))
        frame['steer_endpoint_hold']=((ft[ix]<at[0])|(ft[ix]>at[-1]))
        frame['proxy_accel']=proxy_a[ix];frame['proxy_steer']=proxy_s[ix]
        frame['accel_agrees']=frame.accel_label==frame.proxy_accel;frame['steer_agrees']=frame.steer_label==frame.proxy_steer
        frame.to_csv(OUT/'public_proxy_rows.csv',index=False,encoding='utf-8')
        old=pd.read_csv(ROOT/'research/v4_stage3/public_can_proxy_comparison.csv')
        checks={}
        for oldname,newname in [('can_boot_time','can_boot_time'),('relative_can_time','relative_can_time'),('speed_m_s','speed_filtered_m_s'),('acceleration_m_s2','acceleration_proxy_m_s2'),('steering_degrees','steering_filtered_degrees')]:
            err=float(np.max(np.abs(old[oldname].to_numpy()-frame[newname].to_numpy())));checks[oldname]=err;assert err<=1e-9
        assert old.proxy_accel.tolist()==frame.proxy_accel.tolist() and old.proxy_steer.tolist()==frame.proxy_steer.tolist()
        ya=frame.accel_label.to_numpy();ys=frame.steer_label.to_numpy();keep=ya!='STOPPED'
        ca=confusion(ya,frame.proxy_accel.to_numpy(),ACCEL);cs=confusion(ys[keep],frame.proxy_steer.to_numpy()[keep],STEER)
        signals={'frame_count':len(ft),'speed_CAN_sample_count':len(st),'steering_CAN_sample_count':len(at),
          'frame_dt_min_median_max_seconds':np.quantile(np.diff(ft),[0,.5,1]).tolist(),
          'speed_dt_min_median_max_seconds':np.quantile(np.diff(st),[0,.5,1]).tolist(),'steering_dt_min_median_max_seconds':np.quantile(np.diff(at),[0,.5,1]).tolist(),
          'max_public_nominal_CAN_time_error_seconds':float(frame.nominal_minus_CAN_seconds.abs().max()),
          'speed_full_endpoint_hold_frames':int(np.sum((ft<st[0])|(ft>st[-1]))),'steer_full_endpoint_hold_frames':int(np.sum((ft<at[0])|(ft>at[-1]))),
          'selected_speed_endpoint_hold_rows':int(frame.speed_endpoint_hold.sum()),'selected_steer_endpoint_hold_rows':int(frame.steer_endpoint_hold.sum()),
          'official_continuous_signal_values_available':False,'physical_signal_error_against_official_CAN':'not computable: official continuous values not provided'}
        assert all(sha(ROOT/p)==value for p,value in plan['inputs_and_code_sha256'].items())
        assert all(sha(ROOT/p)==value for p,value in plan['protected_stage3_sha256'].items())
        assert all((ROOT/p).stat().st_size==v['bytes'] and (ROOT/p).stat().st_mtime_ns==v['mtime_ns'] for p,v in plan['existing_zip_stat'].items())
        report={'status':'direct_proxy_GT_audit_complete_no_model_change','matched_source':SEGMENT,'public_rows':10,
          'accel':ca,'steer':cs,'GT_stopped_masked_steer_rows':int((~keep).sum()),'signals':signals,
          'numeric_reproduction_max_error_against_prior_csv':checks,'prior_proxy_labels_exactly_reproduced':True,
          'mismatches':frame.loc[~frame.accel_agrees|~frame.steer_agrees].to_dict('records'),
          'macro_f1_or_composite_score_not_used':'Three acceleration classes have zero GT support; report direct counts/confusion, not a model performance score.',
          'new_holdout':False,'models_loaded':0,'predictions_from_models':0,'training_runs':0,'threshold_candidates':1,
          'threshold_candidates_note':'Only preexisting canonical policy, no new search or estimation.',
          'inputs_code_assets_unchanged':True,'seconds':time.perf_counter()-started,
          'limits':['Repeated known10 sparse GT rows, not independent validation.','All acceleration GT is CONSTANT; other classes and transition timing untested.','One steering disagreement does not identify a hidden threshold, filter or delay.','Raw CAN acceleration content not inspected; IMU is not CAN longitudinal acceleration.','No cause of official score is identified, no immediate model change required.']}
    except Exception:
        save('audit_report.json',{'status':'audit_blocked','error':traceback.format_exc(),'models_loaded':0,'training_runs':0});raise
    save('audit_report.json',report)
    print(json.dumps({'status':report['status'],'accel':ca,'steer':cs,'signals':signals,'seconds':report['seconds']},indent=2),flush=True)

if __name__=='__main__':
    with threadpool_limits(limits=2):main()
