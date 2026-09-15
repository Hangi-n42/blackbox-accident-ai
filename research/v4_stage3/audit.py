"""V4 read-only source audit; no fitting, threshold search, or production writes."""
from pathlib import Path
import sys,json,hashlib,time
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import cv2,numpy as np,pandas as pd,joblib
from scipy.ndimage import uniform_filter1d
from sklearn.metrics import classification_report,confusion_matrix
from threadpoolctl import threadpool_limits
from solution.stage3_fast import FrameFeatureComputer,extract_motion
from solution.stage3 import ACCEL,STEER
OUT=Path(__file__).resolve().parent

def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def save(name,obj):(OUT/name).write_text(json.dumps(obj,indent=2,default=str),encoding='utf8')
def protection():
    paths=[ROOT/'solution/stage3.py',ROOT/'solution/stage3_fast.py',ROOT/'model/stage3/motion_model.joblib']+sorted((ROOT/'artifacts/submissions').glob('*.zip'))
    return {str(p.relative_to(ROOT)):{'bytes':p.stat().st_size,'mtime_ns':p.stat().st_mtime_ns,'sha256':sha(p) if p.suffix!='.zip' else None} for p in paths}

def decode(path,retain=True,stride=2):
    cap=cv2.VideoCapture(str(path));grays=[];n=0;digest=hashlib.sha256();shape=None
    while True:
        ok,bgr=cap.read()
        if not ok:break
        shape=bgr.shape;digest.update(bgr.data)
        if retain and n%stride==0:
            h=max(96,round(bgr.shape[0]*256/bgr.shape[1]))
            grays.append(cv2.cvtColor(cv2.resize(bgr,(256,h)),cv2.COLOR_BGR2GRAY))
        n+=1
    cap.release()
    return grays,{'decoded_frames':n,'decoded_bgr_sequence_sha256':digest.hexdigest(),'shape':shape}

def from_same_grays(grays,count,fps):
    """Same production DIS/temporal/interpolation operations; only pixels supplied in memory."""
    stride=max(1,round(fps/10));dis=cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_FAST)
    computer=FrameFeatureComputer();features=[];centers=[]
    for index in range(1,len(grays)):
        features.append(computer(dis.calc(grays[index-1],grays[index],None)*(fps/stride)))
        centers.append(index*stride-stride*.5)
    raw=np.asarray(features);parts=[raw]
    for window in (5,15,31):parts.append(uniform_filter1d(raw,size=window,axis=0,mode='nearest'))
    smooth=parts[2];ix=np.arange(len(smooth))
    for lag in (5,15):parts.append((smooth[np.minimum(ix+lag,len(ix)-1)]-smooth[np.maximum(ix-lag,0)])/(2*lag/10))
    combined=np.concatenate(parts,axis=1);result=np.empty((count,combined.shape[1]),np.float32)
    for j in range(combined.shape[1]):result[:,j]=np.interp(np.arange(count),centers,combined[:,j])
    return np.nan_to_num(result),np.asarray(centers)/fps

def differences(a,b):
    if a.shape!=b.shape:return {'shape_equal':False,'a_shape':a.shape,'b_shape':b.shape}
    d=a.astype(np.float64)-b.astype(np.float64)
    return {'shape_equal':True,'shape':a.shape,'bit_equal':bool(np.array_equal(a.view(np.uint32),b.view(np.uint32))),
            'max_absolute_error':float(np.abs(d).max()),'mean_absolute_error':float(np.abs(d).mean()),'unequal_values':int(np.count_nonzero(a.view(np.uint32)!=b.view(np.uint32)))}

def main():
    if (OUT/'audit_report.json').exists():raise FileExistsError('Preserve prior V4 audit')
    before=protection();start=time.perf_counter();cv2.setNumThreads(2)
    frozen={'operation':'Audits only before training','threads':2,'gpu':False,
      'same_pixel_gate':'Train20/stride2 selected rows must exactly match deploy10/stride1 from identical gray frames; flow midpoint seconds and mapping must match.',
      'cache_gate':'Regenerated train20 must equal existing training feature cache; mismatch requires investigation before fitting.',
      'corrected_10hz_gate':'Frame mapping/count must match. CRF18 re-encoding can change pixels/features; report rather than require bit identity.',
      'proxy_policy':{'speed_stop_m_s':.3,'acceleration_m_s2':.25,'steering_degrees':2,'speed_smoothing_frames20hz':21,'steer_smoothing_frames20hz':11},
      'no_threshold_or_time_shift_search':True,'public_can_use':'Known excluded exact-match OPEN_001 segment; diagnostic only, never external training.',
      'protection_before':before,'script_sha256':sha(Path(__file__))}
    save('audit_plan_frozen.json',frozen)
    mapping=json.loads((ROOT/'artifacts/public_eval_10hz/stage3/frame_mapping.json').read_text())
    mappings={Path(r['source']).stem:r['target_to_source_frame'] for r in mapping}
    model=joblib.load(ROOT/'model/stage3/motion_model.joblib');reports={};hashes={}
    for source in sorted((ROOT/'Baseline/data/stage3/videos').glob('*.mp4')):
        t=time.perf_counter();grays,info=decode(source);n=info['decoded_frames'];hashes[source.stem]=info
        train,centers20=from_same_grays(grays,n,20.)
        deploy,centers10=from_same_grays(grays,len(grays),10.)
        cached=np.load(ROOT/'research/stage3_cache'/f'{source.stem}.npy')
        corrected=ROOT/'artifacts/public_eval_10hz/stage3/videos'/source.name
        corrected_grays,corrected_info=decode(corrected,stride=1)
        actual=extract_motion(corrected,source_fps=10.)
        rec={'source':info,'corrected':corrected_info,'same_pixel_features':differences(train[::2],deploy),
             'cached_training_features':differences(train,cached),'same_flow_midpoints_seconds':bool(np.array_equal(centers20,centers10)),
             'mapping_exact':mappings[source.stem]==list(range(0,n,2)),
             'corrected_feature_difference':differences(deploy,actual),
             'corrected_gray_mae':float(np.mean([np.abs(a.astype(float)-b.astype(float)).mean() for a,b in zip(grays,corrected_grays)])),
             'corrected_predictions_changed':{task:int(np.count_nonzero(model[task].predict(deploy)!=model[task].predict(actual))) for task in ('accel','steer')},
             'seconds':time.perf_counter()-t}
        reports[source.stem]=rec;save('time_alignment_partial.json',reports)
        print(source.stem,'samepixels',rec['same_pixel_features']['bit_equal'],'cache',rec['cached_training_features']['bit_equal'],
              'reencoded_changed',rec['corrected_predictions_changed'],flush=True)
    overlap=json.loads((ROOT/'research/stage3_external_overlap.json').read_text())['excluded'][0]['segment']
    manifests=[]
    for p in sorted((ROOT/'external_data/comma2k19').glob('*_manifest.json')):manifests.extend(json.loads(p.read_text()))
    row=next(r for r in manifests if r['segment']==overlap);base=ROOT/row['local'];_,external_info=decode(base/'video.hevc',retain=False)
    ft=np.load(base/'global_pose/frame_times').ravel()
    speed_t=np.load(base/'processed_log/CAN/speed/t').ravel();speed_v=np.load(base/'processed_log/CAN/speed/value').ravel()
    steer_t=np.load(base/'processed_log/CAN/steering_angle/t').ravel();steer_v=np.load(base/'processed_log/CAN/steering_angle/value').ravel()
    speed=uniform_filter1d(np.interp(ft,speed_t,speed_v),size=21);accel=np.gradient(speed,ft)
    steer=uniform_filter1d(np.interp(ft,steer_t,steer_v),size=11)
    pa=np.where(speed<.3,3,np.where(accel>.25,0,np.where(accel<-.25,1,2)))
    ps=np.where(steer>2,0,np.where(steer<-2,2,1))
    labels=pd.read_csv(ROOT/'Baseline/data/stage3/labels.csv');labels=labels[labels.ID=='OPEN_001'].copy();ix=labels.frame_index.to_numpy()
    labels['can_boot_time']=ft[ix];labels['relative_can_time']=ft[ix]-ft[0]
    labels['speed_m_s']=speed[ix];labels['acceleration_m_s2']=accel[ix];labels['steering_degrees']=steer[ix]
    labels['proxy_accel']=ACCEL[pa[ix]];labels['proxy_steer']=STEER[ps[ix]]
    labels.to_csv(OUT/'public_can_proxy_comparison.csv',index=False)
    can={'segment':overlap,'source_manifest_record':row,'external_decoding':external_info,
         'full_decoded_pixel_sequence_matches_public':external_info==hashes['OPEN_001'],
         'frame_times_count':len(ft),'counts_match':len(ft)==external_info['decoded_frames'],
         'timestamp_monotonic':bool(np.all(np.diff(ft)>0)),'frame_dt_quantiles':np.quantile(np.diff(ft),[0,.5,1]).tolist(),
         'frame_time_duration_seconds':float(ft[-1]-ft[0]),
         'max_label_nominal_vs_can_seconds':float(np.max(np.abs((ft[ix]-ft[0])-labels.time_seconds.to_numpy()))),
         'speed_timestamp_monotonic':bool(np.all(np.diff(speed_t)>=0)),
         'steer_timestamp_monotonic':bool(np.all(np.diff(steer_t)>=0)),
         'speed_coverage_extrapolated_frames':int(np.sum((ft<speed_t[0])|(ft>speed_t[-1]))),
         'steer_coverage_extrapolated_frames':int(np.sum((ft<steer_t[0])|(ft>steer_t[-1]))),
         'scope':'One already excluded matched public segment, 10 sparse labels; not independent validation; no threshold search.'}
    for task,names,pred in [('accel',ACCEL,pa),('steer',STEER,ps)]:
        truth=np.array([list(names).index(v) for v in labels[task+'_label']]);keep=(labels.accel_label!='STOPPED').to_numpy() if task=='steer' else np.ones(len(labels),bool)
        p=pred[ix];can[task]={'agreements':int((truth[keep]==p[keep]).sum()),'count':int(keep.sum()),
          'confusion':confusion_matrix(truth[keep],p[keep],labels=range(len(names))).tolist(),
          'classification':classification_report(truth[keep],p[keep],labels=range(len(names)),target_names=list(names),output_dict=True,zero_division=0)}
    # Review every existing external frame-time sequence without rerunning extraction.
    integrity=[]
    for r in manifests:
        p=ROOT/r['local'];times=np.load(p/'global_pose/frame_times').ravel();tag='ext_'+r['segment'].replace('/','_').replace('|','_')
        cache=ROOT/'research/stage3_cache'/(tag+'.npy')
        frames=int(np.load(cache,mmap_mode='r').shape[0]) if cache.exists() else None
        integrity.append({'segment':r['segment'],'frame_times':len(times),'cached_frames':frames,
                          'count_match':None if frames is None else len(times)==frames,'positive_time_steps':bool(np.all(np.diff(times)>0)),
                          'dt_min_median_max':np.quantile(np.diff(times),[0,.5,1]).tolist()})
    after=protection();assert before==after
    report={'time_alignment':reports,'public_can':can,'external_timestamp_integrity':integrity,
            'same_pixel_time_gate_passed':all(r['same_pixel_features']['bit_equal'] and r['cached_training_features']['bit_equal'] and r['same_flow_midpoints_seconds'] and r['mapping_exact'] for r in reports.values()),
            'protected_files_unchanged':before==after,'protected_files_after':after,'elapsed_seconds':time.perf_counter()-start,
            'training_executed':False}
    save('audit_report.json',report)
    print(json.dumps({'time_gate':report['same_pixel_time_gate_passed'],'can':can,'seconds':report['elapsed_seconds']},indent=2,default=str),flush=True)

if __name__=='__main__':
    with threadpool_limits(limits=2):main()
