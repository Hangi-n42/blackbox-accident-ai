"""One CPU scan per frozen source; exact V6 scores and fixed index1 correction.

The V6 scanner consumes an ordered virtual path list. Its image reader is bound
to one sequential PyAV decoder, yielding original PIL RGB images without PNGs.
No annotations are parsed; reviewer files are hashed as opaque bytes only.
"""
import os,sys,json,time,hashlib,types,importlib.util
from pathlib import Path
from fractions import Fraction
sys.dont_write_bytecode=True
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='2'
os.environ['CUDA_VISIBLE_DEVICES']=''
import av,numpy as np
OUT=Path(__file__).resolve().parent;V7=OUT.parent;ROOT=V7.parents[1]
SOURCE=V7/'new_validation_sources';QA=V7/'next_review_native_qa'
CODE=ROOT/'artifacts/submissions/verify_v6/model/stage2/code/solution'
CANDIDATE=V7/'solution/stage2_motion_init_v7.py'
IDS=['00017','00018','00019','00021','00022','00023']
V6SHA='3e86e5117230fb4680c1d4af630cb02dd7e9aca99b7c4184d7baed13f479d64e'
REVIEWS={V7/'next_review_a/review.json':'e8eec2f1889c97fbbe8ef05f029837d808047c30ffda7d723cc353fd2b789888',
         V7/'next_review_b/review_summary.json':'ce30cef736a69d12551ecf276ace4606a08d8b9f5910b861f7609b115c832c2c'}
START=time.monotonic()
def deadline():
    if time.monotonic()-START>600:raise TimeoutError('Frozen ten-minute CPU diagnostic limit')
def require(ok,msg):
    if not ok:raise ValueError(msg)
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def read(p):return json.loads(Path(p).read_text('utf8'))
def put(name,obj):
    with (OUT/name).open('x',encoding='utf8') as f:json.dump(obj,f,ensure_ascii=False,indent=2,allow_nan=False)
def ahash(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def main():
    require(not (OUT/'frozen_plan.json').exists(),'Single run only; preserve previous attempt')
    selected=read(SOURCE/'selection_plan.json');acquired=read(SOURCE/'acquisition.json');qa=read(QA/'report.json')
    require([r['ID'] for r in selected['selected']]==IDS and acquired['status']=='complete' and qa['status']=='complete','Frozen six cohort required')
    source_rows={r['ID']:r for r in selected['selected']};qa_rows={r['ID']:r for r in qa['videos']}
    protected=[Path(__file__),CANDIDATE,SOURCE/'selection_plan.json',SOURCE/'acquisition.json',QA/'report.json']
    protected+=sorted(CODE.glob('*.py'))
    bindings={str(p.resolve()):sha(p) for p in protected}
    require(sha(CODE/'stage2_uncapped_jerk_v6c.py')==V6SHA,'Actual V6 scanner source differs')
    for p,h in REVIEWS.items():
        require(sha(p)==h,'Review freeze hash changed');bindings[str(p.resolve())]=h
    for ID in IDS:
        path=SOURCE/(ID+'.mp4');require(path.stat().st_size==source_rows[ID]['bytes'] and sha(path)==source_rows[ID]['sha256'],'Source changed')
        require(qa_rows[ID]['source_sha256']==source_rows[ID]['sha256'],'Native QA source binding differs')
        bindings[str(path.resolve())]=source_rows[ID]['sha256'];p=QA/(ID+'.frames.json');bindings[str(p.resolve())]=sha(p)
    put('frozen_plan.json',{'created_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'IDs':IDS,'files':bindings,
        'policy':'One exact V6 _dual_motion_scan for every original sequentially decoded frame; existing corrected_scores applied once. No tuning/retries.',
        'thresholds':{'jerk_MAD_multiplier':1.4826,'MAD_floor':.001,'base_robust_clip':[0,10],'uncapped_jerk_lower':0,'residual_weight':.6,'appearance_weight':.25,
            'residual_percentile':90,'corrected_target_valid_index':1,'corrected_jerk_contribution':0,'argmax_tie':'numpy first occurrence'},
        'spatial_input':'Original PyAV frame.to_image().convert(RGB), then exact V6 PIL resize160x96 and RGB2GRAY; no resize before original RGB',
        'time_input':'No FPS passed to inference. Original indices and native PTS mapped only after argmax; every decoded frame used.',
        'CPU_threads':2,'GPU':False,'VLM':False,'training':False,'wall_limit_seconds':600,'review_files':'Opaque SHA verification only; annotation content never parsed',
        'accuracy_evaluation':False,'adoption_allowed':False,'prior_human9_advancement_gate':'Failed; this diagnostic cannot overturn it',
        'data_outputs':'One npz per source with features/base/V6 uncapped/corrected scores; no frame PNG dump'})
    package=types.ModuleType('next_motion_frozen_v6');package.__path__=[str(CODE)];sys.modules[package.__name__]=package
    spec=importlib.util.spec_from_file_location(package.__name__+'.candidate',CANDIDATE)
    candidate=importlib.util.module_from_spec(spec);sys.modules[spec.name]=candidate;spec.loader.exec_module(candidate)
    v6=candidate.v6;v6.primitives.cv2.setNumThreads(2);v6.primitives.cv2.ocl.setUseOpenCL(False)
    report={'status':'running','videos':[],'model_loaded':False,'GPU_used':False,'annotations_parsed':False,'accuracy_evaluated':False,
            'prior_human9_gate_remains_failed':True,'adoption_allowed':False}
    try:
        for ID in IDS:
            deadline();start=time.monotonic();source=SOURCE/(ID+'.mp4');mapping=read(QA/(ID+'.frames.json'))
            rows=mapping['frames'];require(mapping['source_sha256']==source_rows[ID]['sha256'],'Mapping source differs')
            count=len(rows);paths=[Path(f'frame_{i:06d}.png') for i in range(count)]
            sample_sha={r['decoded_index']:r['sha256'] for r in qa_rows[ID]['selected_rgb_sha256']};verified_rgb=[]
            with av.open(str(source)) as container:
                stream=container.streams.video[0];stream.codec_context.thread_count=2;decoder=iter(container.decode(stream));cursor=[0]
                def reader(path):
                    deadline();i=cursor[0];require(path==paths[i],'Scanner reordered virtual native frame paths')
                    frame=next(decoder);m=rows[i]
                    require(frame.pts==m['pts'] and Fraction(frame.time_base)==Fraction(m['time_base']),'Sequential native PTS mismatch')
                    require([frame.width,frame.height]==[m['width'],m['height']],'Native frame dimensions changed')
                    image=frame.to_image().convert('RGB')
                    if i in sample_sha:
                        h=ahash(np.asarray(image));require(h==sample_sha[i],'Original RGB bridge differs from own native QA');verified_rgb.append(i)
                    cursor[0]+=1;return image
                old_reader=v6.primitives._read_rgb
                try:
                    v6.primitives._read_rgb=reader
                    valid,base,uncapped,features=v6._dual_motion_scan(paths)
                finally:v6.primitives._read_rgb=old_reader
                require(cursor[0]==count and valid==paths,'Not every original frame consumed')
                require(next(decoder,None) is None,'Native decoder has extra frames')
            corrected=candidate.corrected_scores(features,uncapped)
            base2,scores2=v6._scores_from_features(features)
            require(base.tobytes()==base2.tobytes() and uncapped.tobytes()==scores2.tobytes(),'V6 score recomputation differs')
            require(features.shape==(count,3) and base.shape==uncapped.shape==corrected.shape==(count,),'Frame coverage mismatch')
            require(np.isfinite(features).all() and np.isfinite(corrected).all(),'Nonfinite motion value')
            changed=np.flatnonzero(uncapped.view(np.uint32)!=corrected.view(np.uint32)).tolist()
            require(set(changed)<={1} and (count<2 or corrected[1]<=uncapped[1]),'Correction changed another index or increased index1')
            old=int(np.argmax(uncapped));new=int(np.argmax(corrected))
            require(old==1 or old==new,'Non-boundary prior winner changed')
            name=ID+'.npz';np.savez(OUT/name,features=features,base_scores=base,v6_uncapped_scores=uncapped,corrected_scores=corrected)
            result={'ID':ID,'source_sha256':source_rows[ID]['sha256'],'decoded_frames':count,
                'v6_decoded_index':old,'corrected_decoded_index':new,'prediction_changed':old!=new,
                'v6_native_pts':rows[old]['pts'],'v6_time_base':rows[old]['time_base'],'v6_pts_seconds':rows[old]['pts_seconds'],
                'corrected_native_pts':rows[new]['pts'],'corrected_time_base':rows[new]['time_base'],'corrected_pts_seconds':rows[new]['pts_seconds'],
                'score_changed_indices':changed,'index1_v6_score':float(uncapped[1]) if count>1 else None,'index1_corrected_score':float(corrected[1]) if count>1 else None,
                'features_sha256':ahash(features),'base_scores_sha256':ahash(base),'v6_scores_sha256':ahash(uncapped),'corrected_scores_sha256':ahash(corrected),
                'array_dtype':str(features.dtype),'array_hash_definition':'contiguous native float32 bytes, frame order',
                'npz_file':name,'npz_sha256':sha(OUT/name),'verified_original_RGB_sample_indices':verified_rgb,
                'runtime_seconds':time.monotonic()-start,'annotations_or_accuracy_used':False}
            put(ID+'.json',result);report['videos'].append(result)
            print(json.dumps({k:result[k] for k in ('ID','v6_decoded_index','corrected_decoded_index','runtime_seconds')}),flush=True)
        for p,h in bindings.items():require(sha(p)==h,'Bound source/code/review bytes changed')
        report.update(status='complete',changed_video_count=sum(r['prediction_changed'] for r in report['videos']),
            unchanged_video_count=sum(not r['prediction_changed'] for r in report['videos']),all_bound_files_unchanged=True)
    except BaseException as e:report.update(status='failed_no_retry',error=repr(e));raise
    finally:
        report['runtime_seconds']=time.monotonic()-START;put('report.json',report)
    lines=['# 고정6개 V6/초기화 보정 CPU 진단','',
        '정확도·주석 평가가 아니다. 기존 human9 advancement gate 실패는 그대로 유지한다. 이번 결과만으로 채택하지 않는다.','',
        '| ID | 원본 프레임 수 | V6 index | 보정 index | V6 native 시각(s) | 보정 native 시각(s) | 변경 | 실행(s) |',
        '|---|---:|---:|---:|---:|---:|---|---:|']
    for r in report['videos']:lines.append(f"| {r['ID']} | {r['decoded_frames']} | {r['v6_decoded_index']} | {r['corrected_decoded_index']} | {r['v6_pts_seconds']:.9f} | {r['corrected_pts_seconds']:.9f} | {r['prediction_changed']} | {r['runtime_seconds']:.2f} |")
    lines+=['',f"변경{report['changed_video_count']}/6, 유지{report['unchanged_video_count']}/6. 총{report['runtime_seconds']:.2f}초. CPU2/GPU0, VLM0, training0.",
        '', '모든 원본 프레임을 순차 디코딩하여 실제 V6 scanner에 PIL RGB로 공급했다. resize/core/광류/정규화는 V6 함수가 그대로 실행했다. FPS를 추론에 주지 않았다. 선택 index를 자체 QA의 native PTS에 연결한 시각은 평가용 위치 정보이며 정답 적중 주장이 아니다.',
        '', '각 npz에 feature3열/base/V6 uncapped/보정 score를 보존했다. 다른 score index는 비트 단위로 같고 기존 승자가index1이 아니면 선택도 유지됨을 검사했다. 첫·중간·끝 RGB가 원본QA와 같음을 확인했다. PNG를 덤프하지 않았다.',
        '', '검수 파일은 동결 SHA 확인만 했으며 JSON 내용을 파싱하지 않았다. root가 조건부 구간과 비교한다. 새 영상6개의 결과로 이전 개발 실패를 소급 통과시키거나 추가 threshold/예외를 탐색하지 않는다.']
    (OUT/'summary.md').write_text('\n'.join(lines),encoding='utf8')

if __name__=='__main__':main()
