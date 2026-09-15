"""Fixed six-source native decode audit; no annotations/model/PNG corpus."""
import os,sys,time,json,hashlib
from pathlib import Path
from fractions import Fraction
from collections import Counter
sys.dont_write_bytecode=True
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[k]='2'
import av
OUT=Path(__file__).resolve().parent;ROOT=OUT.parents[2]
SOURCE=ROOT/'research/v7/new_validation_sources'
IDS=['00017','00018','00019','00021','00022','00023']
START=time.monotonic()
def deadline():
    if time.monotonic()-START>180:raise TimeoutError('180-second fixed CPU audit budget exceeded')
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def read(p):return json.loads(Path(p).read_text('utf8'))
def put(name,obj):
    with (OUT/name).open('x',encoding='utf8') as f:json.dump(obj,f,indent=2,ensure_ascii=False,allow_nan=False)
def rational(x):return None if x is None else str(Fraction(x))
def frame_rgb(frame,index):
    a=frame.to_ndarray(format='rgb24')
    return {'decoded_index':index,'pts':frame.pts,'time_base':rational(frame.time_base),'shape':list(a.shape),'dtype':str(a.dtype),'sha256':hashlib.sha256(a.tobytes(order='C')).hexdigest(),'hash_definition':'contiguous HWC rgb24 uint8 bytes; no resize, overlay, image codec or color conversion after PyAV RGB decode'}
def decode(row):
    path=SOURCE/(row['ID']+'.mp4');deadline()
    require=path.stat().st_size==row['bytes'] and sha(path)==row['sha256']
    if not require:raise ValueError('Source SHA/size mismatch: '+row['ID'])
    frames=[];times=[];sizes=Counter();bases=Counter();first=middle=last=None
    with av.open(str(path)) as container:
        streams=list(container.streams.video)
        if len(streams)!=1:raise ValueError('Expected one video stream')
        stream=streams[0];stream.codec_context.thread_count=2
        count_hint=stream.frames;mid_hint=count_hint//2 if count_hint else None
        meta={'stream_index':stream.index,'time_base':rational(stream.time_base),'average_rate':rational(stream.average_rate),
            'base_rate':rational(stream.base_rate),'guessed_rate':rational(stream.guessed_rate),'declared_frames':count_hint,
            'start_time':stream.start_time,'duration':stream.duration,'codec':stream.codec_context.name,
            'declared_resolution':[stream.width,stream.height]}
        for i,frame in enumerate(container.decode(stream)):
            deadline();tb=frame.time_base;stamp=None if frame.pts is None or tb is None else Fraction(frame.pts)*tb
            times.append(stamp);sizes[(frame.width,frame.height)]+=1;bases[rational(tb)]+=1
            frames.append({'decoded_index':i,'pts':frame.pts,'time_base':rational(tb),'pts_seconds':None if stamp is None else float(stamp),
                'pts_seconds_rational':rational(stamp),'width':frame.width,'height':frame.height,'duration_ticks':getattr(frame,'duration',None)})
            if i==0:first=frame
            if i==mid_hint:middle=frame
            last=frame
    n=len(frames)
    if not n:raise ValueError('No decoded frames')
    midpoint=n//2;extra_pass=False
    if count_hint!=n or middle is None:
        extra_pass=True
        with av.open(str(path)) as c:
            s=c.streams.video[0];s.codec_context.thread_count=2
            for i,f in enumerate(c.decode(s)):
                deadline()
                if i==midpoint:middle=f;break
    selected=[frame_rgb(f,i) for f,i in [(first,0),(middle,midpoint),(last,n-1)]]
    deltas=[times[i]-times[i-1] for i in range(1,n) if times[i] is not None and times[i-1] is not None]
    missing=[i for i,t in enumerate(times) if t is None]
    nonmono=[i for i in range(1,n) if times[i] is not None and times[i-1] is not None and times[i]<=times[i-1]]
    positive=[d for d in deltas if d>0];counts=Counter(rational(d) for d in deltas)
    summary={'ID':row['ID'],'source_path':str(path),'source_sha256':row['sha256'],'source_bytes':row['bytes'],'stream':meta,
        'decoded_frames':n,'midpoint_index':midpoint,'extra_midpoint_decode_pass':extra_pass,'missing_pts_indices':missing,'nonmonotonic_pts_indices':nonmono,
        'resolutions':[{'width':w,'height':h,'count':c} for (w,h),c in sizes.items()],'frame_time_bases':dict(bases),
        'first_pts_seconds':None if times[0] is None else float(times[0]),'last_pts_seconds':None if times[-1] is None else float(times[-1]),
        'interval_histogram_seconds_rational':dict(counts),'observed_fps_over_span':float(Fraction(n-1)/(times[-1]-times[0])) if not missing and n>1 and times[-1]>times[0] else None,
        'min_positive_delta_seconds':float(min(positive)) if positive else None,'max_positive_delta_seconds':float(max(positive)) if positive else None,
        'flags':{'missing_pts':bool(missing),'nonmonotonic_pts':bool(nonmono),'variable_dimensions':len(sizes)>1,'variable_frame_time_base':len(bases)>1,
            'nonconstant_native_pts_step':len(counts)>1,'declared_count_mismatch':bool(count_hint and count_hint!=n)},'selected_rgb_sha256':selected}
    put(row['ID']+'.frames.json',{'source_sha256':row['sha256'],'index_definition':'zero-based sequential decoded display order; native PTS retained','frames':frames})
    return summary
def main():
    plan=read(SOURCE/'selection_plan.json');acq=read(SOURCE/'acquisition.json')
    if acq['status']!='complete' or [r['ID'] for r in plan['selected']]!=IDS:raise ValueError('Fixed six completed source cohort required')
    records={r['ID']:r for r in acq['records']}
    for row in plan['selected']:
        if any(records[row['ID']][k]!=row[k] for k in ('bytes','sha256','relative_path')):raise ValueError('Acquisition binding differs')
    oldpaths=[ROOT/f'research/v6/nexar_review_candidates/{i}.mp4' for i in ('00000','00003','00004','00005','00006','00007')]
    oldpaths += [ROOT/f'research/v6/nexar_review_round2/{i}.mp4' for i in ('00008','00010','00013')]
    oldpaths += [ROOT/f'research/v7/fresh_sources_retry1/{i}.mp4' for i in ('00014','00015','00016')]
    old=[{'ID':p.stem,'path':str(p),'bytes':p.stat().st_size,'sha256':sha(p)} for p in oldpaths]
    put('frozen_audit_plan.json',{'selection_plan_sha256':sha(SOURCE/'selection_plan.json'),'acquisition_sha256':sha(SOURCE/'acquisition.json'),
        'audit_code_sha256':sha(Path(__file__)),'IDs':IDS,'cpu_decode_threads':2,'wall_limit_seconds':180,
        'sample_indices':'0, floor(actual_decoded_count/2), actual_decoded_count-1','old_source_file_hashes':old,
        'no_annotation_or_prediction_access':True,'no_GPU':True,'no_download':True,'exact_file_duplicate_only':True})
    summaries=[]
    for row in plan['selected']:
        result=decode(row);summaries.append(result);print(json.dumps({'ID':row['ID'],'decoded':result['decoded_frames'],'flags':result['flags']}),flush=True)
    hashes=[r['source_sha256'] for r in summaries];oldhash={r['sha256'] for r in old}
    duplicates=[{'ID':r['ID'],'old_ID':o['ID']} for r in summaries for o in old if r['source_sha256']==o['sha256']]
    report={'status':'complete','videos':summaries,'six_exact_file_sha_distinct':len(set(hashes))==6,'exact_matches_old12':duplicates,
        'all_sources_still_match':all(sha(Path(r['source_path']))==r['source_sha256'] for r in summaries),'seconds':time.monotonic()-START,
        'limits':'SHA equality detects identical file bytes only. No perceptual, same-incident, route, training/pretraining independence certification. No human/AI annotations or model predictions read. Rational PTS variation is reported, not automatically treated as corrupt video.'}
    put('report.json',report)
    lines=['# 고정6개 원본 영상 QA','',f"완료. 실제 파일 SHA가 선택/획득 기록과 일치하며,6개 서로 및 기존12개 파일과 동일 SHA 중복은 {len(duplicates)}개다.",'',
        '| ID | 실제 프레임 | 해상도 | stream timebase | 관측 FPS | PTS 간격 종류 | 누락/비증가 PTS |','|---|---:|---|---|---:|---:|---|']
    for r in summaries:
        lines.append(f"| {r['ID']} | {r['decoded_frames']} | {r['resolutions'][0]['width']}×{r['resolutions'][0]['height']} | {r['stream']['time_base']} | {r['observed_fps_over_span']:.9f} | {len(r['interval_histogram_seconds_rational'])} | {len(r['missing_pts_indices'])}/{len(r['nonmonotonic_pts_indices'])} |")
    lines+=['','각 영상의 `.frames.json`은 zero-based sequential decoded index와 원시PTS/timebase/정확 유리수시각·해상도를 보존한다. `report.json`에 첫·중간·마지막 RGB SHA를 기록했다. PNG 전체 덤프는 만들지 않았다.',
        '',report['limits'],'',f"실행 {report['seconds']:.2f}초. GPU·다운로드·모델·정답/검수 주석 열람 없음."]
    (OUT/'summary.md').write_text('\n'.join(lines),encoding='utf8')
if __name__=='__main__':main()
