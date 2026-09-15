"""After root confirms both reviews frozen: sample PNG/native PTS only.

Never reads annotation JSON or sheets. No model, GPU, download, or new PNG dump.
"""
import os,sys,argparse,json,hashlib,re,time,math
from pathlib import Path
from fractions import Fraction
sys.dont_write_bytecode=True
for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[name]='2'
HERE=Path(__file__).resolve().parent;V7=HERE.parent
IDS=('00017','00018','00019','00021','00022','00023')
PATTERN=re.compile(r'^(?:frame_|native_frame|overview_frame)([0-9]{1,9})\.png$')

def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def put(path,value):
    with Path(path).open('x',encoding='utf8') as f:json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False)
def select_indices(indices):
    ordered=sorted(set(indices))
    if not ordered:return []
    return sorted({ordered[0],ordered[len(ordered)//2],ordered[-1]})
def native_row(row,kind):
    if kind=='A':
        pts=row.get('pts');tb=row.get('time_base');seconds=row.get('seconds')
        tb=None if tb is None else Fraction(tb)
    else:
        pts=row.get('native_pts');num=row.get('time_base_num');den=row.get('time_base_den');seconds=row.get('pts_seconds')
        tb=None if num is None or den is None else Fraction(num,den)
    return {'index':row.get('frame'),'pts':pts,'time_base':tb,'time':None if pts is None or tb is None else pts*tb,'seconds':seconds}

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--reviews-frozen',action='store_true',required=True,
        help='Root confirmation that both reviews are frozen; no annotation files are inspected to infer this')
    parser.add_argument('--review-a',type=Path,default=V7/'next_review_a');parser.add_argument('--review-b',type=Path,default=V7/'next_review_b')
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    out=args.output.resolve()
    if not out.is_relative_to(HERE) or out.exists():raise ValueError('New output below next_review_native_qa required')
    started=time.monotonic()
    def deadline():
        if time.monotonic()-started>180:raise TimeoutError('Fixed180second CPU verification budget')
    own=read(HERE/'report.json');assert own['status']=='complete'
    by={r['ID']:r for r in own['videos']};assert set(by)==set(IDS)
    bindings={str(HERE/'report.json'):sha(HERE/'report.json'),str(Path(__file__)):sha(Path(__file__))}
    selections=[];errors=[]
    for label,review_root,map_name in [('A',args.review_a,'native_mapping.json'),('B',args.review_b,'native_pts.json')]:
        review_root=review_root.resolve()
        extractor=review_root/'extract_review.py'
        if extractor.is_file():bindings[str(extractor)]=sha(extractor)
        for ID in IDS:
            folder=review_root/ID;mapping=folder/map_name
            files={}
            # Anchored filenames exclude every contact sheet and other PNG type.
            if folder.is_dir():
                for path in folder.glob('*.png'):
                    m=PATTERN.fullmatch(path.name)
                    if m:files.setdefault(int(m.group(1)),[]).append(path.resolve())
            indices=select_indices(files)
            if len(indices)<3:errors.append({'reviewer':label,'ID':ID,'error':'fewer_than_three_distinct_individual_frame_ids','count':len(files)})
            selected=[{'index':i,'path':str(p),'sha256':sha(p)} for i in indices for p in sorted(files[i])]
            for row in selected:bindings[row['path']]=row['sha256']
            if not mapping.is_file():errors.append({'reviewer':label,'ID':ID,'error':'missing_native_mapping'})
            else:bindings[str(mapping)]=sha(mapping)
            selections.append({'reviewer':label,'ID':ID,'mapping_path':str(mapping),'available_distinct_frame_ids':len(files),
                'available_min':min(files) if files else None,'available_max':max(files) if files else None,
                'selected_indices':indices,'selected_images':selected})
    for ID in IDS:
        p=HERE/(ID+'.frames.json');bindings[str(p)]=sha(p)
        source=Path(by[ID]['source_path']);assert sha(source)==by[ID]['source_sha256'];bindings[str(source)]=by[ID]['source_sha256']
    out.mkdir(parents=True)
    put(out/'frozen_verification_plan.json',{'root_attested_both_reviews_frozen':True,'selection_rule':'Lowest, middle-by-sorted-distinct-ID, highest available individual PNG frame indices for each source/reviewer; verify every recognized PNG at selected indices',
        'filename_regex':PATTERN.pattern,'selections':selections,'files':bindings,'annotation_JSON_read':False,'contact_sheets_read':False,
        'CPU_threads':2,'GPU':False,'maximum_seconds':180,'scope':'Sampled image provenance and native PTS only, no semantic annotation approval'})
    import av
    import numpy as np
    from PIL import Image
    image_results=[];mapping_results=[]
    for ID in IDS:
        deadline();qa=read(HERE/(ID+'.frames.json'));q={r['decoded_index']:r for r in qa['frames']}
        selected=[s for s in selections if s['ID']==ID]
        wanted={r['index'] for s in selected for r in s['selected_images']};rgb={};native={}
        for s in selected:
            mpath=Path(s['mapping_path']);issues=[]
            if not mpath.is_file():continue
            mapped=read(mpath)  # Only explicit native_mapping/native_pts allowlist.
            if mapped.get('source_sha256')!=qa['source_sha256']:issues.append({'error':'source_sha_mismatch'})
            parsed=[]
            for row in mapped.get('frames',[]):
                try:parsed.append(native_row(row,s['reviewer']))
                except (ValueError,TypeError,ZeroDivisionError):issues.append({'error':'invalid_pts_row','frame':row.get('frame')})
            seen=[r['index'] for r in parsed]
            if seen!=list(range(len(q))):issues.append({'error':'frame_index_order_count_or_coverage','expected_count':len(q),'actual_count':len(parsed)})
            for row in parsed:
                index=row['index']
                if index not in q:continue
                qr=q[index];qtb=None if qr['time_base'] is None else Fraction(qr['time_base'])
                expected=None if qr['pts'] is None or qtb is None else qr['pts']*qtb
                if row['time'] is None:issues.append({'error':'missing_pts','frame':index})
                if row['pts']!=qr['pts'] or row['time_base']!=qtb or row['time']!=expected:issues.append({'error':'native_pts_or_timebase_mismatch','frame':index})
                if row['seconds'] is None or expected is None or not math.isfinite(float(row['seconds'])) or abs(float(row['seconds'])-float(expected))>1e-12:issues.append({'error':'seconds_field_mismatch','frame':index})
            deltas=[b['time']-a['time'] for a,b in zip(parsed,parsed[1:]) if a['time'] is not None and b['time'] is not None]
            nonmono=sum(d<=0 for d in deltas)
            if nonmono:issues.append({'error':'nonmonotonic_intervals','count':nonmono})
            mapping_results.append({'reviewer':s['reviewer'],'ID':ID,'mapping_sha256':bindings[str(mpath)],'frames':len(parsed),
                'distinct_intervals_seconds_rational':sorted({str(d) for d in deltas}),'issues':issues,'passed':not issues})
            errors.extend({'reviewer':s['reviewer'],'ID':ID,**x} for x in issues)
        with av.open(by[ID]['source_path']) as c:
            stream=c.streams.video[0];stream.codec_context.thread_count=2
            for i,frame in enumerate(c.decode(stream)):
                deadline()
                if i in wanted:
                    rgb[i]=frame.to_ndarray(format='rgb24');native[i]=(frame.pts,Fraction(frame.time_base))
                if wanted and i>=max(wanted):break
        for s in selected:
            for sample in s['selected_images']:
                i=sample['index'];path=Path(sample['path']);result={'reviewer':s['reviewer'],'ID':ID,**sample}
                if i not in rgb or i not in q:
                    result.update(passed=False,error='selected_frame_missing_or_out_of_range');errors.append(result);image_results.append(result);continue
                with Image.open(path) as im:actual=np.asarray(im.convert('RGB'))
                expected=rgb[i];shape_equal=actual.shape==expected.shape
                equal=shape_equal and np.array_equal(actual,expected)
                qa_pts=(q[i]['pts'],Fraction(q[i]['time_base']))
                pts_equal=native[i]==qa_pts
                unchanged=sha(path)==sample['sha256']
                result.update(passed=equal and pts_equal and unchanged,RGB_exact_equal=equal,shape_equal=shape_equal,
                    PNG_RGB_shape=list(actual.shape),original_RGB_shape=list(expected.shape),PNG_RGB_sha256=hashlib.sha256(actual.tobytes()).hexdigest(),
                    original_RGB_sha256=hashlib.sha256(expected.tobytes()).hexdigest(),native_pts=native[i][0],
                    time_base=str(native[i][1]),pts_seconds=float(native[i][0]*native[i][1]),native_PTS_equals_own_QA=pts_equal,
                    PNG_file_unchanged=unchanged,max_RGB_abs_error=int(np.max(np.abs(actual.astype(np.int16)-expected.astype(np.int16)))) if shape_equal else None)
                image_results.append(result)
                if not result['passed']:errors.append({'reviewer':s['reviewer'],'ID':ID,'frame':i,'error':'PNG_RGB_PTS_or_file_changed'})
    changed=[p for p,h in bindings.items() if sha(p)!=h]
    errors.extend({'error':'bound_file_changed','path':p} for p in changed)
    result={'status':'passed' if not errors else 'failed','mapping_results':mapping_results,'image_results':image_results,'errors':errors,
        'seconds':time.monotonic()-started,'annotation_semantics_checked':False,'model_or_accuracy_evaluation':False,
        'limitations':'Only three distinct available native frame IDs per source/reviewer, with every recognized PNG at these IDs, are pixel-verified. Unselected PNGs/contact sheets are not certified. Exact native mapping validation is not human/official GT approval.'}
    put(out/'report.json',result)
    text=f"# 검수 PNG 원본 대조\n\n상태: {result['status']}. 개별 PNG {len(image_results)}개, native mapping {len(mapping_results)}개 검사. 오류 {len(errors)}개.\n\n{result['limitations']}\n\n주석 JSON·모델·정답은 읽지 않았다. 두 검수의 동결 여부는 root의 명시적 실행 플래그에 의존한다.\n"
    (out/'summary.md').write_text(text,encoding='utf8')
    print(json.dumps({'status':result['status'],'images':len(image_results),'mappings':len(mapping_results),'errors':len(errors)}))
    if errors:raise SystemExit(1)

if __name__=='__main__':main()
