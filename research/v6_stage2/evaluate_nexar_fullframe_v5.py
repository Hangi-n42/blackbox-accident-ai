"""CPU diagnostic evaluation: frozen V5 full-frame versus prior 10Hz, known human labels only."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
IDS=['00000','00003','00004','00005','00006','00007']
COHORTS = {'development':['00000','00003','00004'], 'reserved':['00005','00006','00007']}
REVIEWS = {
    '00000':'NEXAR_REVIEW_00000_review_1789400071868.json',
    '00003':'NEXAR_REVIEW_00003_review_1789444125537.json',
    '00004':'NEXAR_REVIEW_00004_review_1789444066492.json',
    '00005':'NEXAR_REVIEW_00005_review_1789444069007.json',
    '00006':'NEXAR_REVIEW_00006_review_1789444072672.json',
    '00007':'NEXAR_REVIEW_00007_review_1789404064071.json',
}
INTAKES = {
    'old':(HERE/'nexar_review_intake_integrity.json','23c9a7d4413e184eebd8550dd8b2c1fb1462b1d6a4916b95963485f64b90f1fe'),
    'new':(HERE/'nexar_review_intake_integrity_addendum.json','486838bfcc551d54caf2889a61cf3dba178b1c1b2e3d66af2a23c35707cbe937'),
}
EPS = 1e-12
def require(value, message):
    if not value:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f,'sha256').hexdigest()


def intake_for(ID):
    path,expected=INTAKES['old' if ID in ('00000','00007') else 'new']
    require(sha(path)==expected,'Intake evidence changed')
    item=next(x for x in read(path)['results'] if x['ID']==f'NEXAR_REVIEW_{ID}')
    hashes=item['sha256']
    return item, dict(intake_path=str(path),intake_sha256=expected,
                      expected_review_sha256=hashes.get('preserved_copy',hashes.get('review')),
                      expected_source_sha256=hashes['source'])


def known_labels(review,times):
    labels={}
    for name in ('contact','entry'):
        item=review.get(name)
        if isinstance(item,dict) and item.get('status')=='observed':
            require(type(item.get('frame')) is int and item['frame'] in times,'Observed label lacks valid original frame')
            require(item.get('pts_seconds')==times[item['frame']],'Observed label/native PTS mismatch')
            labels[name]=dict(frame=item['frame'],pts_seconds=item['pts_seconds'])
        else:labels[name]=None
    side=review.get('side');space=review.get('space')
    require(side in (None,'','uncertain','UNKNOWN','LEFT','RIGHT'),'Unexpected side encoding')
    require(space in (None,'','uncertain','UNKNOWN','0','1',0,1) and type(space) is not bool,'Unexpected space encoding')
    labels['side']=side if side in ('LEFT','RIGHT') else None
    labels['space']=int(space) if space in ('0','1',0,1) else None
    if review.get('already_entered_at_start') is True:
        require(labels['entry'] is not None and labels['entry']['frame']==min(times),'Already-entered label differs from first source frame')
    return labels


def validate_prediction(pred,numbers):
    require(set(pred)=={'collision_frame','entry_frame','entry_side','evasion_space'},'Unexpected output fields')
    require(all(type(pred[k]) is int and pred[k] in numbers for k in ('collision_frame','entry_frame')),'Invalid output original frame')
    require(pred['entry_side'] in ('LEFT','RIGHT') and type(pred['evasion_space']) is int and pred['evasion_space'] in (0,1),'Invalid output class')


def measure(pred,labels,times):
    result={}
    for name,key in (('contact','collision_frame'),('entry','entry_frame')):
        gt=labels[name]
        if gt is None:result[name]=None;continue
        error=times[pred[key]]-gt['pts_seconds']
        result[name]=dict(signed_error_seconds=error,abs_error_seconds=abs(error),correct=abs(error)<=.3+EPS,
                          predicted_frame=pred[key],predicted_pts_seconds=times[pred[key]])
    for name,key in (('side','entry_side'),('space','evasion_space')):
        result[name]=None if labels[name] is None else dict(truth=labels[name],prediction=pred[key],correct=labels[name]==pred[key])
    return result


def macro_f1(truth,prediction,classes):
    if not truth:return None
    scores=[]
    for c in classes:
        tp=sum(a==c and b==c for a,b in zip(truth,prediction))
        fp=sum(a!=c and b==c for a,b in zip(truth,prediction))
        fn=sum(a==c and b!=c for a,b in zip(truth,prediction))
        scores.append(2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.)
    return sum(scores)/len(classes)


def summarize(rows,variant):
    output={}
    for name in ('contact','entry','side','space'):
        known=[r[variant][name] for r in rows if r[variant][name] is not None]
        ids=[r['ID'] for r in rows if r[variant][name] is not None]
        value=dict(known_count=len(known),unknown_count=len(rows)-len(known),known_ids=ids,
                   correct_count=sum(x['correct'] for x in known),accuracy=None if not known else sum(x['correct'] for x in known)/len(known))
        if name in ('contact','entry'):
            value['mean_abs_error_seconds']=None if not known else sum(x['abs_error_seconds'] for x in known)/len(known)
            value['tolerance_seconds']=.3
        else:
            value['macro_f1']=macro_f1([x['truth'] for x in known],[x['prediction'] for x in known],['LEFT','RIGHT'] if name=='side' else [0,1])
            value['macro_classes']=['LEFT','RIGHT'] if name=='side' else [0,1]
        output[name]=value
    return output


def old_trace(ID,frozen):
    spec=frozen['baseline10hz_by_id'][ID]
    rp=Path(spec['report_path']);fp=Path(spec['freeze_path'])
    require(frozen['baseline10hz_refs'][str(rp)]==sha(rp) and frozen['baseline10hz_refs'][str(fp)]==sha(fp),'Old 10Hz reference SHA mismatch')
    report=read(rp);oldfreeze=read(fp)
    require(report['status']=='complete' and report['network_attempts']==0 and report['freeze_sha256']==sha(fp),'Incomplete/unbound old report')
    t=next(v for v in report['videos'] if v['ID']==ID)
    # Reserved C execution preserves its original V5 baseline separately. Never use C candidate.
    expected_field='prediction' if ID in ('00000','00003','00004') else 'baseline'
    require(spec['prediction_field']==expected_field,'Wrong historical prediction policy')
    prediction=t[expected_field]
    diagnostics=t['baseline_diagnostics'] if 'baseline_diagnostics' in t else t['diagnostics']
    record=next(v for v in oldfreeze['videos'] if v['ID']==ID)
    source=record['input'] if 'input' in record else record
    return prediction,diagnostics,t['frame_numbers'],source,dict(report_path=str(rp),report_sha256=sha(rp),freeze_path=str(fp),freeze_sha256=sha(fp))


def evaluate_video(record,trace,frozen):
    ID=record['ID'];source=record['input'];mapping=source['source_frame_pts']
    intake,binding=intake_for(ID);reviewpath=HERE/'user_reviews'/REVIEWS[ID]
    require(sha(reviewpath)==binding['expected_review_sha256'],'Preserved user review changed')
    review=read(reviewpath)
    require(review['ID']=='NEXAR_REVIEW_'+ID and review['record_type']=='human_review_draft' and review['evaluation_eligible'] is False,'Unexpected human reference provenance')
    require(source['source_sha256']==review['source_video_sha256']==binding['expected_source_sha256']==sha(source['source_path']),'Source video binding mismatch')
    require(review['frame_mapping_sha256']==intake['sha256'].get('JS_JSON_stringify_full_frames',intake['sha256'].get('JS_JSON_stringify_full_case_frames')),'User native-map binding mismatch')
    times={x['frame']:x['pts_seconds'] for x in mapping}
    require(list(times)==list(range(len(mapping))) and all(math.isfinite(t) for t in times.values()),'Invalid native mapping')
    require(all(b>a for a,b in zip(times.values(),list(times.values())[1:])),'Nonmonotonic native PTS')
    audited=intake['full_native_mapping'];require(len(mapping)==len(audited),'Native mapping count mismatch')
    for a,b in zip(mapping,audited):
        require(a['frame']==b['frame'] and a['pts_seconds']==b['pts_seconds'],'Native frame/time differs from intake')
        if 'native_pts' in a:require(a['native_pts']==b['native_pts'],'Native integer PTS mismatch')
        if 'time_base' in a:require(a['time_base']==[b['time_base_numerator'],b['time_base_denominator']],'Native timebase mismatch')
    numbers=trace['frame_numbers']
    require(numbers==list(times) and [x['frame'] for x in source['input_images']]==numbers,'Full-frame input missing/renumbered')
    require(trace['source_frame_pts']==mapping and trace['input_manifest_sha256']==source['input_manifest_sha256'],'Trace input binding mismatch')
    require(hashlib.sha256(json.dumps(source['input_images'],sort_keys=True).encode()).hexdigest()==source['input_manifest_sha256'],'Full-frame manifest digest mismatch')
    for key in ('case','mapping'):
        require(sha(record[key+'_path'])==record[key+'_sha256'],'Full frame metadata binding changed')
    for image in source['input_images']:
        p=(Path(record['input_root'])/image['path']).resolve()
        require(p.is_relative_to(Path(record['input_root']).resolve()) and sha(p)==image['file_sha256'],'Full PNG SHA mismatch')
        require(image['pts_seconds']==times[image['frame']],'Full PNG PTS mismatch')
    pred=trace['prediction'];validate_prediction(pred,numbers)
    scores=trace['motion_scores']
    require(len(scores)==len(numbers) and all(type(s) in (int,float) and math.isfinite(s) for s in scores),'Invalid V5 motion scores')
    require(pred['collision_frame']==numbers[max(range(len(numbers)),key=lambda i:scores[i])],'Full V5 motion argmax mismatch')
    old,old_diag,old_numbers,old_source,ref=old_trace(ID,frozen)
    require(old_source['source_sha256']==source['source_sha256'],'Sampling comparison source mismatch')
    require(old_numbers==sorted(set(old_numbers)) and set(old_numbers).issubset(numbers),'Invalid old sampled numbers')
    old_times={x['frame']:x['pts_seconds'] for x in old_source['source_frame_pts']}
    require(old_times==times,'Sampling comparison native PTS mismatch')
    validate_prediction(old,old_numbers)
    labels=known_labels(review['review'],times)
    full=measure(pred,labels,times);prior=measure(old,labels,times)
    changes={name:None if full[name] is None else dict(old_correct=prior[name]['correct'],full_correct=full[name]['correct'],changed_to_correct=not prior[name]['correct'] and full[name]['correct'],changed_to_wrong=prior[name]['correct'] and not full[name]['correct']) for name in full}
    return dict(ID=ID,review_path=str(reviewpath),review_sha256=sha(reviewpath),label_binding=binding,
                source_sha256=source['source_sha256'],labels=labels,full_prediction=pred,old10hz_prediction=old,
                full=full,old10hz=prior,changes=changes,full_frame_count=len(numbers),old10hz_frame_count=len(old_numbers),
                old10hz_reference=ref,full_internal_collision=trace['diagnostics']['collision_replacement']['base_collision_frame'],
                old10hz_internal_collision=old_diag['collision_replacement']['base_collision_frame'])


def evaluate(run_dir):
    out=run_dir/'evaluation.json';require(not out.exists(),'Refuse evaluation overwrite')
    fp=run_dir/'freeze.json';frozen=read(fp)
    require([v['ID'] for v in frozen['videos']]==IDS,'Expected exactly six exposed reviewed source IDs')
    for p,h in frozen['files'].items():require(sha(p)==h,'Frozen source/protocol changed: '+p)
    for p,h in frozen['baseline10hz_refs'].items():require(sha(p)==h,'Old reference changed: '+p)
    rp=run_dir/'report.json';report=read(rp)
    require(report['status']=='complete' and report['network_attempts']==0 and report['call_count']==report['max_calls']==frozen['max_calls']==24 and report['freeze_sha256']==sha(fp),'Invalid full-frame run')
    require(report['experiment']==frozen['experiment']=='v5_fullframe_diagnostic' and report['model_loads']==1 and report['gt_read'] is False,'Wrong run provenance')
    require([v['ID'] for v in report['videos']]==IDS,'Unexpected report IDs')
    for t in report['videos']:
        require(len(t['calls'])==4 and all(c['status']=='complete' for c in t['calls']) and [c['max_new_tokens'] for c in t['calls']]==[64,48,40,40],'Frozen V5 four-call budget changed')
    rows=[evaluate_video(a,b,frozen) for a,b in zip(frozen['videos'],report['videos'])]
    result=dict(created_utc=datetime.now(timezone.utc).isoformat(),purpose='exposed-source sampling diagnostic only',
                videos=rows,full=summarize(rows,'full'),old10hz=summarize(rows,'old10hz'),
                S2=None,official_S2=None,gate_passed=None,ground_truth_promotion=False,independent_validation=False,
                freeze_sha256=sha(fp),report_sha256=sha(rp),evaluator_path=str(Path(__file__).resolve()),evaluator_sha256=sha(__file__),
                review_file_sha256s={r['review_path']:r['review_sha256'] for r in rows},
                package_weights_rehashed_by_evaluator=False,shared_pixel_equivalence_independently_rechecked_by_evaluator=False,
                limitations=['Single-human draft labels, some entry/side labels unknown; no composite S2.',
                    'Full native PNG versus prior 10Hz PNG tests local sampling density, not hidden official JPEG rendering.',
                    'Hidden FPS and frame-to-time distribution are unknown; original frame integers are not general VFR timestamps.',
                    'All six sources are exposed; descriptive comparisons are not independent candidate acceptance.'])
    with out.open('x',encoding='utf-8') as f:json.dump(result,f,ensure_ascii=False,indent=2,allow_nan=False)
    print(json.dumps({'S2':None,'full':result['full'],'old10hz':result['old10hz']}))


def self_test():
    times={0:0.,1:.11,2:.43}
    labels=known_labels({'contact':{'status':'observed','frame':1,'pts_seconds':.11},'entry':{'status':'uncertain'},'side':'','space':'1'},times)
    assert labels['entry'] is None and labels['side'] is None and labels['space']==1
    pred=dict(collision_frame=2,entry_frame=0,entry_side='LEFT',evasion_space=1)
    validate_prediction(pred,list(times));m=measure(pred,labels,times)
    assert not m['contact']['correct'] and m['entry'] is None and m['side'] is None and m['space']['correct']
    assert macro_f1(['LEFT'],['LEFT'],['LEFT','RIGHT'])==.5
    assert macro_f1([],[],['LEFT','RIGHT']) is None
    try:known_labels({'contact':{'status':'observed','frame':1,'pts_seconds':.1}},times)
    except ValueError:pass
    else:raise AssertionError('Wrong native timestamp accepted')
    print('CPU contracts PASS: native PTS, unknown labels, complete class macro-F1; no run/GT files opened')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run-dir',type=Path);p.add_argument('--self-test',action='store_true');a=p.parse_args()
    if a.self_test:self_test()
    else:require(a.run_dir is not None,'--run-dir required');evaluate(a.run_dir.resolve())


