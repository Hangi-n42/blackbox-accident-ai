"""CPU evaluator for the single frozen V6C uncapped-jerk candidate. No model imports.

python -I -B evaluate_uncapped_jerk_v6c.py --run-dir FROZEN_RUN
Writes FROZEN_RUN/evaluation.json once. --self-test runs pure gate contracts only.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
PROTOCOL = HERE/'uncapped_jerk_v6c_protocol.json'
import numpy as np
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
GATE = dict(all_three_known_contact_labels_required=True, minimum_additional_correct_within_0_3_seconds=1,
            maximum_previously_correct_cases_lost=0, mean_absolute_contact_time_error_must_not_increase=True,
            noncollision_prediction_fields_must_be_identical=True, no_network_attempts_and_valid_original_frame_outputs=True)


def require(value, message):
    if not value:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f,'sha256').hexdigest()


def gate(rows, offline_calls_valid):
    require(len(rows)==3, 'Exactly three complete contact rows required')
    require(all(math.isfinite(r[k]) and r[k]>=0 for r in rows for k in ('baseline_abs_error_seconds','candidate_abs_error_seconds')), 'Invalid error')
    base=[r['baseline_abs_error_seconds']<=.3+EPS for r in rows]
    new=[r['candidate_abs_error_seconds']<=.3+EPS for r in rows]
    gained=sum(not a and b for a,b in zip(base,new)); lost=sum(a and not b for a,b in zip(base,new))
    mean_base=sum(r['baseline_abs_error_seconds'] for r in rows)/3
    mean_new=sum(r['candidate_abs_error_seconds'] for r in rows)/3
    conditions=dict(all_three_known_contact_labels=True, additional_correct_at_least_one=sum(new)-sum(base)>=1,
                    previously_correct_losses_zero=lost==0, mean_absolute_error_nonincrease=mean_new<=mean_base+EPS,
                    noncollision_fields_identical=all(r['other_fields_identical'] for r in rows),
                    offline_and_calls_and_original_numbers_valid=bool(offline_calls_valid))
    return dict(gate_passed=all(conditions.values()), conditions=conditions,baseline_correct_count=sum(base),
                candidate_correct_count=sum(new),additional_correct_count=sum(new)-sum(base),gained=gained,lost=lost,
                baseline_mean_abs_error_seconds=mean_base,candidate_mean_abs_error_seconds=mean_new,
                tolerance_seconds=.3,roundoff_epsilon_seconds=EPS)


def intake_for(ID):
    path,expected=INTAKES['old' if ID in ('00000','00007') else 'new']
    require(sha(path)==expected,'Intake evidence changed')
    item=next(x for x in read(path)['results'] if x['ID']==f'NEXAR_REVIEW_{ID}')
    hashes=item['sha256']
    return item, dict(intake_path=str(path),intake_sha256=expected,
                      expected_review_sha256=hashes.get('preserved_copy',hashes.get('review')),
                      expected_source_sha256=hashes['source'])


def video_result(record, trace):
    ID=record['ID']; source=record['input']; paths=record['input_root']
    review_path=HERE/'user_reviews'/REVIEWS[ID]
    intake,binding=intake_for(ID)
    require(sha(review_path)==binding['expected_review_sha256'],'Exact user review changed')
    review=read(review_path)
    require(review['ID']==f'NEXAR_REVIEW_{ID}' and review['record_type']=='human_review_draft' and review['evaluation_eligible'] is False,'Unexpected label status')
    require(source['source_sha256']==review['source_video_sha256']==binding['expected_source_sha256']==sha(source['source_path']),'Source SHA mismatch')
    require(review['frame_mapping_sha256']==intake['sha256'].get('JS_JSON_stringify_full_frames',intake['sha256'].get('JS_JSON_stringify_full_case_frames')),'Review UI frame mapping binding mismatch')
    mapping=source['source_frame_pts']; times={x['frame']:x['pts_seconds'] for x in mapping}
    require(len(times)==len(mapping) and list(times)==list(range(len(mapping))),'Invalid full native frame map')
    require(all(math.isfinite(t) for t in times.values()),'Invalid PTS')
    audited=intake['full_native_mapping']
    require(len(audited)==len(mapping),'Native audit length mismatch')
    for a,b in zip(mapping,audited):
        require(a['frame']==b['frame'] and a['pts_seconds']==b['pts_seconds'] and a['native_pts']==b['native_pts'] and a['time_base']==[b['time_base_numerator'],b['time_base_denominator']],'Native audit binding differs')
    contact=review['review']['contact']
    require(contact['status']=='observed' and type(contact['frame']) is int and contact['frame'] in times,'Three known contact labels required')
    require(contact['pts_seconds']==times[contact['frame']],'Human contact seconds must exactly match native PTS')
    valid=trace['frame_numbers']; sampled=[x['frame'] for x in source['input_images']]
    require(valid and valid==sorted(set(valid)) and set(valid).issubset(sampled),'Invalid decodevalid frame list')
    for image in source['input_images']:
        p=(Path(paths)/image['path']).resolve()
        require(p.is_relative_to(Path(paths).resolve()) and sha(p)==image['file_sha256'],'Input PNG binding mismatch')
        require(image['pts_seconds']==times[image['frame']],'Input PNG timestamp mismatch')
    old,new=trace['baseline'],trace['candidate']
    for p in (old,new):
        require(set(p)=={'collision_frame','entry_frame','entry_side','evasion_space'},'Unexpected prediction fields')
        require(all(type(p[k]) is int and p[k] in valid for k in ('collision_frame','entry_frame')),'Invalid output original frame')
        require(p['entry_side'] in ('LEFT','RIGHT') and type(p['evasion_space']) is int and p['evasion_space'] in (0,1),'Invalid categorical output')
    statistics=validate_scores(trace, record)
    signed_old=times[old['collision_frame']]-contact['pts_seconds']
    signed_new=times[new['collision_frame']]-contact['pts_seconds']
    for key in ('baseline_top_two', 'candidate_top_two'):
        for point in statistics[key]:
            point['pts_seconds']=times[point['frame']]
            point['signed_error_seconds']=times[point['frame']]-contact['pts_seconds']
    correct_old=abs(signed_old)<=.3+EPS; correct_new=abs(signed_new)<=.3+EPS
    return dict(ID=ID,review_path=str(review_path),review_sha256=sha(review_path),
                source_path=source['source_path'],source_sha256=source['source_sha256'],
                label_binding=binding,human_contact_draft=contact,baseline=old,candidate=new,
                baseline_signed_error_seconds=signed_old,candidate_signed_error_seconds=signed_new,
                baseline_abs_error_seconds=abs(signed_old),candidate_abs_error_seconds=abs(signed_new),
                baseline_correct=correct_old,candidate_correct=correct_new,
                gained=not correct_old and correct_new,lost=correct_old and not correct_new,
                other_fields_identical=all(old[k]==new[k] for k in ('entry_frame','entry_side','evasion_space')),
                score_audit=statistics,independent_score_recalculation=True,
                label_scope='Single-human local contact reference only; not official/adjudicated GT')


def recompute(features):
    """Independent float32 implementation; no candidate/runtime module imported."""
    require(isinstance(features,list) and features, 'Missing feature rows')
    require(all(isinstance(r,list) and len(r)==3 and all(type(v) in (int,float) and math.isfinite(v) and v>=0 for v in r) for r in features),'Invalid feature matrix')
    values=np.asarray(features,dtype=np.float32)
    require(np.isfinite(values).all() and np.all(values[0]==0),'Invalid float32 features/initial row')
    normalized=[]; statistics=[]
    for col in values.T:
        median=np.median(col)
        mad=np.median(np.abs(col-median))
        scale=mad*1.4826
        denominator=max(float(scale),1e-3)
        z=(col-median)/denominator
        normalized.append(z)
        statistics.append(dict(median=float(median),mad=float(mad),scaled_mad=float(scale),
                               denominator=denominator,denominator_floor_active=float(scale)<1e-3,
                               upper_clip_count=int(np.count_nonzero(z>10))))
    clipped=[np.clip(z,0,10) for z in normalized]
    base=clipped[0]+.6*clipped[1]+.25*clipped[2]
    new=np.maximum(normalized[0],0)+.6*clipped[1]+.25*clipped[2]
    base[0]=0;new[0]=0
    require(base.dtype==new.dtype==np.float32 and np.isfinite(new).all(),'Invalid recalculated scores')
    return base,new,statistics


def score_hash(values):
    return hashlib.sha256(np.ascontiguousarray(values).tobytes()).hexdigest()


def validate_scores(trace,record):
    numbers=trace['frame_numbers']
    require(numbers and all(type(n) is int for n in numbers) and numbers==sorted(set(numbers)), 'Unsorted/invalid original frame numbers')
    base,new,stats=recompute(trace['features'])
    require(len(base)==len(numbers),'Feature/frame count mismatch')
    for key,expected in (('base_scores',base),('new_scores',new)):
        raw=trace[key]
        require(isinstance(raw,list) and len(raw)==len(numbers) and all(type(v) in (int,float) and math.isfinite(v) for v in raw),'Invalid recorded scores')
        # Exact JSON round-trip of float32, including rejection of sub-float32 tampering.
        require(raw==expected.tolist(),f'{key} differs from independent float32 recalculation')
        require(raw[0]==0,'Initial score must be zero')
    old,changed=trace['baseline'],trace['candidate']
    require(old['collision_frame']==numbers[int(np.argmax(base))],'Baseline argmax/earliest tie/original number mismatch')
    require(changed['collision_frame']==numbers[int(np.argmax(new))],'Candidate argmax/earliest tie/original number mismatch')
    require(all(old[k]==changed[k] for k in ('entry_frame','entry_side','evasion_space')),'Noncollision fields changed')
    detail=trace['diagnostics']['uncapped_jerk']
    require({k:v for k,v in trace['diagnostics'].items() if k!='uncapped_jerk'}==trace['baseline_diagnostics'],'Baseline VLM diagnostic context changed')
    require(detail['old_collision_frame']==old['collision_frame'] and detail['new_collision_frame']==changed['collision_frame'],'Diagnostic frame mismatch')
    require(detail['version']=='uncapped_jerk_v6c' and detail['base_score_hash_available'] is True,'Missing score provenance')
    require(detail['base_score_sha256']==score_hash(base) and detail['new_score_sha256']==score_hash(new),'Score byte hash mismatch')
    require(detail['base_score_dtype']==detail['new_score_dtype']=='float32','Score dtype changed')
    require(detail['changed_target_only'] is True and detail['baseline_context_for_other_fields_retained'] is True,'Unexpected intervention')
    require(detail['collision_changed']==(old['collision_frame']!=changed['collision_frame']),'Changed flag mismatch')
    if 'baseline_trace' in record:
        cached=record['baseline_trace']
        require(trace['base_scores_equal_frozen_V5'] is True,'Development V5 equality missing')
        require(old==cached['prediction'] and trace['baseline_diagnostics']==cached['diagnostics'] and numbers==cached['frame_numbers'],'Development cached baseline mismatch')
        require(np.array_equal(base,np.asarray(cached['motion_scores'],dtype=np.float32)),'Original V5 float32 score bytes differ')
    else:
        require(trace['base_scores_equal_frozen_V5'] is None,'Unexpected reserved cache-equality claim')
    def top(values):
        order=sorted(range(len(numbers)),key=lambda i:(-float(values[i]),numbers[i]))[:2]
        return [dict(frame=numbers[i],score=float(values[i])) for i in order]
    return dict(jerk=stats[0],residual=stats[1],appearance=stats[2],
                baseline_top_two=top(base),candidate_top_two=top(new),
                baseline_score_sha256=score_hash(base),candidate_score_sha256=score_hash(new),
                float32_exact=True,earliest_tie_policy_verified=True)


def reserved_guard(frozen):
    dev=HERE/'uncapped_jerk_v6c_development'
    ep=dev/'evaluation.json';fp=dev/'freeze.json';rp=dev/'report.json'
    require(all(p.is_file() for p in (ep,fp,rp)),'Reserved blocked: development evaluation missing')
    previous=read(ep)
    require(previous['phase']=='development' and previous['gate_passed'] is True,'Reserved blocked: development failed')
    require(previous['report_sha256']==sha(rp) and previous['freeze_sha256']==sha(fp),'Development result binding changed')
    require(previous['evaluator_sha256']==sha(__file__) and previous['protocol_sha256']==sha(PROTOCOL),'Evaluation policy changed')
    require(gate(previous['videos'],previous['offline_calls_valid'])['gate_passed'],'Development numeric gate not reproducible')
    old=read(fp);report=read(rp)
    require(report['status']=='complete' and report['phase']=='development' and report['call_count']==report['max_calls']==0 and report['network_attempts']==0 and report['freeze_sha256']==sha(fp),'Invalid development report')
    require([r['ID'] for r in previous['videos']]==COHORTS['development'],'Unexpected development IDs')
    for path in (PROTOCOL,HERE/'candidates/stage2_uncapped_jerk_v6c.py',HERE/'run_uncapped_jerk_v6c.py'):
        require(old['files'][str(path)]==frozen['files'][str(path)]==sha(path),'Candidate changed between phases')
    for path in (ep,fp,rp,Path(__file__).resolve()):
        require(frozen['files'][str(path)]==sha(path),'Reserved lacks exact development/evaluator binding')
    for path,digest in previous['review_file_sha256s'].items():
        require(frozen['files'][path]==digest==sha(path),'Development review changed')
    return dict(development_evaluation_sha256=sha(ep),development_report_sha256=sha(rp),development_freeze_sha256=sha(fp))


def evaluate(run_dir):
    output=run_dir/'evaluation.json';require(not output.exists(),'Refuse evaluation overwrite')
    fp=run_dir/'freeze.json';frozen=read(fp);phase=frozen['phase']
    require(phase in COHORTS and [v['ID'] for v in frozen['videos']]==COHORTS[phase],'Unexpected phase/IDs')
    # This executes before opening reserved predictions or any reserved review.
    guard=reserved_guard(frozen) if phase=='reserved' else None
    require(frozen['files'][str(PROTOCOL)]==sha(PROTOCOL),'Protocol SHA mismatch')
    protocol=read(PROTOCOL)
    require(protocol['gate_each_cohort']==GATE and protocol['development_ids']==COHORTS['development'] and protocol['reserved_ids']==COHORTS['reserved'],'Gate/phase policy changed')
    for path,digest in frozen['files'].items():require(sha(path)==digest,'Frozen artifact changed: '+path)
    rp=run_dir/'report.json';report=read(rp)
    require(report['phase']==phase and report['freeze_sha256']==sha(fp),'Report/freeze mismatch')
    require([v['ID'] for v in report['videos']]==COHORTS[phase],'Unexpected report IDs')
    count=0 if phase=='development' else 4
    valid=report['status']=='complete' and report['network_attempts']==0 and report['call_count']==report['max_calls']==frozen['max_calls']==3*count
    require(valid,'Incomplete/offline/call-budget violation')
    for trace in report['videos']:
        require(len(trace['calls'])==count and all(c['status']=='complete' for c in trace['calls']),'Per-video call policy changed')
        require([c['max_new_tokens'] for c in trace['calls']]==([] if count==0 else [64,48,40,40]),'Token budget changed')
    rows=[video_result(a,b) for a,b in zip(frozen['videos'],report['videos'])]
    result=dict(created_utc=datetime.now(timezone.utc).isoformat(),phase=phase,**gate(rows,valid),videos=rows,
                offline_calls_valid=valid,freeze_sha256=sha(fp),report_sha256=sha(rp),
                evaluator_path=str(Path(__file__).resolve()),evaluator_sha256=sha(__file__),
                protocol_sha256=sha(PROTOCOL),reserved_guard=guard,
                review_file_sha256s={r['review_path']:r['review_sha256'] for r in rows},
                S2=None,official_accuracy=None,ground_truth_promotion=False,independent_validation_proven=False,
                package_weights_rehashed_by_evaluator=False,
                interpretation='Predeclared three-case contact gate on single-human drafts, not complete S2 or a leaderboard guarantee.',
                missing_fields_policy='Other human labels not read. Three baseline fields checked for value equality only.')
    with output.open('x',encoding='utf-8') as f:json.dump(result,f,ensure_ascii=False,indent=2,allow_nan=False)
    print(json.dumps({k:result[k] for k in ('phase','gate_passed','gained','lost','S2')}))


def self_test():
    def rows(base,new,same=True):
        return [dict(baseline_abs_error_seconds=a,candidate_abs_error_seconds=b,other_fields_identical=same) for a,b in zip(base,new)]
    assert gate(rows([.3,.6,.8],[.3,.3,.7]),True)['gate_passed']
    assert not gate(rows([.3,.6,.8],[.3001,.3,.1]),True)['gate_passed']
    assert not gate(rows([.2,.6,.8],[.2,.3,2]),True)['gate_passed']
    assert not gate(rows([.2,.6,.8],[.2,.3,.7],False),True)['gate_passed']
    assert not gate(rows([.2,.6,.8],[.2,.3,.7]),False)['gate_passed']
    try:gate(rows([.1],[.1]),True)
    except ValueError:pass
    else:raise AssertionError('Incomplete cohort accepted')
    features=[[0,0,0]]*6+[[.1,100,100],[.1,100,100]]
    b,n,s=recompute(features)
    assert b[0]==n[0]==0 and b[6]==b[7] and n[6]==n[7] and n[6]>b[6]
    assert b[6]==18.5 and s[0]['upper_clip_count']==2 and s[0]['denominator_floor_active']
    numbers=[4,7,12,16,25,31,42,59]
    old=dict(collision_frame=42,entry_frame=7,entry_side='LEFT',evasion_space=1)
    detail=dict(version='uncapped_jerk_v6c',old_collision_frame=42,new_collision_frame=42,
                base_score_hash_available=True,base_score_sha256=score_hash(b),new_score_sha256=score_hash(n),
                base_score_dtype='float32',new_score_dtype='float32',changed_target_only=True,
                baseline_context_for_other_fields_retained=True,collision_changed=False)
    trace=dict(frame_numbers=numbers,features=features,base_scores=b.tolist(),new_scores=n.tolist(),
               baseline=old,candidate=dict(old),baseline_diagnostics={},diagnostics={'uncapped_jerk':detail},
               base_scores_equal_frozen_V5=None)
    assert validate_scores(trace,{})['candidate_top_two'][0]['frame']==42
    trace['new_scores'][6]+=1
    try:validate_scores(trace,{})
    except ValueError:pass
    else:raise AssertionError('Score tampering accepted')
    one,other,_=recompute([[0,0,0]])
    assert one.tolist()==other.tolist()==[0.0]
    print('CPU synthetic gate/score/tie/tampering/single-frame contracts PASS; no reports or labels read')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-dir',type=Path);p.add_argument('--self-test',action='store_true');a=p.parse_args()
    if a.self_test:self_test()
    else:require(a.run_dir is not None,'--run-dir required');evaluate(a.run_dir.resolve())
