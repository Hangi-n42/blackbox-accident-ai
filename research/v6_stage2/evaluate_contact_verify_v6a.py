"""CPU evaluator for the single frozen V6A candidate. No model imports.

python -I -B evaluate_contact_verify_v6a.py --run-dir FROZEN_RUN
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
PROTOCOL = HERE/'contact_verify_v6a_protocol.json'
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


def reserved_guard(frozen):
    dev=HERE/'contact_verify_v6a_development'
    evaluation=dev/'evaluation.json'; fp=dev/'freeze.json'; rp=dev/'report.json'
    require(all(p.is_file() for p in (evaluation,fp,rp)), 'Reserved blocked: development evaluation missing')
    previous=read(evaluation)
    require(previous['phase']=='development' and previous['gate_passed'] is True, 'Reserved blocked: development did not pass')
    require(previous['report_sha256']==sha(rp) and previous['freeze_sha256']==sha(fp), 'Development binding changed')
    require(previous['evaluator_sha256']==sha(__file__), 'Evaluator changed after development')
    require(gate(previous['videos'],previous['offline_calls_valid'])['gate_passed'], 'Development gate not reproducible')
    old=read(fp)
    for path in (PROTOCOL,HERE/'candidates/stage2_contact_verify_v6a.py',HERE/'run_contact_verify_v6a.py'):
        require(old['files'][str(path)]==frozen['files'][str(path)]==sha(path), 'Frozen policy differs across phases')
    require(frozen['files'][str(evaluation)]==sha(evaluation) and frozen['files'][str(fp)]==sha(fp), 'Reserved freeze lacks development gate binding')
    return dict(development_evaluation_sha256=sha(evaluation),development_report_sha256=sha(rp),development_freeze_sha256=sha(fp))


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
    d=trace['diagnostics']['contact_verification']; offered=d['offered_original_frames']
    require(trace['calls'][-1]['raw']==d['raw_output'],'Recorded verifier raw output differs')
    require(offered and offered==sorted(set(offered)) and len(offered)<=18 and set(offered).issubset(valid),'Invalid offered candidates')
    require(d['old_collision_frame']==old['collision_frame'] and d['new_collision_frame']==new['collision_frame'],'Verification output mismatch')
    centers=[trace['baseline_diagnostics']['collision_replacement']['base_collision_frame'],old['collision_frame']]
    require(d['centers_original_frames']==centers and set(centers).issubset(offered),'Centers not retained')
    require(d['window_original_frames']==[[valid[i] for i in w] for w in d['window_indices']],'Window numbering mismatch')
    require(sorted(set(sum(d['window_original_frames'],[])))==offered,'Window union mismatch')
    for center,window in zip(centers,d['window_indices']):
        ci=valid.index(center)
        require(len(window)<=9 and ci in window and all(max(0,ci-10)<=i<=min(len(valid)-1,ci+10) for i in window),'Invalid local window')
    if d['accepted']:
        require(d['fallback'] is False and type(d['parsed_output']) is dict and type(d['parsed_output'].get('collision_frame')) is int and d['parsed_output']['collision_frame']==new['collision_frame'] and new['collision_frame'] in offered,'Invalid accepted verifier output')
    else:
        require(d['fallback'] is True and old['collision_frame']==new['collision_frame'],'Fallback changed motion output')
    if 'baseline_trace' in record:
        require(old==record['baseline_trace']['prediction'] and trace['baseline_diagnostics']==record['baseline_trace']['diagnostics'] and valid==record['baseline_trace']['frame_numbers'],'Development baseline reuse mismatch')
    nearest=min(offered,key=lambda n:(abs(times[n]-contact['pts_seconds']),times[n],n))
    signed_old=times[old['collision_frame']]-contact['pts_seconds']; signed_new=times[new['collision_frame']]-contact['pts_seconds']
    correct_old=abs(signed_old)<=.3+EPS;correct_new=abs(signed_new)<=.3+EPS
    return dict(ID=ID,review_path=str(review_path),review_sha256=sha(review_path),source_path=source['source_path'],source_sha256=sha(source['source_path']),
                label_binding=binding,human_contact_draft=contact,baseline=old,candidate=new,
                baseline_signed_error_seconds=signed_old,candidate_signed_error_seconds=signed_new,
                baseline_abs_error_seconds=abs(signed_old),candidate_abs_error_seconds=abs(signed_new),
                baseline_correct=correct_old,candidate_correct=correct_new,gained=not correct_old and correct_new,lost=correct_old and not correct_new,
                other_fields_identical=all(old[k]==new[k] for k in ('entry_frame','entry_side','evasion_space')),
                candidate_coverage=dict(offered_frames=offered,exact_human_frame_included=contact['frame'] in offered,nearest_frame=nearest,
                    nearest_pts_seconds=times[nearest],nearest_abs_error_seconds=abs(times[nearest]-contact['pts_seconds']),
                    possible_within_tolerance=abs(times[nearest]-contact['pts_seconds'])<=.3+EPS,selected_within_tolerance=correct_new),
                prior_internal_collision_frame=centers[0],prior_fine_candidates=trace['baseline_diagnostics'].get('collision_candidates'),
                new_windows=d['window_original_frames'],verifier_fallback=d['fallback'],verifier_fallback_reason=d['fallback_reason'],
                missing_noncontact_labels={k:review['review'].get(k) for k in ('entry','side','space')},
                label_scope='Single-human local contact reference only; not official/adjudicated GT')


def evaluate(run_dir):
    output=run_dir/'evaluation.json';require(not output.exists(),'Refuse evaluation overwrite')
    fp=run_dir/'freeze.json';frozen=read(fp);phase=frozen['phase']
    require(phase in COHORTS and [x['ID'] for x in frozen['videos']]==COHORTS[phase],'Unexpected phase/ID set')
    # Guard before opening any reserved prediction report or label records.
    guard=reserved_guard(frozen) if phase=='reserved' else None
    require(frozen['files'][str(PROTOCOL)]==sha(PROTOCOL),'Protocol binding mismatch')
    protocol=read(PROTOCOL)
    require(protocol['gate_each_cohort']==GATE and protocol['development_ids']==COHORTS['development'] and protocol['reserved_ids']==COHORTS['reserved'],'Unexpected protocol gate')
    for path,digest in frozen['files'].items():require(sha(path)==digest,f'Frozen artifact changed: {path}')
    rp=run_dir/'report.json';report=read(rp)
    require(report['freeze_sha256']==sha(fp) and report['phase']==phase,'Report/freeze mismatch')
    require([x['ID'] for x in report['videos']]==COHORTS[phase],'Unexpected prediction ID set')
    calls=1 if phase=='development' else 5
    offline_calls_valid=report['status']=='complete' and report['network_attempts']==0 and report['call_count']==report['max_calls']==frozen['max_calls']==3*calls
    require(offline_calls_valid,'Incomplete/offline/call budget violation')
    for trace in report['videos']:
        require(len(trace['calls'])==calls and all(c['status']=='complete' for c in trace['calls']),'Per-file call mismatch')
        require([c['max_new_tokens'] for c in trace['calls']]==([48] if phase=='development' else [64,48,40,40,48]),'Token policy mismatch')
    rows=[video_result(a,b) for a,b in zip(frozen['videos'],report['videos'])]
    decision=gate(rows,offline_calls_valid)
    result=dict(created_utc=datetime.now(timezone.utc).isoformat(),phase=phase,**decision,videos=rows,
                offline_calls_valid=offline_calls_valid,freeze_sha256=sha(fp),report_sha256=sha(rp),
                evaluator_path=str(Path(__file__).resolve()),evaluator_sha256=sha(__file__),protocol_sha256=sha(PROTOCOL),reserved_guard=guard,
                review_file_sha256s={row['review_path']:row['review_sha256'] for row in rows},
                S2=None,official_accuracy=None,ground_truth_promotion=False,independent_validation_proven=False,
                package_weights_rehashed_by_evaluator=False,
                interpretation='Same predeclared gate on three human contact drafts. Gate pass is a small local component result, not a leaderboard guarantee or complete S2.',
                missing_fields_policy='Unknown side/entry remain unknown; other baseline fields checked for byte-value equality, not claimed correct.')
    with output.open('x',encoding='utf-8') as f:json.dump(result,f,ensure_ascii=False,indent=2,allow_nan=False)
    print(json.dumps({k:result[k] for k in ('phase','gate_passed','gained','lost','S2')}))


def self_test():
    def rows(base,new,unchanged=True):
        return [dict(baseline_abs_error_seconds=a,candidate_abs_error_seconds=b,other_fields_identical=unchanged) for a,b in zip(base,new)]
    assert gate(rows([.3,.6,.8],[.3,.3,.7]),True)['gate_passed']
    assert gate(rows([.3,.6,.8],[.3+EPS/2,.3,.7]),True)['gate_passed']
    assert not gate(rows([.3,.6,.8],[.3001,.3,.1]),True)['gate_passed']
    assert not gate(rows([.2,.6,.8],[.2,.5,.7]),True)['gate_passed']
    assert not gate(rows([.2,.6,.8],[.2,.3,2.0]),True)['gate_passed']
    assert not gate(rows([.3,.6,.8],[.3,.3,.7],False),True)['gate_passed']
    assert not gate(rows([.3,.6,.8],[.3,.3,.7]),False)['gate_passed']
    try:gate(rows([.1],[.1]),True)
    except ValueError:pass
    else:raise AssertionError('Incomplete cohort accepted')
    print('8 pure gate contracts PASS; no run reports, labels or predictions read')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run-dir',type=Path);p.add_argument('--self-test',action='store_true');a=p.parse_args()
    if a.self_test:self_test()
    else:require(a.run_dir is not None,'--run-dir required');evaluate(a.run_dir.resolve())
