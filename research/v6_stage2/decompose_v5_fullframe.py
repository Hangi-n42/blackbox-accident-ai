"""CPU-only decomposition after evaluate_nexar_fullframe_v5.py completes.

No inference, image decode, new labels, candidate scores or aggregate S2.
Run: python -I -B decompose_v5_fullframe.py --run-dir FULL_RUN --output NEW_FOLDER
"""
import sys,os
sys.dont_write_bytecode=True
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='2'
import argparse,json,hashlib,importlib.util,datetime
from pathlib import Path
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
EVALUATOR=HERE/'evaluate_nexar_fullframe_v5.py'
BASE=ROOT/'artifacts/submissions/verify_v5/model/stage2/code/solution/stage2.py'
V2=BASE.with_name('stage2_v2.py')
MOTION=BASE.with_name('stage2_motion_collision.py')
EPS=1e-12
def require(x,m):
    if not x:raise ValueError(m)
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def write(p,x):
    with Path(p).open('x',encoding='utf8') as f:json.dump(x,f,ensure_ascii=False,indent=2,allow_nan=False)
def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);return mod

def coverage(frames,gt,times):
    require(frames==sorted(set(frames)) and all(type(n)is int and n in times for n in frames),'Invalid ordered original-frame candidate list')
    if gt is None:return None
    nearest=min(frames,key=lambda n:(abs(times[n]-gt['pts_seconds']),times[n],n)) if frames else None
    err=None if nearest is None else abs(times[nearest]-gt['pts_seconds'])
    return dict(count=len(frames),frames=frames,exact_GT_frame_included=gt['frame'] in frames,nearest_frame=nearest,
      nearest_abs_error_seconds=err,within_0_3_seconds=err is not None and err<=.3+EPS)

def choice_audit(base,parsed,key,paths,indices,default,expected):
    numbers=[base._frame_number(paths[i]) for i in indices]
    interpreted=base._integer(parsed.get(key));raw=parsed.get(key)
    if interpreted in numbers:mode='offered_value_accepted'
    elif interpreted is not None:mode='out_of_offer_value_snapped_by_frame_number'
    else:mode='missing_or_invalid_value_default_by_valid_index'
    selected=base._choice(parsed,key,paths,indices,default)
    require(base._frame_number(paths[selected])==expected,'Frozen choice policy replay differs from recorded output')
    return dict(raw_field=raw,integer_interpretation=interpreted,mode=mode,policy_selected_frame=expected,
      default_valid_index=default,default_original_frame=base._frame_number(paths[default]),
      output_is_policy_selection=True,not_necessarily_direct_VLM_choice=mode!='offered_value_accepted')

def recorded_answer(call):
    # The original development recorder stores raw_output; the later recorder
    # stores raw. Both are original model text in SHA-bound reports.
    keys=[k for k in ('raw','raw_output') if k in call]
    require(keys and all(isinstance(call[k],str) for k in keys),'Original raw model answer missing')
    require(all(call[k]==call[keys[0]] for k in keys),'Conflicting recorded raw fields')
    return call[keys[0]]

def variant(base,pred,diag,numbers,calls,times,labels,measured):
    require(numbers==sorted(set(numbers)) and set(numbers).issubset(times),'Invalid input frame numbers')
    require(len(calls)>=4 and [c['max_new_tokens'] for c in calls[:4]]==[64,48,40,40] and all(c['status']=='complete' for c in calls[:4]),'Baseline first four calls incomplete')
    paths=[Path(f'frame_{n:06d}.png') for n in numbers] # Number-only replay; never opened.
    parsed=[base._json_object(recorded_answer(c)) for c in calls[:4]]
    for name,p in zip(('coarse','collision','entry','space'),parsed):require(p==diag[name],f'Parsed raw/diagnostic mismatch: {name}')
    internal=diag['collision_replacement']['base_collision_frame'];final=pred['collision_frame']
    require(final==diag['collision_replacement']['collision_frame'] and internal in numbers and final in numbers,'Collision diagnostic mismatch')
    require(final==diag['motion_proposal'],'Final collision differs from recorded motion proposal')
    mi=numbers.index(final)
    coarse_indices=base._uniform_indices(0,len(numbers)-1,10);coarse=[numbers[i] for i in coarse_indices]
    fine=diag['collision_candidates'];entry=diag['entry_candidates']
    require(all(n in numbers for n in fine+entry),'Candidate not in valid input')
    ci=numbers.index(internal);prefix=numbers[:ci+1]
    require(entry==[numbers[i] for i in base._uniform_indices(0,ci,12)],'Entry candidates differ from frozen uniform prefix policy')
    space_context=sorted(set([numbers[max(0,ci-2)],internal,numbers[min(len(numbers)-1,ci+2)]]))
    parser=dict(coarse=choice_audit(base,parsed[0],'collision_frame',paths,coarse_indices,mi,diag['coarse_collision_frame']),
      fine=choice_audit(base,parsed[1],'collision_frame',paths,[numbers.index(n) for n in fine],mi,internal),
      entry=choice_audit(base,parsed[2],'entry_frame',paths,[numbers.index(n) for n in entry],0,pred['entry_frame']))
    # Independently replay side/space fallback, preserving their different contexts.
    side=str(parsed[0].get('entry_side','')).upper().strip();side_valid=side in ('LEFT','RIGHT')
    side_value=side if side_valid else 'LEFT';require(side_value==pred['entry_side'],'Side parsing differs')
    space=base._integer(parsed[3].get('evasion_space'));space_valid=space in (0,1)
    space_value=space if space_valid else 0;require(space_value==pred['evasion_space'],'Space parsing differs')
    contact=None
    if labels['contact'] is not None:
        gt=labels['contact'];co=coverage(coarse,gt,times);fi=coverage(fine,gt,times);inp=coverage(numbers,gt,times);iv=coverage([internal],gt,times);fv=coverage([final],gt,times)
        if not inp['within_0_3_seconds']:stage='input_sampling_or_decode_omission'
        elif not fi['within_0_3_seconds']:stage='fine_candidate_omission'
        elif not iv['within_0_3_seconds']:stage='fine_selection_or_parser_failure'
        else:stage='internal_policy_within_tolerance'
        impact='unchanged_correctness'
        if iv['within_0_3_seconds'] and not fv['within_0_3_seconds']:impact='override_lost_correct_contact'
        elif not iv['within_0_3_seconds'] and fv['within_0_3_seconds']:impact='override_gained_correct_contact'
        require(fv['within_0_3_seconds']==measured['contact']['correct'],'Final contact result differs from evaluator')
        contact=dict(input=inp,coarse_offered=co,fine_offered=fi,internal_policy_selected=iv,final_motion_selected=fv,
          internal_failure_stage=stage,motion_override_impact=impact,internal_frame=internal,final_frame=final,
          internal_signed_error_seconds=times[internal]-gt['pts_seconds'],final_signed_error_seconds=times[final]-gt['pts_seconds'],
          coarse_recall_is_not_a_required_fine_recall_condition=True)
    en=None
    if labels['entry'] is not None:
        gt=labels['entry'];inp=coverage(numbers,gt,times);pre=coverage(prefix,gt,times);cand=coverage(entry,gt,times);sel=coverage([pred['entry_frame']],gt,times)
        if not inp['within_0_3_seconds']:stage='input_sampling_or_decode_omission'
        elif not pre['within_0_3_seconds']:stage='internal_contact_prefix_omission'
        elif not cand['within_0_3_seconds']:stage='uniform_entry_candidate_omission'
        elif not sel['within_0_3_seconds']:stage='entry_selection_or_parser_failure'
        else:stage='entry_policy_within_tolerance'
        require(sel['within_0_3_seconds']==measured['entry']['correct'],'Entry result differs from evaluator')
        en=dict(input=inp,prefix=pre,candidates=cand,selected=sel,failure_stage=stage,prefix_end_frame=internal,
          exact_GT_excluded_by_prefix=not pre['exact_GT_frame_included'],prefix_has_no_tolerance_candidate=not pre['within_0_3_seconds'],
          selected_frame=pred['entry_frame'],signed_error_seconds=times[pred['entry_frame']]-gt['pts_seconds'])
    def discrete(name,key,valid,context):
        gt=labels[name]
        result=dict(known=gt is not None,truth=gt,prediction=pred[key],correct=None if gt is None else pred[key]==gt,
          used_fallback=not valid,context=context)
        require((measured[name] is None)==(gt is None),'Unknown treatment differs')
        if gt is not None:require(result['correct']==measured[name]['correct'],'Known categorical result differs')
        return result
    return dict(frame_count=len(numbers),contact=contact,entry=en,parser=parser,
      side=discrete('side','entry_side',side_valid,'first coarse overview call'),
      space=discrete('space','evasion_space',space_valid,dict(internal_contact_frame=internal,offered_frames=space_context)),
      no_recontextualization_after_final_motion_override=True)

def flag(x):return '미상' if x is None else ('O' if x else 'X')
def change(a,b):return f'{a} → {b}'
def label(x):
    return {'unchanged_correctness':'정오 유지','override_lost_correct_contact':'교체로 정답 손실','override_gained_correct_contact':'교체로 정답 획득',
      'input_sampling_or_decode_omission':'입력 표본 누락','internal_contact_prefix_omission':'내부 접촉의 범위 제한',
      'uniform_entry_candidate_omission':'균등 후보 누락','entry_selection_or_parser_failure':'후보 선택/파서 실패',
      'entry_policy_within_tolerance':'허용 오차 내 선택','offered_value_accepted':'제공 번호 수락',
      'out_of_offer_value_snapped_by_frame_number':'범위 밖 번호 대체','missing_or_invalid_value_default_by_valid_index':'무효 응답 기본값'}.get(x,x)
def markdown(rows):
    text=['# V5 전체 프레임 대 기존 10Hz 오류 분해','',
      '모든 셀은 기존 10Hz → 전체 프레임 순서다. 허용 오차는 native PTS 기준 ±0.3초다. 6개 모두 이미 사용한 원천이며 단일 사람 초안의 알려진 항목만 진단한다. 독립 검증·후보 채택·공식 S2가 아니다.','',
      '## 접촉: 후보 포함·내부 정책 선택·최종 광류 교체','',
      '| ID | fine 후보 포함 | 내부 선택 정답 | 최종 정답 | 최종 오차(초) | 최종 교체 영향 |','|---|---|---|---|---|---|']
    for r in rows:
        a,b=r['old10hz']['contact'],r['full']['contact']
        if a is None:text.append(f"| {r['ID']} | 미상 | 미상 | 미상 | 미상 | 미상 |");continue
        text.append('| '+ ' | '.join([r['ID'],change(flag(a['fine_offered']['within_0_3_seconds']),flag(b['fine_offered']['within_0_3_seconds'])),change(flag(a['internal_policy_selected']['within_0_3_seconds']),flag(b['internal_policy_selected']['within_0_3_seconds'])),change(flag(a['final_motion_selected']['within_0_3_seconds']),flag(b['final_motion_selected']['within_0_3_seconds'])),change(f"{a['final_signed_error_seconds']:+.3f}",f"{b['final_signed_error_seconds']:+.3f}"),change(label(a['motion_override_impact']),label(b['motion_override_impact']))])+' |')
    text+=['','## 진입: 접두 구간·균등 후보·선택','',
      'prefix 누락은 정답 프레임 자체 제외와 ±0.3초 안 후보 전무를 별도로 저장한다. 표의 실패 단계는 허용 오차 기준으로 처음 실패한 단계다.','',
      '| ID | prefix 내 후보 존재 | 균등 후보 포함 | 선택 정답 | 실패 단계 |','|---|---|---|---|---|']
    for r in rows:
        a,b=r['old10hz']['entry'],r['full']['entry']
        if a is None:text.append(f"| {r['ID']} | 미상 | 미상 | 미상 | 미상 |");continue
        text.append('| '+' | '.join([r['ID'],change(flag(a['prefix']['within_0_3_seconds']),flag(b['prefix']['within_0_3_seconds'])),change(flag(a['candidates']['within_0_3_seconds']),flag(b['candidates']['within_0_3_seconds'])),change(flag(a['selected']['within_0_3_seconds']),flag(b['selected']['within_0_3_seconds'])),change(label(a['failure_stage']),label(b['failure_stage']))])+' |')
    text+=['','## 알려진 방향·공간 및 파서 경로','',
      '| ID | 방향 정답 | 공간 정답 | coarse 파서 | fine 파서 | entry 파서 |','|---|---|---|---|---|---|']
    for r in rows:
        a,b=r['old10hz'],r['full'];cells=[r['ID'],change(flag(a['side']['correct']),flag(b['side']['correct'])),change(flag(a['space']['correct']),flag(b['space']['correct']))]
        cells += [change(label(a['parser'][k]['mode']),label(b['parser'][k]['mode'])) for k in ('coarse','fine','entry')]
        text.append('| '+' | '.join(cells)+' |')
    text+=['','`offered_value_accepted`는 해석한 번호를 그대로 수락한 경우다. `out_of_offer_value_snapped_by_frame_number`는 제공하지 않은 번호를 가장 가까운 제공 번호로 옮긴 경우다. `missing_or_invalid_value_default_by_valid_index`는 필드가 없거나 무효여서 기본 후보를 택한 경우다. 내부 선택 성공을 모두 VLM의 직접 정답 선택으로 해석하면 안 된다.',
      '', 'fine 후보는 coarse 주변과 별도 광류 후보를 합하므로 coarse 후보 포함 실패를 fine 실패의 필수 원인으로 단정하지 않는다. 방향은 첫 coarse 문맥에서, 공간과 진입은 내부 접촉 문맥에서 나온다. 최종 motion 교체는 그 세 항목의 문맥을 다시 계산하지 않는다. Unknown은 오답이나 정답으로 집계하지 않았다. 이미지/광류 재계산과 모델 호출은 하지 않았다.']
    return '\n'.join(text)+'\n'

def main(run_dir,output):
    require(not output.exists(),'Refuse overwrite')
    fp=run_dir/'freeze.json';rp=run_dir/'report.json';ep=run_dir/'evaluation.json'
    frozen,report,evaluation=read(fp),read(rp),read(ep)
    require(report['status']=='complete' and report['freeze_sha256']==sha(fp),'Full baseline incomplete/unbound')
    require(evaluation['freeze_sha256']==sha(fp) and evaluation['report_sha256']==sha(rp) and evaluation['evaluator_sha256']==sha(EVALUATOR),'Complete evaluation binding changed')
    require(evaluation['S2'] is None and evaluation['independent_validation'] is False,'Wrong evaluation scope')
    ev=load('existing_fullframe_evaluator',EVALUATOR)
    require([x['ID'] for x in evaluation['videos']]==ev.IDS==[x['ID'] for x in report['videos']]==[x['ID'] for x in frozen['videos']],'Six-source order mismatch')
    bound={str(p):sha(p) for p in (Path(__file__),fp,rp,ep,EVALUATOR,BASE,V2,MOTION)}
    for p in (BASE,V2,MOTION):
        key='model/stage2/code/solution/'+p.name
        require(frozen['package_binding']['files'][key]==sha(p),'Actual baseline source changed')
    for p,h in evaluation['review_file_sha256s'].items():require(sha(p)==h,'Human review changed');bound[p]=h
    for p,h in frozen['baseline10hz_refs'].items():require(sha(p)==h,'Old reference changed');bound[p]=h
    output.mkdir(parents=True)
    write(output/'plan_frozen.json',dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),files_sha256=bound,
      tolerance_seconds=.3,epsilon=EPS,IDs=ev.IDS,scope='Post-exposure descriptive decomposition. Frozen evaluator label semantics; no new GT or candidate.',
      selection_failure_rule='First loss of native-time tolerance coverage along valid inputs -> fine / entry prefix -> uniform candidates -> policy-selected frame. Motion override is separate.'))
    base=load('read_only_frozen_stage2',BASE);rows=[]
    for record,trace,erow in zip(frozen['videos'],report['videos'],evaluation['videos']):
        ID=erow['ID'];times={x['frame']:x['pts_seconds'] for x in record['input']['source_frame_pts']}
        # Reuse the exact evaluator function; do not infer missing labels.
        review=read(erow['review_path']);labels=ev.known_labels(review['review'],times);require(labels==erow['labels'],'Evaluated label interpretation changed')
        old,diag,nums,source,ref=ev.old_trace(ID,frozen)
        historical=next(x for x in read(ref['report_path'])['videos'] if x['ID']==ID)
        require(old==erow['old10hz_prediction'] and trace['prediction']==erow['full_prediction'],'Evaluator predictions changed')
        require({x['frame']:x['pts_seconds'] for x in source['source_frame_pts']}==times,'Old/new native PTS mismatch')
        rows.append(dict(ID=ID,labels=labels,source_sha256=erow['source_sha256'],review_sha256=erow['review_sha256'],
          old10hz=variant(base,old,diag,nums,historical['calls'][:4],times,labels,erow['old10hz']),
          full=variant(base,trace['prediction'],trace['diagnostics'],trace['frame_numbers'],trace['calls'],times,labels,erow['full'])))
    require(all(sha(p)==h for p,h in bound.items()),'Bound artifacts changed during diagnostic')
    write(output/'report.json',dict(status='complete_error_decomposition',videos=rows,binding=bound,S2=None,independent_validation=False,
      model_calls=0,labels_reinterpreted=False,pixels_read=0,known_labels_only=True,all_six_sources_exposed=True))
    with (output/'summary.md').open('x',encoding='utf8') as f:f.write(markdown(rows))
    print(json.dumps(dict(status='complete_error_decomposition',rows=len(rows),S2=None,output=str(output))))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run-dir',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();main(a.run_dir.resolve(),a.output.resolve())
