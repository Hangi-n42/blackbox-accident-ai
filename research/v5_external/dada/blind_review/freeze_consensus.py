"""Freeze independent visual reconciliation and a prediction-free external filter."""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
DATA = HERE.parent

def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for part in iter(lambda:f.read(1024*1024),b''):
            h.update(part)
    return h.hexdigest()

def write_new(path, value):
    if path.exists():
        raise RuntimeError(f'Frozen output exists: {path.name}')
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')

sources=['root_labels.json','agent1_labels.json','root_reserve_labels.json','labels_reserve_agent1.json']
docs={name:json.loads((HERE/name).read_text(encoding='utf-8')) for name in sources}
r0={r['ID']:r for r in docs['root_labels.json']['clips']}
a0={r['ID']:r for r in docs['agent1_labels.json']['labels']}
r1={r['ID']:r for r in docs['root_reserve_labels.json']['clips']}
a1={r['ID']:r for r in docs['labels_reserve_agent1.json']['labels']}
selected=['DADA_8_036','DADA_8_039','DADA_9_001','DADA_10_142','DADA_11_045','DADA_11_129']
time_ids=[x for x in selected if x!='DADA_8_039']
replacement={'DADA_10_120':'DADA_10_155','DADA_49_002':'DADA_49_026','DADA_49_015':'DADA_49_036','DADA_50_172':'DADA_50_133'}
now=datetime.now(timezone.utc).isoformat()
records=[]
for clip in sorted(set(r0)|set(r1)):
    old=clip in r0
    root=(r0 if old else r1)[clip]
    agent=(a0 if old else a1)[clip]
    category='conditional_same_target_origin_hypothesis_filter' if clip in selected else 'diagnostic_only_no_adoption_score'
    if clip in time_ids:
        rr=root['entry_hypothesis']; aa=agent['lane_entry']['interval']
        entry=[min(rr[0],aa[0]),max(rr[1],aa[1])]
    else:
        entry=None
    # A temporal hypothesis is retained only when both independent reviewers supplied one.
    rc=root.get('contact_hypothesis') if old else root.get('contact_interval')
    ac=agent['first_physical_contact']['interval'] if old else agent.get('contact_interval')
    contact=[min(rc[0],ac[0]),max(rc[1],ac[1])] if rc and ac else None
    records.append({
        'ID':clip, 'acquisition_cohort':'initial12' if old else 'fixed_reserve4',
        'original_input_directory':f'research/v5_external/dada/inputs/images/{clip}',
        'source_archive_clip_id':clip.replace('DADA_','').replace('_','/'),
        'review_sources':sources[:2] if old else sources[2:],
        'replacement_requested_for':next((k for k,v in replacement.items() if v==clip),None),
        'replaced_by_acquisition_only':replacement.get(clip),
        'classification':category, 'official_GT_valid':False,
        'conditional_counterpart':root.get('target') if clip in selected else None,
        'origin_side':root.get('entry_side') if clip in selected else None,
        'origin_scope':'conditional on same specified target and inferred ego corridor; does not prove ego contact' if clip in selected else 'not scored',
        'lane_entry_union_interval':entry,
        'lane_entry_exact':None,
        'contact_union_hypothesis_not_scored':contact,
        'collision_exact':None,'evasion_space_GT':None,
        'contact_review_root':root.get('ego_contact'),
        'contact_review_agent1':agent.get('ego_vehicle_contact_status',agent.get('ego_contact_status')),
        'independent_root_observation':root,
        'independent_agent1_observation':agent,
        'caution':('Target initially right but ego-turn/target-motion confound; entry remains unknown.' if clip=='DADA_8_039' else
                   'Multiple targets; do not choose the correct-looking one after seeing predictions.' if clip=='DADA_9_009' else
                   'Frame1 entry was conditional on unconfirmed silver VW target; not admitted to temporal gate.' if clip=='DADA_50_031' else
                   'Reserve acquisition does not promote an unresolved scene to verified ego collision.' if not old else
                   'Union preserves reviewer uncertainty; no midpoint, intersection, or exact GT conversion.')
    })
provenance_paths=['provenance.md','selection_frozen.json','replacements_frozen.json','selected_members.json','overlap_audit.json']
consensus={
    'frozen_at_utc':now,'official_ground_truth_valid_clips':0,
    'review_type':'prediction-blind independent AI review reconciliation, not official/human-expert GT',
    'independent_review_sha256':{name:sha(HERE/name) for name in sources},
    'acquisition_provenance_file_sha256':{name:sha(DATA/name) for name in provenance_paths},
    'acquisition_provenance_handling':'Provenance/selection files hashed opaquely; numeric accident metadata not read by agent1. See original preserved provenance for author/source/license.',
    'prediction_results_read':False,
    'temporal_blinding_limit':'Contact sheets were selected around author accident metadata. Review is not blind to those temporal windows.',
    'conditional_origin_cohort':selected,'entry_interval_cohort':time_ids,
    'diagnostic_only_cohort':[r['ID'] for r in records if r['ID'] not in selected],
    'replacement_mapping':replacement,
    'reserve_outcome':'All four reserve scenes remain diagnostic-only; no further acquisition.',
    'frame_rule':'original 1-based filenames; no FPS/PTS manufactured from video-clock OCR',
    'interval_rule':'union of independent coarse visual hypotheses; 8_039 entry unknown; interval membership is not correctness',
    'records':records}
write_new(HERE/'consensus_frozen.json',consensus)

# Bind exactly the six raw PNG inputs without decoding, resaving, or loading a model.
binding={}
for clip in selected:
    folder=DATA/'inputs/images'/clip
    paths=sorted(folder.glob('*.png'),key=lambda p:int(p.stem))
    nums=[int(p.stem) for p in paths]
    assert paths and nums==sorted(set(nums))
    binding[clip]=[{'name':p.name,'original_number':int(p.stem),'bytes':p.stat().st_size,'sha256':sha(p)} for p in paths]
write_new(HERE/'external_input_binding.json',binding)
code_paths=['solution/stage2_motion_collision.py','solution/stage2_v2.py','solution/stage2_v5_simple.py','solution/vlm_candidate.py','solution/vlm.py']
plan={
    'frozen_at_utc':now,
    'purpose':'single paired V3 versus V5-simple falsification filter; not generalization validation or official accuracy',
    'before_external_inference':True,
    'consensus_sha256':sha(HERE/'consensus_frozen.json'),
    'input_binding_sha256':sha(HERE/'external_input_binding.json'),
    'independent_review_sha256':consensus['independent_review_sha256'],
    'predictor_source_sha256':{p:sha(ROOT/p) for p in code_paths},
    'baseline':'solution.stage2_motion_collision._predict_file with frozen Qwen4B NF4 and V3 renderer/preprocessor',
    'candidate':'solution.stage2_v5_simple._predict_file; single already-defined candidate only',
    'run_binding_required_before_first_prediction':['exact runtime/model asset hashes','runner and audit-wrapper hash','all input hashes match external_input_binding.json'],
    'policy_changes_prohibited':'No predictor/output/prompt/candidate/model changes. Boundary logging only; logging wrapper must be frozen before inference.',
    'input_and_run_order':selected,
    'paired_order':'V3 then SIMPLE within each clip; one fixed pass, no best repeat selection',
    'shared_input':'same original PNG files and frame list, same motion scores, same model/preprocessor settings for both methods',
    'origin_cohort':{r['ID']:r['origin_side'] for r in records if r['ID'] in selected},
    'entry_interval_cohort':{r['ID']:r['lane_entry_union_interval'] for r in records if r['ID'] in time_ids},
    'metric_definition':{
        'origin':'number of label agreements out of six conditional target-origin hypotheses; not Accuracy/F1',
        'entry_frame_distance':'d(p,[L,U]) = max(L-p,0,p-U), measured in original filename-number units',
        'invalid_prediction':'noninteger, missing, nonexistent original number, or invalid side category => contract FAIL; never clamp or silently omit',
        'inside_interval':'d=0 means no contradiction found within a coarse hypothesis, NOT a correct prediction',
        'primary_no_seconds':'Do not report Accuracy@0.3s or convert primary frame distances to real elapsed seconds',
        'secondary_nominal_sensitivity':'Separately list count with d>9 using nominal author 30fps assumption only; not a gate or official timing accuracy. Actual per-PNG PTS unavailable.',
        'contact':'three union contact hypotheses descriptive only, no collision timing adoption gate',
        'space':'no confirmed space labels; record outputs only, never fabricate correctness'},
    'quality_audit':{
        'malformed':'per clip any of four raw responses failing the frozen parser/object contract; count clips and calls separately',
        'field_fallback':'per clip and output field: parser/value failure causing an explicit default or retained fallback; preserve raw response, offered frame list, parser result, and final provenance',
        'intermediate_fallback':'record every consequential intermediate default; report per-clip any fallback and semantic stage separately, do not conflate different query roles',
        'common_field_gate':'for entry_frame, entry_side, evasion_space separately, clips with explicit fallback must not increase; collision_frame is fixed motion for both, not a model-default event',
        'valid_first':'a valid model-selected first original frame is distinct from an invalid default-first fallback; neither proves semantic observation',
        'unobservable':'If malformed/default provenance cannot be compared reliably for either method, quality gate is INDETERMINATE, never assumed zero'},
    'gates':[
        'All six paired outputs pass schema/original-number contract and preserve identical motion collision.',
        'Candidate conditional origin agreement count >= V3 on fixed six.',
        'On each of five entry intervals, candidate frame-distance <= V3.',
        'Five entry distances: candidate sum strictly < V3 sum and candidate maximum <= V3 maximum.',
        'Number of clips with malformed output and number with consequential fallback do not increase; shared-field fallback clip counts do not increase.',
        'Audit fully observable; required unknown audit => INDETERMINATE, not PASS.'
    ],
    'decision':'Any gate FAIL => reject this candidate under external filter. Any unmeasurable required gate => INDETERMINATE. All PASS => survives falsification filter only; does not prove official/generalization performance or authorize automatic deployment.',
    'strict_decrease_edge_case':'If V3 total distance is already zero, strict improvement is impossible and candidate cannot pass; do not weaken rule after results.',
    'gate_limitations':['Side count can hide which conditional examples differ; retain every paired row.',
        'Wide visual unions reduce diagnostic power; do not narrow them after seeing outputs.',
        'No exact official GT in sixteen scenes; six conditional cases are a reviewed subset, not a representative target-domain test.',
        'No evasion-space ground truth, so unchanged or plausible-looking space answers do not establish nonregression.',
        'External filter is supplemental to existing public-contract, runtime and package gates; six clips cannot prove full 60-minute execution.'],
    'after_results_forbidden':['relaxing thresholds','changing union intervals','replacing clips','selecting best fold or repeat','prompt/offset/ROI margin tuning on external outcomes','claiming interval membership is GT success']}
write_new(HERE/'external_plan_frozen.json',plan)
text=['# DADA 독립 시각 합의 및 외부 반증 필터','',
      '**유효한 공식 GT는 0개다.** 원래 12개와 사전 reserve 4개를 모두 보존했다. 모델 예측 전에 대상 기원측 6개와 넓은 진입 시각구간 5개만 조건부 가설로 동결했다.','',
      '두 검토자는 모델 예측을 보지 않았다. 그러나 제공 contact sheet는 저자 사고 메타데이터의 시간창 중심이므로 시간창 자체에 blind하지 않았다. 정확한 accident_frame 숫자를 직접 읽지 않았다는 것과 구분한다.','',
      '| 영상 | 합의한 조건부 기원측 | 차선진입 union | 접촉 union: 채점하지 않음 |',
      '|---|---|---|---|']
for r in records:
    if r['ID'] in selected:
        text.append(f"| {r['ID']} | {r['origin_side']} | {r['lane_entry_union_interval'] or 'unknown'} | {r['contact_union_hypothesis_not_scored'] or 'unknown'} |")
text += ['', '8_039는 대상이 오른쪽에 있던 사실과 차선 진입 시점을 분리해 진입을 unknown으로 남겼다. 11_129는 두 검토자의 차이를 [25,68] union으로 보존했다. 어느 구간도 중간값·교집합·단일 프레임 정답으로 바꾸지 않는다.','',
         '진단 전용: '+', '.join(consensus['diagnostic_only_cohort'])+'. 이 중 9_009는 다중 대상, 50_031은 선행 VW가 실제 접촉 상대라는 조건이 미확정이다. 원래 식별 불가 4개와 reserve 4개도 공식 ego-vehicle 사고로 승격하지 않는다. 추가 reserve는 취득하지 않는다.','',
         '원본/교체 관계, 원천 clip ID, 이미지 경로, 취득 provenance 파일 SHA 및 독립 검토 4개 JSON SHA는 consensus_frozen.json에 있다. 독립 원본을 덮어쓰지 않았다.','',
         '## 결과 전 외부 필터','',
         '동일 6개 입력에 V3→SIMPLE 순서로 한 번씩 실행한다. 모델·원본 PNG·처리 규칙을 동결하며 함수 경계의 로그만 허용한다. 실제 모델 자산 및 runner/audit-wrapper 해시를 첫 추론 전에 추가 실행 바인딩으로 고정해야 한다.','',
         '- 방향은 조건부 합의 6개와의 일치 수가 V3 이상이어야 한다. 공식 Accuracy/F1로 부르지 않는다.',
         '- 진입 5개는 d=max(L-p,0,p-U)를 원본 프레임 번호 단위로 계산한다. 각 영상 비악화, 최대 거리 비악화, 총거리 엄격 감소를 모두 요구한다.',
         '- malformed와 명시적 fallback은 영상/출력필드별로 비교한다. 정상 모델이 선택한 첫 프레임과 기본값 첫 프레임을 구분한다. 계측 불가능은 0회가 아니라 INDETERMINATE다.',
         '- 구간 안의 예측은 정답이 아니라 이 넓은 가설과의 모순을 찾지 못한 경우다. 접촉과 공간의 정확도는 계산하지 않는다.',
         '- ±9프레임 분석은 저자 명목 30fps 가정의 별도 민감도 기록으로만 남긴다. 실제 PNG별 PTS가 없으므로 대회 0.3초 정확도를 계산하지 않고 영상 시계 OCR로 FPS GT를 만들지 않는다.',
         '- 모든 기준 통과도 가설 반증 필터를 살아남았다는 뜻뿐이며 일반화 성능 입증이나 자동 제출 승인이 아니다. 실패 후 기준을 완화하지 않는다.','',
         'V3 총거리가 이미 0이면 엄격 감소가 불가능하므로 통과시키지 않는다. 방향 총 일치 수는 개별 교체를 숨길 수 있어 모든 paired 행을 함께 보존한다. 회피 공간 정답은 없어 그 필드 비악화를 이 자료로 검증할 수 없다. 여섯 영상 처리 시간은 전체 60분 상한의 보장이 아니다.','',
         'external_plan_frozen.json은 이 기준 전체와 predictor SHA를 담고, external_input_binding.json은 6개 영상의 모든 원본 PNG 번호·크기·SHA를 담는다.']
(HERE/'consensus_frozen.md').write_text('\n'.join(text)+'\n',encoding='utf-8')
print(json.dumps({'consensus_sha256':sha(HERE/'consensus_frozen.json'),'plan_sha256':sha(HERE/'external_plan_frozen.json'),
    'input_binding_sha256':sha(HERE/'external_input_binding.json'),'input_pngs':sum(map(len,binding.values())),
    'official_GT':0,'conditional_origins':6,'entry_intervals':5,'diagnostic_only':10}))
