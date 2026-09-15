"""Serialize the agent's completed pixel review; no model or candidate outputs."""
import hashlib,json
from pathlib import Path
from datetime import datetime,timezone
HERE=Path(__file__).resolve().parent
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def interval(mapping,start,end):
    by={r['frame']:r for r in mapping['frames']}
    return dict(frame_interval=[start,end],pts_seconds_interval=[by[start]['pts_seconds'],by[end]['pts_seconds']],
                meaning='AI-selected observed-event inspection interval, NOT a verified contact interval')

notes={
'00014':dict(
    applicability='uncertain_ego_physical_contact',observed_event=[574,590],
    observed_facts=[
        '교차로에서 카메라 차량이 회전한다. 전방 맞은편 차량 여러 대와 횡단보도/교통섬이 보인다.',
        '576부터578 사이 배경과 수평선이 급격히 기울고 이후 프레임에도 회전/흔들림이 이어진다.',
        '576 원본에서 짙은색 정면 차량과 그 왼쪽 밝은색 차량은 보이나 카메라 차량과의 차체 접촉면은 보이지 않는다.',
        '이후 카메라는 교차로 쪽을 바라보고 거의 같은 위치에서 다른 차량들이 지나간다.'],
    counterpart=dict(status='unknown_collision_counterpart',description=None,
                     visible_candidates=['정면의 짙은색 세단','그 옆 밝은색 차량','이전에 앞을 가로지른 흰색 SUV'],
                     reason='회전/흔들림만으로 실제 접촉 차량을 지정할 수 없다. 화면 밖 상대도 배제할 수 없다.'),
    contact_reason='직접적인 최초 접촉점이 보이지 않는다. 카메라 변화는 관찰했으나 차량 충돌과 다른 원인을 픽셀만으로 분리하지 못했다.',
    entry_reason='실제 충돌 상대와 회전 중 ego 차로 연장 경계를 확정하지 못하므로 첫 wheel-entry 구간을 부여하지 않는다.',
    side=None,space=None,conditional=[],
    evidence=['overview_00.png','overview_01.png','overview_02.png','event_16_21_00.png','event_16_21_01.png','event_16_21_02.png','dense_574_590_00.png','native_frame000576.png']),
'00015':dict(
    applicability='close_pass_observed_physical_contact_uncertain',observed_event=[580,600],
    observed_facts=[
        '고속도로에서 짙은 회색 세단이 카메라 차량 바로 앞에 있고 밝은색 SUV는 오른쪽 차로에 있다.',
        '580~591에서 앞 세단 후면이 커진 뒤 화면 왼쪽으로 이동하며 카메라 차량이 오른쪽 차로 쪽으로 지나간다.',
        '588 및591 원본에서 세단 후측면과 카메라 차량의 간격은 작지만 차체가 닿는 면은 하단 가림/시야 밖에 있다.',
        '592~599에서 붉은 물체가 화면 왼쪽 위쪽에 나타나고 움직인다. 이것을 파손물이나 충돌 확증으로 지정하지 않는다.',
        '이후 카메라 차량은 계속 주행한다.'],
    counterpart=dict(status='same_visible_sedan_identified_contact_unconfirmed',description='영상 시작부터 바로 앞 차로를 주행한 짙은 회색 세단',
                     reason='주행 위치와 외형이 이어진다. 오른쪽 밝은 SUV와 구별된다. 다만 실제 접촉 상대라는 결론은 조건부다.'),
    contact_reason='근접 통과는 관찰되지만 접촉면/첫 물리접촉은 직접 확인되지 않는다. 흔들림과 붉은 물체 이동만으로 충돌을 확정하지 않는다.',
    entry_reason='접촉 상대가 이 세단이라는 조건에서 시작부터 같은 ego 차로 안에 있다. 실제 접촉 상대가 미확정이므로 주 평가 entry는 unknown으로 유지한다.',
    side=None,space=None,
    conditional=[dict(field='entry_frame',value=0,status='conditional_AI_observation',condition='실제 접촉 상대가 해당 앞 회색 세단인 경우',
                      reason='원본 첫 프레임에서 이미 ego 차로 전방에 있다.'),
                 dict(field='evasion_space',value=1,status='conditional_AI_observation',condition='580~600 근접 사건을 접촉 문맥으로 가정한 경우',
                      reason='오른쪽 인접 차로에 실제 진입할 수 있는 도로가 보이며 카메라 차량이 그쪽으로 진행한다. 속도/성공가능성 판정은 하지 않았다.')],
    evidence=['overview_00.png','overview_01.png','overview_02.png','event_18_22_00.png','event_18_22_01.png','dense_580_600_00.png','native_frame000588.png','native_frame000591.png']),
'00016':dict(
    applicability='close_side_approach_and_stop_physical_contact_uncertain',observed_event=[530,576],
    observed_facts=[
        '도심 회전 후 카메라 차량이 우측 도로 가장자리의 은색 SUV 옆으로 접근한다. 은색 SUV 앞에는 파란 차량이 있다.',
        '530~576에서 은색 SUV 옆면이 화면 오른쪽을 크게 차지하고 카메라 차량이 옆에 멈춘다.',
        '544 및549 원본에서 SUV 바퀴와 도로 가장자리 표시 일부가 보이지만 카메라 차량과 닿을 수 있는 하단 부분은 가려져 있다.',
        '이후 같은 위치에서 정지한 장면이 이어지고 뒤쪽 시간대에는 사람이 차량 옆으로 온다. 이것만으로 사고를 확정하지 않는다.'],
    counterpart=dict(status='same_visible_silver_SUV_identified_contact_unconfirmed',description='우측 은색 SUV, 바로 앞에 파란 차량이 있는 차량',
                     reason='사건 전후 동일 외형과 앞 파란 차량의 배치가 유지된다. 물리접촉 상대라는 결론은 미확정이다.'),
    contact_reason='하단 접촉 가능 부분이 가려져 실제 최초 접촉이 보이지 않는다. 옆 정지와 이후 사람 등장만으로 접촉을 확정하지 않는다.',
    entry_reason='은색 SUV의 첫 바퀴가 ego 주행 차로 경계를 넘어온 장면을 분리하지 못했다. 카메라 차량의 접근과 상대 차량의 진입을 같은 사건으로 치환하지 않는다.',
    side=None,space=None,
    conditional=[dict(field='entry_side',value='RIGHT',status='conditional_AI_spatial_observation',condition='은색 SUV를 상대라고 가정할 때의 위치',
                      reason='상대 차량은 이미지 오른쪽 도로 가장자리에서 보인다. 실제 lane-entry 발생과 최초 방향은 별도로 미확정이다.'),
                 dict(field='evasion_space',value=1,status='conditional_AI_observation',condition='530~576 접근/정지 구간을 접촉 문맥으로 가정한 경우',
                      reason='차량 왼쪽과 전방에 차가 진입할 수 있는 도로가 보인다. 접촉 정확한 시점과 속도/회피 성공은 판정하지 않는다.')],
    evidence=['overview_00.png','overview_01.png','overview_02.png','event_15_20_00.png','event_15_20_01.png','event_15_20_02.png',
              'dense_530_576_00.png','dense_530_576_01.png','dense_530_576_02.png','native_frame000544.png','native_frame000549.png'])}

acquisition=read(HERE.parent/'fresh_sources_retry1/acquisition.json')
source_sha={r['ID']:r['sha256'] for r in acquisition['records']}
reviews=[]
for ID,note in notes.items():
    folder=HERE/ID;mapping=read(folder/'native_pts.json')
    assert sha(HERE.parent/'fresh_sources_retry1'/f'{ID}.mp4')==source_sha[ID]==mapping['source_sha256']
    evidence={str(folder/p):sha(folder/p) for p in note.pop('evidence')}
    observed=note.pop('observed_event')
    review=dict(ID=ID,label_origin='AI_secondary_evidence',is_human=False,is_official_GT=False,
        source_sha256=mapping['source_sha256'],native_pts_path=str(folder/'native_pts.json'),native_pts_sha256=sha(folder/'native_pts.json'),
        native_frame_count=mapping['frame_count'],full_decode_complete=True,overview_sampling='first native frame at/after each integer second',
        dense_inspection='native PNG every frame in selected event window; no model candidate/GT used to choose it',
        observed_event=interval(mapping,*observed),contact=dict(status='unknown',frame=None,pts_seconds=None,verified_interval=None,reason=note.pop('contact_reason')),
        entry=dict(status='unknown',frame=None,pts_seconds=None,verified_interval=None,reason=note.pop('entry_reason')),
        entry_side=dict(status='unknown',value=note.pop('side')),evasion_space=dict(status='unknown',value=note.pop('space')),
        accuracy_evaluation_eligible=False,evidence_sha256=evidence,**note)
    with (folder/'review.json').open('x',encoding='utf-8') as f:json.dump(review,f,ensure_ascii=False,indent=2)
    reviews.append(review)
report=dict(created_utc=datetime.now(timezone.utc).isoformat(),reviewer='Codex /root/stage2_research AI visual review',
    evidence_type='AI_secondary_evidence',source_ids=['00014','00015','00016'],fixed_source_selection=True,
    model_predictions_for_these_sources_read=False,model_or_GPU_inference_performed=False,
    human_GT_created_or_promoted=False,contact_confirmed_count=0,exact_contact_evaluable_count=0,
    limitations=['No audio review. Images alone cannot establish unseen physical contact.',
                 'Event inspection windows are not verified contact intervals and must not be scored as GT.',
                 'Conditional field observations are not accepted four-target labels.',
                 'Root provided aggregate old-nine development outcome during review; no fresh-three predictions were accessed.'],
    acquisition_sha256=sha(HERE.parent/'fresh_sources_retry1/acquisition.json'),
    source_plan_sha256=sha(HERE.parent/'fresh_sources_retry1/plan.json'),
    extractor_sha256=sha(HERE/'extract_review.py'),serialization_script_sha256=sha(__file__),reviews=reviews)
with (HERE/'review_summary.json').open('x',encoding='utf-8') as f:json.dump(report,f,ensure_ascii=False,indent=2)
print(json.dumps(dict(status='complete',reviews=len(reviews),contact_confirmed=0,exact_contact_evaluable=0)))
