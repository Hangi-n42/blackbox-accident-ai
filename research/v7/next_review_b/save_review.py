"""Serialize independent AI visual notes; no predictions or other reviewer files."""
import hashlib,json
from datetime import datetime,timezone
from pathlib import Path
HERE=Path(__file__).resolve().parent
SOURCE=HERE.parent/'new_validation_sources'
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def read(p):return json.loads(Path(p).read_text('utf8'))
def unknown(reason):return dict(status='unknown',value=None,reason=reason)
notes={
'00017':dict(applicability='unknown_ego_contact',
    actor='정지 신호 대기 중 자차 바로 앞 은색 Volkswagen 세단. 실제 충돌 상대라고 확정하지 않음.',reference=570,
    contact=None,event=[543,585],
    reason='앞차와의 거리가 줄고 카메라 구도가 변하지만 범퍼 접촉면은 화면 아래에 가려져 있다. 앞차 또는 뒤차와의 물리 접촉을 입증하는 영상 단서를 확보하지 못했다. 정차/미세 전진과 구분 불가.',
    entry_reason='관찰한 은색 앞차는 첫 프레임부터 같은 차로에 있지만 실제 충돌 상대인지 미확인이다. 이를 실제 상대의 entry0으로 승격하지 않는다.',
    side=None,space=None),
'00018':dict(applicability='unknown_ego_contact',
    actor=None,reference=570,contact=None,event=[516,606],
    reason='교차로 정지 후 카메라가 움직인다. 앞쪽에 자차와 접촉하는 차량이 보이지 않고 실제 상대를 특정할 수 없다. 보이지 않는 후방 충돌 또는 단순 출발을 이 영상만으로 구별할 수 없다.',
    entry_reason='실제 상대가 보이지 않아 최초 바퀴 진입이나 시작부터 같은 차로 여부를 판독할 수 없다.',side=None,space=None),
'00019':dict(applicability='ego_contact_inferred_not_directly_visible',
    actor='오른쪽에서 왼쪽으로 자차 코앞을 가로지르는 회색 소형 SUV, 뒤쪽 세로형 적색 램프. 597~645에서 같은 차체를 연속 확인.',reference=624,
    contact=[609,615],event=[570,645],
    reason='609 이전 SUV 앞부분이 자차 코앞으로 들어오고 612 부근에서 구도가 급변하며 차체가 화면 아래를 가로지른다. 이후 SUV가 왼쪽에 남고 자차가 정지한다. 근접만이 아닌 전후 상대 운동을 근거로 접촉을 추론했으나 실제 맞닿은 부위는 가려져 있어 정확한 첫 접촉 프레임은 확정하지 않는다. 급제동/극근접 대안이 완전히 배제된 것은 아니다.',
    entry_reason='같은 SUV는 오른쪽에서 진입하지만 자차가 교차로에서 회전 중이며 투영 차로 경계와 첫 바퀴 접촉이 명확하지 않다. 바퀴는 가까운 구간에서 가려진다.',side='RIGHT',space=None),
'00021':dict(applicability='ego_contact_inferred_not_directly_visible',
    actor='자차와 같은 차로에서 앞서 정차한 흰색 Volvo XC60 SUV. 360 이후부터 근접 구간까지 동일 후면을 확인.',reference=600,
    contact=[600,604],event=[570,630],
    reason='정차한 Volvo에 자차가 접근하여 600~604에서 차간 간격이 소실되고 상대 위치/카메라 구도가 급변한 후 근접 정지 상태가 지속된다. 이후 앞차 운전자가 나와 자차 쪽으로 다가오는 장면은 상황 추론의 보조 근거다. 범퍼 맞닿음은 자차 전면/영상 하단에 가려져 직접 관찰하지 못했다.',
    entry_reason='Volvo는 처음 식별되는 구간부터 같은 차로를 주행하지만 원본 시작에는 회전 전 장면이다. 같은 상대가 원본 첫 프레임부터 이미 차로 안이었다고 확정하거나 entry0으로 만들 수 없다.',side=None,space=1,
    space_reason='600~604 전후 전체 장면에서 자차 오른쪽의 인접 포장 차로가 연속적으로 보이고 바로 옆 차량이 없으며 차로 폭의 주행 공간이 존재한다. 속도/반응시간/회피 성공 가능성은 판단하지 않았다. 접촉 시각 자체가 추론 구간이라는 조건을 유지한다.'),
'00022':dict(applicability='unknown_ego_contact',
    actor='왼쪽 근접 위치에서 자차 바로 앞으로 이동해 정차하는 검은 소형 해치백. 600~645에서 같은 측면/후면을 확인하나 실제 접촉 상대인지는 확정 불가.',reference=618,
    contact=None,event=[579,624],
    reason='검은 해치백이 왼쪽에서 매우 가깝게 끼어든 뒤 앞에서 멈추고 이후 사람이 나온다. 그러나 차체 접촉면이 아래쪽에 가려지고 영상상 확실한 접촉 전후의 불연속을 분리하지 못했다. 극근접 끼어들기/접촉을 판별할 근거가 부족하므로 시간 구간을 실제 접촉 정답으로 만들지 않는다.',
    entry_reason='검은 차량의 왼쪽→앞 이동은 보이나 실제 충돌 상대 확정이 없고 차로 경계와 바퀴 최초 접촉이 보이지 않는다.',side=None,space=None),
'00023':dict(applicability='ego_contact_inferred_not_directly_visible',
    actor='왼쪽에서 오른쪽으로 가로지르는 흰색 단일 캡 픽업, 긴 적재함과 검은 앞 범퍼. 578~630에서 동일 차량을 연속 확인.',reference=600,
    contact=[598,604],event=[570,630],
    reason='흰 픽업이 자차의 붉은 보닛 바로 앞을 지나며 598~604 사이 상대 차체와 카메라 구도가 급격하게 변한다. 이어 픽업이 바로 앞에 정지하고 운전자가 내려온다. 이 전후 증거로 자차 접촉을 추론하지만 최초 맞닿음은 붉은 보닛과 하단 가림 때문에 직접 보이지 않아 단일 프레임 정답은 부여하지 않는다.',
    entry_reason='픽업이 왼쪽에서 자차 진행 경로를 가로지르지만 자차는 도로 진입 위치에 있고 투영 차로 경계를 확정하기 어렵다. 첫 바퀴 접촉 시점을 근접/차체 겹침으로 대체하지 않는다.',side='LEFT',space=None)
}
acquisition=read(SOURCE/'acquisition.json')
assert acquisition['status']=='complete'
assert sha(SOURCE/'selection_plan.json')==acquisition['selection_plan_sha256']
expected={r['ID']:r for r in acquisition['records']}
rows=[]
for ID,note in notes.items():
    folder=HERE/ID;mapping=read(folder/'native_pts.json');source=SOURCE/f'{ID}.mp4'
    assert sha(source)==expected[ID]['sha256']==mapping['source_sha256']
    frames={r['frame']:r for r in mapping['frames']}
    def interval(numbers):
        return None if numbers is None else dict(frame_interval=numbers,pts_seconds_interval=[frames[n]['pts_seconds'] for n in numbers])
    sheets=sorted(folder.glob('overview_*.png'))+sorted(folder.glob('event_*.png'))
    # These sheets were all actually opened with view_image. Native panels were
    # read within sheets, not claimed to have been separately opened full-size.
    evidence=[dict(path=str(p.resolve()),sha256=sha(p),viewed=True,viewing='view_image contact sheet') for p in sheets]
    pixels=[dict(path=str(p.resolve()),sha256=sha(p),viewed_as_sheet_panel=True) for p in sorted(folder.glob('*.png')) if p not in sheets]
    contact=dict(status='inferred_interval' if note['contact'] else 'unknown',exact_frame=None,
                 directly_visible=False,interval=interval(note['contact']),reason=note['reason'],
                 alternative_explanations_not_fully_excluded=True)
    row=dict(ID=ID,record_type='AI_secondary_evidence',reviewer='independent_AI_reviewer_B',
        human_GT=False,official_GT=False,predictions_read=False,other_reviewer_read=False,
        source_path=str(source.resolve()),source_sha256=sha(source),native_frame_count=len(frames),
        native_pts_mapping=dict(path=str((folder/'native_pts.json').resolve()),sha256=sha(folder/'native_pts.json')),
        applicability=note['applicability'],contact=contact,
        same_actor=dict(description=note['actor'],status='inferred_collision_actor' if note['contact'] else 'unconfirmed_or_unknown',
            reference_frame=note['reference'],reference_pts_seconds=frames[note['reference']]['pts_seconds'],
            reference_image=str((folder/f'native_frame{note["reference"]:06d}.png').resolve())),
        entry=unknown(note['entry_reason']),already_entered_at_original_start=unknown(note['entry_reason']),
        side=dict(status='conditional_on_inferred_actor' if note['side'] else 'unknown',value=note['side'],
            reason='동일 추론 상대의 영상상 출발 쪽이며 이동 방향/충돌 후 위치와 구분. 접촉 자체가 추론이라는 조건 유지.' if note['side'] else '실제 상대 또는 원래 진입 쪽을 확정할 수 없음.'),
        space=dict(status='conditional_on_inferred_contact' if note['space'] is not None else 'unknown',value=note['space'],
            reason=note.get('space_reason','실제 접촉 시각/상대 또는 충분한 포장 주행 폭을 확정할 수 없어 공간 클래스를 강제하지 않음.')),
        inspected_event_range_not_contact_GT=interval(note['event']),viewed_evidence=evidence,
        panel_images=pixels,detail_manifest_sha256=sha(folder/'event_manifest.json'),
        limitations=['전체 native frame을 한 번 해독하였으나 사람처럼 모든 프레임을 재생 검수한 것은 아님: 전체1초간격시트와 사건주변2~6프레임간격시트를 시간순으로 관찰.',
                    '장면은 고정6개 모두 유지하며 불확실 사례 교체 없음. 이전14~16 대체 아님.',
                    '직접 관찰 접촉0개. 추론 구간은 공식/사람 정답 또는 정확도 GT가 아님.'])
    with (folder/'review.json').open('x',encoding='utf8') as f:json.dump(row,f,ensure_ascii=False,indent=2)
    rows.append(row)
report=dict(created_utc=datetime.now(timezone.utc).isoformat(),record_type='AI_secondary_evidence',
    reviewer='independent_AI_reviewer_B',selection_plan_sha256=sha(SOURCE/'selection_plan.json'),
    acquisition_sha256=sha(SOURCE/'acquisition.json'),extractor_sha256=sha(HERE/'extract_review.py'),
    serializer_sha256=sha(__file__),source_count=6,source_replacements=0,model_calls=0,GPU_used=False,
    predictions_read=False,other_reviewer_annotations_read=False,exact_contact_labels_created=0,
    inferred_contact_interval_count=3,unknown_contact_count=3,human_GT=False,official_GT=False,
    videos=rows)
with (HERE/'review_summary.json').open('x',encoding='utf8') as f:json.dump(report,f,ensure_ascii=False,indent=2)
print(json.dumps({'status':'frozen','sha256':sha(HERE/'review_summary.json'),'sources':6,'inferred_intervals':3,'unknown':3}))
