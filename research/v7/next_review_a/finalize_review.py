"""Bind independently viewed AI notes to native evidence; no prediction reads."""
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent / 'new_validation_sources'
def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):
    return json.loads(p.read_text(encoding='utf-8'))

notes = [
 dict(ID='00017', ego_contact='unknown', actor=None,
      candidate_actor='White/silver Volkswagen sedan directly ahead; collision identity unconfirmed.',
      interval=None, side=None, space=None,
      observation='Night congestion: ego closes on the Volkswagen rear, whose lower bumper disappears behind the hood. Fine frames 556–606 show approach but no visible touch, deformation, debris, or uniquely attributable impact. Later the sedan moves away. Proximity alone does not establish contact.',
      entry_note='The candidate sedan is already ahead in the ego lane at frame 0, but this is not promoted to entry=0 because its collision identity is unknown.',
      uncertainty='An occluded contact cannot be excluded. Near-miss or rear impact outside view also cannot be established.', native=[]),
 dict(ID='00018', ego_contact='unknown', actor=None, candidate_actor=None,
      interval=None, side=None, space=None,
      observation='Ego waits at an intersection and later changes orientation. Crossing vehicles early and distant vehicles remain visible, but no close counterpart or visible physical contact is established in the chronology, event 460–640, or fine 510–554.',
      entry_note='No collision counterpart can be traced, so neither lane entry nor side can be assigned.',
      uncertainty='Camera movement alone does not establish rear impact. The pedestrian, crossing van, and distant truck are not assigned as collision actors.', native=[]),
 dict(ID='00019', ego_contact='inferred', actor='Gray Honda CR-V/crossover crossing immediately ahead, right to left.',
      candidate_actor=None, interval=[608,616], side='RIGHT', space=None,
      observation='The crossover passes extremely close across the hood; frame 612 shows an abrupt image jolt/blur and the vehicle subsequently stops to the left. This supports an ego-contact inference, but the lower contact surface is occluded.',
      entry_note='The same actor approaches from image right. First wheel contact with the ego-lane extension cannot be resolved because of occlusion and intersection geometry.',
      uncertainty='First physical touch is not directly visible; the interval is inferred, not an exact annotation. No reliable full-width usable-space judgment.', native=[610]),
 dict(ID='00021', ego_contact='inferred', actor='White Volvo XC60 SUV followed in the ego lane.',
      candidate_actor=None, interval=[598,606], side=None, space=1,
      observation='The Volvo brakes, ego closes, and frame 602 shows an abrupt image jolt at very close rear separation. Later the Volvo driver exits and walks back. Hood/privacy masking hides the first touching surfaces.',
      entry_note='Once confidently identified, the Volvo is already ahead. The initial ego turn and distant vehicles prevent tracing its first wheel entry back to frame 0; entry and origin remain unknown.',
      uncertainty='Contact interval is inferred. Space=1 is a conditional AI judgment of visible paved adjacent road to the right at the contact interval; no calibrated width or reaction-time claim.', native=[602]),
 dict(ID='00022', ego_contact='inferred', actor='Black compact hatchback crossing from the immediate left into the road ahead.',
      candidate_actor=None, interval=[598,610], side='LEFT', space=0,
      observation='The black hatchback crosses very close to the left/front of the hood, the ego orientation changes, and both vehicles stop with a later driver exit. This supports a contact inference; the physical contact surface stays below the visible hood boundary.',
      entry_note='Left-origin trajectory is visible for the same actor; exact wheel entry is hidden and the merge lane boundary is unclear.',
      uncertainty='Inference is weaker than a directly visible touch; a very close pass plus steering is an alternative not conclusively excluded. Space=0 is conditional on this contact interpretation: right curb/grass and close left traffic constrain the visible paved corridor.', native=[604]),
 dict(ID='00023', ego_contact='inferred', actor='White Ford Ranger pickup turning across the ego hood.',
      candidate_actor=None, interval=[596,600], side='LEFT', space=None,
      observation='Pickup front wheel/lower body approaches the red hood at 596. By 599–600 the hood contour is locally distorted with new flecks and image displacement. The pickup stops across the hood and its driver later exits. Damage/contact consequences are visible, but first touching surfaces are below the hood.',
      entry_note='Immediately before crossing the ego path the pickup moves image-left to image-right. Earlier main-road approach is different; LEFT describes the local pre-entry origin only. The unmarked driveway extension and obscured wheel boundary prevent a first-entry frame.',
      uncertainty='Earliest-contact interval is inferred from before/after appearance, not exact direct touch. Usable full-vehicle road width to either side is not established.', native=[596,599,600]),
]

acq = read(SOURCE / 'acquisition.json')
assert acq['status'] == 'complete'
sources = {r['ID']:r for r in acq['records']}
records=[]
for note in notes:
    ident=note['ID']; folder=HERE/ident; source=sources[ident]
    assert sha(SOURCE/f'{ident}.mp4') == source['sha256']
    mapping_path=folder/'native_mapping.json'; mapping=read(mapping_path)
    frames={r['frame']:r for r in mapping['frames']}
    sheets=[]; evidence={}
    for name in ['overview','event','fine'] + (['entry'] if ident=='00021' else []):
        mp=folder/f'{name}_manifest.json'; m=read(mp)
        assert sha(folder/f'{name}.jpg') == m['sheet_sha256']
        entries=m['selected']
        for entry in entries: evidence[entry['frame']]=entry
        sheets.append(dict(path=str(folder/f'{name}.jpg'), sha256=m['sheet_sha256'],
                           manifest_path=str(mp),manifest_sha256=sha(mp), indices=m['indices'],
                           viewed=True, presentation='resized contact sheet; native PTS captions'))
    interval=note.pop('interval'); native=note.pop('native')
    bound=None if interval is None else [frames[i] for i in interval]
    native_evidence=[]
    for index in native:
        p=folder/f'frame_{index:06d}.png'
        native_evidence.append(dict(**frames[index],path=str(p),sha256=sha(p),
                                   RGB_sha256=evidence.get(index,{}).get('RGB_sha256'),viewed_individually=True))
    records.append(dict(**note, evidence_type='AI_secondary_evidence',
                        source=source,source_sha256_post_review_verified=True,
                        native_mapping_path=str(mapping_path),native_mapping_sha256=sha(mapping_path),
                        native_frame_count=len(frames),contact_exact_frame=None,
                        directly_observed_first_contact_interval=None,
                        inferred_contact_interval=bound, entry_frame=None,entry_interval=None,
                        already_in_lane_at_start=None,
                        field_conditions='Side and space judgments are conditional on the stated contact/actor inference, not independent ground truth.',
                        viewed_sheets=sheets,individually_viewed_native_frames=native_evidence))
report=dict(status='independent_review_frozen',reviewer='stage1_research',
            evidence_type='AI_secondary_evidence', official_ground_truth_count=0,human_review_count=0,
            source_selection='Six fixed IDs selected before visual review; all retained including unknowns.',
            selection_plan_sha256=sha(SOURCE/'selection_plan.json'),acquisition_sha256=sha(SOURCE/'acquisition.json'),
            blinding='Other reviewer labels and all model predictions were not read.',
            method='36 uniform native-index frames cover each full chronology, followed by event and fine sheets; only explicitly listed native PNGs were viewed individually. Not every source frame was visually reviewed.',
            timing='Zero-based decoded original indices; native PTS/time_base. ID00019 is 30.6Hz; no universal 30fps conversion.',
            uncertainty_policy='No forced exact labels. Inferred intervals distinguish occluded touch from visible consequences. Unknown actor prevents conditional candidate observations being promoted to task labels.',
            records=records)
out=HERE/'review.json'
if out.exists(): raise FileExistsError(out)
out.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
lines=['# 독립 AI 영상 검토 A','',
       '고정된 6개 원본을 모두 검토했다. 사람 검수·공식 정답은 0개이며, 아래는 AI 보조 관찰이다. 다른 검토자의 라벨이나 모델 예측을 읽지 않았다.', '',
       '전체 영상은 36개 균등 시점으로 확인하고 사건 전후를 확대했다. 모든 원본 프레임을 눈으로 확인했다는 의미는 아니다. 직접 최초 접촉면은 여섯 영상 모두 확정하지 못했다. 추정 구간을 정확 프레임으로 사용하면 안 된다.', '',
       '|ID|자차 접촉 판단|추정 접촉 원본 프레임 구간|진입측(조건부)|물리 공간(조건부)|',
       '|---|---|---|---|---|']
for r in records:
    b=r['inferred_contact_interval']; interval='미상' if b is None else f"{b[0]['frame']}–{b[1]['frame']} ({b[0]['seconds']:.6f}–{b[1]['seconds']:.6f}초)"
    lines.append(f"|{r['ID']}|{r['ego_contact']}|{interval}|{r['side'] if r['side'] is not None else '미상'}|{r['space'] if r['space'] is not None else '미상'}|")
lines += ['', '차선 진입 시점 및 시작부터 진입 여부는 모두 미상이다. 화면에 처음 나타난 시점을 차선 진입으로 바꾸지 않았다. 00017의 전방 Volkswagen은 첫 화면부터 차선 안에 있지만 접촉 대상인지 불명확하다. 00021의 Volvo도 식별 가능한 시점에는 전방에 있으나 초기 회전 이전까지 동일 대상을 연결할 수 없다.', '',
          '00019의 시간은 1/19584 time base의 native PTS를 사용했다(30.6Hz). 나머지 영상과 일률적인 30fps 환산을 하지 않았다. 원본 SHA, 전체 native mapping SHA, 실제 본 sheet의 SHA, 개별 확대 PNG SHA/RGB SHA 및 불확실성은 review.json과 연결된 manifest에 보존했다.', '',
          '접촉 추정 근거는 00019의 매우 가까운 횡단·흔들림·후속 정지, 00021의 급접근·흔들림·운전자 하차, 00022의 근접 횡단·자세 변화·후속 정지, 00023의 후드 외형 변화와 새 파편성 흔적이다. 이 중 00022는 근접 통과와 조향이라는 대체 설명을 확실히 배제하지 못한다. 00017/18은 근접 또는 카메라 움직임만으로 접촉을 확정하지 않았다.', '',
          '00021의 공간 1과 00022의 공간 0은 해당 추정 접촉 구간의 보이는 도로 공간에 대한 조건부 AI 판독이며 차량 폭 보정이나 회피 반응시간 판정이 아니다. 00023의 LEFT는 접촉 직전 경로를 기준으로 하며 앞선 큰 도로의 접근 방향과 구분한다. 다른 판독자가 미상이라면 이를 강제 확정할 근거로 쓰지 않는다.', '',
          f"review.json SHA256: {sha(out)}", '']
(HERE/'review.md').write_text('\n'.join(lines),encoding='utf-8')
print(json.dumps({'records':len(records),'review_sha256':sha(out),'path':str(out)}))
