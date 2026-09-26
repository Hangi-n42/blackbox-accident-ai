# New-source first-frame policy independent verification

**PASS.** No new model load or inference by this CPU verifier.

Frozen files: 1793; prior969 preserved;600 source PNG/native PTS checked;48 baseline calls replayed;5 state processor inputs recreated exactly.
Counts: {'actual_model_calls': 53, 'selected_cases': 12, 'entry_eligible': 3, 'state_queried': 5, 'state_reference_scored': 3, 'state_reference_correct': 3, 'known_outside_queried': 2, 'known_outside_inside_responses': 0, 'other_three_unchanged': 12}.
Gate: {'both_strata': True, 'gains': 1, 'losses': 0, 'possible_loss_cases': 0, 'new_false_first': 0, 'known_outside_new_first': 0, 'paired_MAE_nonincrease': True, 'pass': True}.
Shared-truth paired delta bounds: {'accuracy': [0.3333333333333333, 0.3333333333333333], 'mae_seconds': [-0.5666666666666667, -0.5666666666666667]}.

| Case | Baseline entry | Treatment entry | State | Timing reference |
|---|---:|---:|---|---|
| CCD_001498 | 4 | 4 | None | {'status': 'unknown', 'lower_frame': None, 'upper_frame': None, 'evaluation_eligible': False, 'evidence': {'A': '눈 덮인 무차선 통로이며 차량 여러 대가 가까이 있으나 실제 자차 사고상대와 최초 바퀴 진입을 확정하지 못함.', 'B': '좌측 세단이 근접 통과하지만 실제 사고 상대를 고정할 시각 근거가 부족하며 눈 아래 차로 경계도 보이지 않는다.', 'resolution': '두 검수 모두 실제 자차 사고 상대 미확정. 후보가 같다는 것만으로 counterpart_agreement를 true로 하지 않음.'}} |
| CCD_000585 | 32 | 32 | None | {'status': 'unknown', 'lower_frame': None, 'upper_frame': None, 'evaluation_eligible': False, 'evidence': {'A': 'f0는 먼 도로만 보이며 상대 차체·바퀴를 식별할 수 없다. f23–32 접근 상대는 보이나 최초 바퀴와 중앙선 관계를 원거리 압축 및 희미한 선 때문에 좁힐 수 없다.', 'B': 'f26~33 접근 차량과 자차의 우측 이탈은 보이나 접촉/사고 상대의 확정 근거가 부족하다. f0 원거리 개체도 식별 불가.', 'resolution': '두 검수 모두 접근 차량을 추적했으나 실제 접촉 상대는 미확정. f0 식별도 불가.'}} |
| CCD_001278 | 13 | 13 | None | {'status': 'unknown', 'lower_frame': None, 'upper_frame': None, 'evaluation_eligible': False, 'evidence': {'A': 'f8·12·13의 밴 선행 바퀴는 좌측 경계 연장 바깥에 있고 f15–16에 오른쪽 선행 바퀴가 연장 경계에 도달한 것으로 보인다. 비·원거리 흐림으로 단일 프레임은 확정하지 않음.', 'B': 'f5/10/14 밴과 좌측 빗금 구역 경계 주변은 보이나 빗방울·작은 바퀴·먼 거리 때문에 마지막 확실한 진입 전과 첫 진입 후를 고정할 수 없다.', 'resolution': '같은 검은 밴에 합의. A는 f13~16, B는 unknown이므로 규칙에 따라 진입 UNKNOWN 유지. 추가 가상 경계/시점 생성 없음.'}} |
| CCD_001454 | 4 | 4 | None | {'status': 'unknown', 'lower_frame': None, 'upper_frame': None, 'evaluation_eligible': False, 'evidence': {'A': '넓은 교차로 바닥에 자차 차선 경계 또는 일관되게 연장할 선이 보이지 않는다. f30·36 바퀴는 보이나 임의 가상 경계를 정해 최초 진입을 확정할 수 없다.', 'B': 'f24/32 넓은 교차로 내부에서 상대가 접근하나 자차 차로 경계 또는 신뢰할 연장선을 특정할 수 없다.', 'resolution': '같은 은색 해치백에 합의. f0 부재와 넓은 무표시 교차로 때문에 상태·진입·표식 보류.'}} |
| CCD_000103 | 17 | 17 | None | {'status': 'unknown', 'lower_frame': None, 'upper_frame': None, 'evaluation_eligible': False, 'evidence': {'A': '눈·얼음으로 차선이 가려지고 자차 영상 시야도 회전한다. f18·28·36 바퀴와 측면은 보이나 자차 차선의 최초 경계 접촉을 정할 연장선이 없다.', 'B': 'f18/32 동일 측면 차량은 보이나 눈 덮인 무표시 노면과 자차 회전으로 첫 바퀴가 넘는 차로 경계를 재구성할 수 없다.', 'resolution': '같은 흰 구형 차량에 합의. 눈 덮인 경계와 자차 회전으로 진입 미상. 원본 960x720.'}} |
| CCD_000470 | 4 | 4 | None | {'status': 'unknown', 'lower_frame': None, 'upper_frame': None, 'evaluation_eligible': False, 'evidence': {'A': 'f34·36·40에 횡단 상대가 보이나 노면의 차로 구분선이 거의 보이지 않는다. 도로 가장자리만으로 자차 차선 폭을 임의 결정하지 않아 첫 바퀴 경계 시점은 보류.', 'B': 'f35/40 횡단 후 f49 후면이 자차 바로 앞에 있는 동일 세단은 식별되지만, 무표시 도로 중앙의 자차 차로 경계는 특정 불가.', 'resolution': '같은 좌측 횡단 세단에 합의. f0 식별 및 차로 경계는 미상. 선행 소형차를 대신 표시하지 않음.'}} |
| CCD_001237 | 24 | 24 | OUTSIDE | {'status': 'during_clip', 'lower_frame': 8, 'upper_frame': 21, 'evaluation_eligible': True, 'evidence': {'A': 'f0·12에는 우측 출입 공간에 있고 f20에는 앞바퀴가 본선 노면에 걸쳐 있다. f14–18 표지판 기둥/바퀴 가림과 희미한 노면 가장자리 때문에 최초 경계 접촉을 더 좁히지 않음.', 'B': 'f0 및 f8 우측 옆길/도로 가장자리 밖의 동일 차량. f15/18 중간 프레임은 기둥과 원근 때문에 첫 바퀴를 정밀 판독하기 어렵고, f21에는 전륜이 도로 가장자리 연장선 안쪽에 들어와 있어 보수적 f8~21 구간.', 'resolution': 'A f12~20과 B f8~21은 같은 진입 사건의 중첩 구간. 보수적 합집합 f8~21. 기둥 가림과 희미한 가장자리로 더 좁히지 않음.'}} |
| CCD_000052 | 17 | 0 | INSIDE | {'status': 'before_start', 'lower_frame': 0, 'upper_frame': 0, 'evaluation_eligible': True, 'evidence': {'A': 'f0 동일 상대가 자차 앞, 이중 중앙선 우측의 같은 진행 차로에 이미 있다. 0/0은 제출 인코딩이며 물리적 진입 시각은 영상 시작 전으로 미상이다. 뒤의 자차 중앙선 횡단 이후 근접 상황을 최초 진입으로 대체하지 않음.', 'B': 'f0 이중 중앙선 우측에서 자차와 같은 대기 차로에 이미 놓인 두 번째 차량. 0/0은 대회 제출 인코딩이며 실제 물리적 최초 진입은 영상 시작 전 미상이다.', 'resolution': '같은 두 번째 대기 차량에 합의. 최초 f0 같은 차로 상태를 적용. 후반 자차 중앙선 횡단 뒤 상황으로 최초 진입을 대체하지 않음.', 'encoding_note': '0/0 is competition submission encoding only. Physical entry occurred before the clip at an unknown time.'}} |
| CCD_000540 | 44 | 44 | OUTSIDE | {'status': 'during_clip', 'lower_frame': 32, 'upper_frame': 37, 'evaluation_eligible': True, 'evidence': {'A': 'f0는 오른쪽 흰 점선 바깥 차로. f32 선행 왼쪽 바퀴가 경계 바깥이며 f34–36 경계/그 연장선을 걸쳐 자차 차로로 이동한다. 원거리 픽셀과 점선 공백 때문에 단일 프레임은 확정하지 않음.', 'B': 'f0 오른쪽 인접 차로. f34 좌측 바퀴가 점선 경계 바깥에 남아 있고 f37에는 경계 연장선 안쪽으로 들어와 있어 첫 바퀴 접촉을 f34~37로 지지한다.', 'resolution': 'A f32~36과 B f34~37은 같은 좌측 진입 사건의 중첩 구간. 보수적 합집합 f32~37. 원거리 바퀴와 점선 공백의 한계.'}} |
| CCD_000643 | 15 | 15 | UNCERTAIN | {'status': 'unknown', 'lower_frame': None, 'upper_frame': None, 'evaluation_eligible': False, 'evidence': {'A': 'f0 탑차는 전방 같은 진행 방향이지만 폭넓은 노면이 눈으로 덮여 차로 경계와 그 연장선을 확인할 수 없다. f20·28 횡방향 회전은 보이나 첫 바퀴 경계 접촉은 확정 불가.', 'B': 'f0 정면 진행 통로에 탑차가 있으나 눈 덮인 넓은 노면에서 차로 수와 양쪽 경계를 확인할 수 없다. f31 횡회전에서도 최초 경계 접촉을 확인 불가.', 'resolution': '같은 탑차에 합의하나 눈 덮인 넓은 도로의 차로 경계가 없어 상태/진입 미상. 표식 가능성과 차로 상태 확정은 별개.'}} |
| CCD_000022 | 12 | 12 | OUTSIDE | {'status': 'unknown', 'lower_frame': None, 'upper_frame': None, 'evaluation_eligible': False, 'evidence': {'A': 'f0·8·34 넓은 무차선 공간에서 같은 흰 세단은 분명하지만 자차 차로 경계나 신뢰 가능한 연장선이 보이지 않는다. 바퀴가 보인다는 사실만으로 첫 경계 접촉을 정할 수 없음.', 'B': 'f0 및 f24 상대는 보이지만 넓고 무표시인 교차로에서 자차 차로의 우측 경계와 연장선을 고정할 수 없어 첫 바퀴 기준 상태/진입 미상.', 'resolution': '같은 흰 세단에 합의하나 무표시 교차로의 경계가 없어 상태/진입 미상. 우측 화면 위치를 OUTSIDE로 대체하지 않음.'}} |
| CCD_000493 | 12 | 12 | None | {'status': 'unknown', 'lower_frame': None, 'upper_frame': None, 'evaluation_eligible': False, 'evidence': {'A': 'f0 보이는 헤드라이트는 먼저 지나가는 다른 차량이다. 실제 상대는 당시 식별 불가. f32·37·39·41 상대가 자차 진행 궤적에 가까워지나 적설로 중앙 차선 경계가 보이지 않아 첫 바퀴 경계 접촉은 정할 수 없다.', 'B': 'f35/38/40 검은 차량의 자차 쪽 접근은 보이나 눈과 노면 자국을 실제 차로 경계로 확정할 수 없어 최초 바퀴 경계 접촉 구간 미상.', 'resolution': '같은 세 번째 접근 검은 차량에 합의. f0 먼저 보이는 두 차량은 다른 차량. 적설로 차로 경계 불명.'}} |

All policy branches, unchanged contact/side/space outputs, fixed eligible denominators, before/during strata, interval grades and known-OUTSIDE audits were independently recomputed.
Timing grades use rational seconds; paired accuracy uses closed-interval set differences and paired MAE extrema use shared-target endpoints.

## Limits

- Source-group disjointness and a bounded overlap screen do not certify universal incident independence.
- AI adjudicated interval/state references and privileged counterpart marks are not certified human or official GT.
- Timing-unknown cases remain outside timing scores but within marker/state and known-OUTSIDE audits.
- Baseline uses four shared real calls per case; treatment is a deterministic entry-only override.
- CPU replay validates recorded execution and exact processor hashes, not visual ground truth or model reasoning.
- Mac only; no CUDA equivalence, automatic target detection, production deployment or official S2 claim.

Frozen files and parent results were not changed. Full per-case scores, processor metadata, resources and hashes are in independent_verification.json.
