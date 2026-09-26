# Stage2 CCD540 바퀴·차선 상태 진단 결과

2026-09-20 KST. **단순 확대는 이번 상태 판별을 개선하지 못했다.** CCD_000540의 f31·35·39를 개별 질문했으나 축소·확대 입력 모두 OUTSIDE라고 답했다. 전문가 AI 참조에서 최소 한 바퀴가 차로 안쪽에 있는 f39도 놓쳤다. 판정 가능한 두 프레임의 일치율은 두 조건 모두 1/2이며 확대 이득은 0건이다.

Mac 실제 질의 6회, 두 전문가의 독립 시각 참조, 입력 검수, 실행 후 독립 검증을 완료했다. 한 번 노출된 사고의 원인 진단이며 진입 출력이나 제출 코드를 변경하지 않았다.

## 목적과 고정 조건

이전 실험에서 540의 첫 프레임은 OUTSIDE로 맞게 판단했지만, 기존 진입 선택은 f44 오답을 유지했다. 기존 후보에는 허용 오차 안의 f35도 있었다. 이번에는 같은 상대의 바퀴와 자차 차선 경계 관계를 **개별 프레임에서 판별할 수 있는지** 확인했다.

첫 바퀴가 경계에 닿거나 경계 안쪽이면 INSIDE, 모든 바퀴가 접촉 없이 바깥이면 OUTSIDE, 관계를 읽을 수 없으면 UNCERTAIN이라는 질문을 사용했다. 차량 전체가 들어와야 한다는 기준이 아니다. 첫 프레임용 질문에서 현재 프레임의 바퀴 상태 질문으로 바뀌었으므로, 이전 실험과 정확도를 직접 비교하지 않는다. 이번 두 조건 안에서는 문구가 완전히 같다.

| 고정 항목 | 내용 |
|---|---|
| 사고 상대 | 원형 후면 스페어타이어 커버가 있는 같은 흰 SUV |
| 프레임 | 원본 f31·35·39, native PTS 3.1·3.5·3.9초 |
| 표식 | 상부 창 안쪽의 노란 2px 사각형; 바퀴·차선 추가 가림 없음 |
| 축소 입력 | 기존 `_sheet` 구성의 384×256 단일 프레임 |
| 확대 입력 | 축소본 전체를 정확히 3배 최근접 확대, 1152×768 |
| 질의·실행 | 같은 문구·40토큰 한도·엄격한 JSON 파서·모델·Mac native/sync/deepstack |

확대에는 새로운 원본 세부·크롭·보간 복원·다른 후보가 들어가지 않았다. 각 프레임·조건마다 새 worker에서 한 번만 질의했으며 결과를 보고 추가 질문이나 변형을 실행하지 않았다.

## 전문가 참조와 결과

두 전문가가 새 진단 응답과 동료 기록을 보지 않고 원본을 각각 확인했다. 다만 이 사고의 과거 진입 구간과 실험 맥락은 알고 있었으므로 완전 미노출 참조가 아니다. 세 상태는 모두 일치했다. A의 상부 표식 좌표를 선택하고 B가 실제 렌더링의 대상·비가림·픽셀 변환을 확인했다.

| 원본 프레임 | 합의한 AI 참조 | 축소 응답 | 확대 응답 | 평가 |
|---|---|---|---|---|
| f31 | OUTSIDE | OUTSIDE | OUTSIDE | 둘 다 참조 일치 |
| f35 | UNKNOWN | OUTSIDE | OUTSIDE | 정오 판정 불가, 정확도에서 제외 |
| f39 | INSIDE | OUTSIDE | OUTSIDE | 둘 다 오답 |

f31은 가까운 바퀴와 경계 사이의 바깥쪽 간격이 보인다. f39는 보이는 앞바퀴가 자차 차로 내부에 들어온 상태다. f35는 바퀴 접지점과 점선 공백의 연장선이 흐려 접촉·미접촉을 안정적으로 구분하지 못했다. 과거 f32~37 진입 참조에 포함된다는 이유로 f35를 INSIDE 정답으로 만들지 않았다.

- 평가 분모는 조건당 **2프레임**이며 3프레임이 아니다. 두 조건 모두 1/2 일치, 개선 0건·손실 0건이다.
- 확정 OUTSIDE와 INSIDE의 구분 검사는 두 조건 모두 실패했다. 1/2 일치는 항상 OUTSIDE라고 답하는 상수 기준과 같다.
- 유효하지 않은 JSON과 UNCERTAIN 응답은 0건이다. f35의 OUTSIDE를 오답 또는 정답으로 세지 않았다.
- 응답열은 두 조건 모두 `OUTSIDE → OUTSIDE → OUTSIDE`다. 시간순 역전이 없다는 사실은 구분 능력의 증거가 아니다.

## 무엇을 확인했고, 원인은 어디까지 알 수 있는가

**다중 후보에서 시점을 고르는 질문뿐 아니라, 표시된 동일 상대의 개별 바퀴 상태를 묻는 질문에서도 오류가 관측됐다.** f39가 차로 안쪽에 있다는 참조를 이번 질문에서 회복하지 못했다. 따라서 후보 수나 최종 시간 선택 규칙만 바꾸는 접근보다 공간 관계 판별을 더 진단할 근거가 생겼다.

다만 이번 상태 질문은 기존 Q3의 내부 중간 출력을 관측한 것이 아니라 별도의 진단 질문이다. 이 실패가 원래 f44 선택의 유일한 원인이라고 단정할 수 없다. 기존 Q3와는 단일 이미지 구성·명시적 상대표식·질문이 다르다.

입력 검수는 표식이 같은 SUV를 지칭한다는 것을 확인했지만, 모델이 실제로 그 차량을 보았는지는 확인하지 못했다. 차량과 바퀴의 위치를 잘못 보았는지, 자차 차선 경계를 잘못 연장했는지, “최소 한 바퀴” 조건을 적용하지 못했는지도 JSON의 상태 하나로 구분할 수 없다. 세 프레임에서 OUTSIDE가 반복됐다는 사실을 모델 전체의 일반적 편향으로 확대하지 않는다.

실제 이미지 토큰은 축소 96개에서 확대 864개로 증가했다. 원본의 새 세부를 넣지 않은 채 입력 크기·절대 표식 크기·이미지 격자를 바꿨지만 응답은 동일했다. **이 사례에서는 단순 확대만으로 충분하지 않았다.** 원본 세부의 추가가 도움이 되는지는 이번 비교로 확인하지 못했다.

## 독립 검증과 실행

동결 1,829개 파일의 실행 전후 해시와 이전 동결 1,793개 보존을 확인했다. 6 worker 모두 exit 0이고 실제 모델 호출은 6회다. 독립 검증기가 6개 원응답의 엄격 파싱·평가를 재계산하고, CPU에서 6개 전처리 입력의 텐서·프롬프트 해시를 재생해 실제 호출과 모두 일치함을 확인했다. 검증기의 모델 로드·추가 추론은 없었다.

| 실제 전처리 | 축소 | 확대 |
|---|---:|---:|
| 이미지 격자 `[t,h,w]` | `[1,16,24]` | `[1,48,72]` |
| 이미지 토큰 | 96 | 864 |
| 전체 입력 토큰 | 217 | 985 |
| 이미지 외 토큰 | 121 | 121 |
| 3 worker 부모 측 경과 시간 합 | 19.680초 | 54.244초 |
| 최대 MLX 메모리 | 3.541GB | 5.511GB |

부모 측 worker 경과 시간 합은 73.924초, child 내부 시간 합은 66.391초다. 프로세스 시작·모델 로드·질의·저장 범위가 포함되며 자료 검수와 독립 검증 시간은 제외된다. 단일 실행의 관측이며 반복 속도 벤치마크나 다른 실험 대비 성능 비교는 아니다.

## 다음 단일 진단 제안

**같은 1152×768 캔버스·표식·질문을 유지하고, 현재 단순 확대본과 원본 세부를 보존해 재구성한 입력을 비교**하는 것이 적절하다. 헤더·여백·표식 픽셀은 고정하고 장면 픽셀만 원본에서 가져와, 입력 크기·이미지 토큰 증가와 세부 정보의 차이를 분리한다. 세부·흐림·계단 현상은 함께 달라질 수 있으므로 특정 시각 기제 하나의 단독 인과까지 주장하지 않는다.

앞선 고해상도 실험은 다른 첫 프레임들의 차로 상태 질문이었다. 이번 제안은 f31·35·39의 바퀴 상태 질문과 새로 확인한 f39 오류에 대한 대조다. f35의 미상 참조는 그대로 유지하고, 이번 결과를 근거로 f39나 f35를 자동 진입 시점으로 선택하지 않는다. 이 추가 진단은 아직 실행하지 않았다.

공식 S2, 일반화 성능, 자동 상대 지정 성능은 이번 실험의 평가 대상이 아니다. 전문가 상대표식과 개발용 AI 참조에 의존한 한 사고의 결과라는 한계가 있다.

## 기록

- [사전 프로토콜](/Users/hyeongi/projects/blackbox-accident-ai/artifacts/stage2_wheel_state_540_20260920/protocol.json) · [실험 코드](/Users/hyeongi/projects/blackbox-accident-ai/artifacts/stage2_wheel_state_540_20260920/experiment.py) · [동결 manifest](/Users/hyeongi/projects/blackbox-accident-ai/artifacts/stage2_wheel_state_540_20260920/freeze.json)
- [독립 참조 A](/Users/hyeongi/projects/blackbox-accident-ai/artifacts/stage2_wheel_state_540_20260920/review_a.md) · [독립 참조 B](/Users/hyeongi/projects/blackbox-accident-ai/artifacts/stage2_wheel_state_540_20260920/review_b.md) · [합의 참조](/Users/hyeongi/projects/blackbox-accident-ai/artifacts/stage2_wheel_state_540_20260920/references.json)
- [입력 검수](/Users/hyeongi/projects/blackbox-accident-ai/artifacts/stage2_wheel_state_540_20260920/input_peer_review.md) · [사전 실행 검증](/Users/hyeongi/projects/blackbox-accident-ai/artifacts/stage2_wheel_state_540_20260920/preflight_review.md)
- [평가 결과](/Users/hyeongi/projects/blackbox-accident-ai/artifacts/stage2_wheel_state_540_20260920/evaluation.json) · [실행 기록](/Users/hyeongi/projects/blackbox-accident-ai/artifacts/stage2_wheel_state_540_20260920/run/report.json) · [독립 검증](/Users/hyeongi/projects/blackbox-accident-ai/artifacts/stage2_wheel_state_540_20260920/independent_verification.md)
- f39 입력: [축소](/Users/hyeongi/projects/blackbox-accident-ai/artifacts/stage2_wheel_state_540_20260920/inputs/f39_low.png) · [확대](/Users/hyeongi/projects/blackbox-accident-ai/artifacts/stage2_wheel_state_540_20260920/inputs/f39_upsampled.png)

- [전문가 사후 해석](/Users/hyeongi/projects/blackbox-accident-ai/artifacts/stage2_wheel_state_540_20260920/post_result_review.md)
