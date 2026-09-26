# Stage2 첫 프레임 상태: 입력 해상도 대조 결과

2026-09-20, Apple Silicon Mac 실행. **고정한 개발 참조 5건에서 일치율이 2/5에서 4/5로 개선됐다. 단순 확대와 원본 세부를 보존한 큰 입력의 결과는 같았다.** 688·196의 오답이 정답으로 바뀌고 기존 정답 2건은 유지되어 사전 성공 조건을 통과했다. 실제 모델 호출 15회 및 전문가 에이전트의 시각 검수·독립 검증을 완료했다.

## 목적과 통제

앞선 실험에서는 사고 상대를 노란색으로 표시하고 명시적으로 지시했음에도, 첫 프레임의 차로 안·밖 질문에 5건 모두 OUTSIDE라고 답했다. 이번에는 질문을 그대로 유지하고 입력 크기와 원본 세부의 영향을 비교했다.

같은 첫 프레임·사고 상대·표식의 상대 위치와 두께·장면 구성·질문·모델·40토큰 출력 한도·엄격한 JSON 파서·기존 참조를 고정했다. 참조 정답과 native PTS 메타데이터는 모델 입력에 별도로 추가하지 않았다. 원본 화면에 포함된 시간 자막은 유지했다. 다음 세 조건을 각 5회, 새 Mac worker에서 한 번씩 실행했다.

| 조건 | 입력 | 실제 이미지 토큰 | 전체 입력 토큰 | 참조 일치 |
|---|---|---:|---:|---:|
| 현재 축소 입력 | 기존 표시 이미지 384×256 그대로 | 96 | 177 | 2/5 (40%) |
| 단순 확대 | 축소 입력을 최근접 보간으로 3배 확대, 1152×768 | 864 | 945 | 4/5 (80%) |
| 원본 세부 사용 | 원본 장면에서 재구성, 1152×768 | 864 | 945 | 4/5 (80%) |

고해상도군은 1280×720 원본 장면을 1152×648로 bicubic 축소해 같은 캔버스에 배치했다. 원본 폭의 90%이며, 원본을 전혀 줄이지 않은 입력은 아니다. 헤더·여백·표식 픽셀은 단순 확대군과 정확히 같다. 두 큰 입력의 차이는 표식을 제외한 장면 픽셀에만 있다. 단순 확대에는 새로운 원본 세부가 들어가지 않는다.

질문은 다음 문장을 포함한 기존 상태 질문과 완전히 같다.

> The vehicle marked in yellow is the collision counterpart. In this first frame, is this vehicle already inside the camera car's driving lane? At an intersection, extend the camera car's lane boundaries forward. Return JSON with lane_state: INSIDE, OUTSIDE, or UNCERTAIN. Use UNCERTAIN if the vehicle or lane relation cannot be determined.

## 사례별 결과

| 사례 | 고정 AI 참조 | 축소 | 단순 확대 | 원본 세부 사용 |
|---|---|---|---|---|
| CCD_000688 | INSIDE | OUTSIDE | INSIDE | INSIDE |
| CCD_000801 | INSIDE | OUTSIDE | OUTSIDE | OUTSIDE |
| CCD_000453 | OUTSIDE | OUTSIDE | OUTSIDE | OUTSIDE |
| CCD_000196 | INSIDE | OUTSIDE | INSIDE | INSIDE |
| CCD_000728 | OUTSIDE | OUTSIDE | OUTSIDE | OUTSIDE |

INSIDE 3건의 재현율은 0/3→2/3, OUTSIDE 2건은 2/2로 유지됐다. 무조건 INSIDE를 선택하는 기준은 3/5, 무조건 OUTSIDE는 2/5이므로 이번 개선은 한 가지 답만 고르는 변화가 아니다. UNCERTAIN·잘못된 JSON은 모든 조건에서 0건이었다.

사전 조건인 최소 1건 개선·손실 0건·INSIDE 개선 포함·축소군의 과거 응답 재현을 충족했다. 새 축소군은 과거 5건의 원문과 판정 모두 일치했다. 단순 확대 대비 원본 세부 사용의 추가 정답 이득은 0건이다.

## 무엇을 설명할 수 있는가

**관측된 두 건의 개선에 원본 세부 추가는 필수적이지 않았다.** 원본 세부가 없는 단순 확대에서도 같은 개선이 발생했다. 따라서 “세부 정보 손실만 복구하면 해결된다”는 설명은 이 결과로 지지되지 않는다. 반대로 원본 세부가 다른 사례에서도 무효라는 뜻은 아니다.

실제 전처리 격자는 축소군 `[1,16,24]`에서 두 확대군 `[1,48,72]`로 바뀌었고 이미지 토큰은 9배가 됐다. 두 확대군의 텍스트 토큰·attention mask·이미지 격자는 같고 이미지 텐서만 달랐다. 입력 확대와 함께 차량·표식·차선의 절대 크기, 이미지 토큰 수, 보간 형태가 달라졌다. **어느 내부 요인이 응답을 바꿨는지는 분리하지 못했다.** 같은 크기의 두 큰 입력도 세부·흐림·계단 현상이 함께 달라지므로 특정 시각 기제의 단독 인과를 확정할 수 없다.

801은 큰 입력에서도 OUTSIDE 오답을 유지했다. 시각 검수에서 교차로 접근부의 차로 경계, 차량 아래 가려진 점선, 도로 가장자리로 판단해야 하는 공간 관계, 원본의 얼룩·흐림이 확인됐다. 이는 관찰된 난점이며 오답의 확정 원인은 아니다. 기존 INSIDE 참조는 변경하지 않았다. 688은 도색 차선이 없는 진행 통로를 기준으로 한 참조라는 한계도 유지한다.

## 검증과 비용

입력·코드·참조 등 동결 969개 파일과 이전 동결 909개 파일의 보존, 원본 첫 프레임과 표식의 출처, 세 조건의 픽셀 재구성, 15개 worker의 종료·실제 호출·원문·엄격 파서·평가를 독립 검증했다. CPU에서 15개 입력의 전처리를 다시 실행해 실제 호출의 4종 텐서 해시와 대화 프롬프트 해시가 모두 일치함을 확인했다. 이 검증에는 추가 모델 추론이 없었다.

15개 worker 실행 시간 합계는 52.478초였다. 조건별 5건 합계는 축소 13.229초, 단순 확대 19.444초, 원본 세부 사용 19.805초였다. MLX 최대 메모리는 축소 3.541GB, 두 확대군 5.511GB였다. 시간은 새 worker 시작·모델 로드·질의·결과 저장을 포함하며 반복 성능 벤치마크는 아니다.

## 적용 범위와 다음 제안

이 결과는 **기존에 노출된 개발용 AI 참조 5건과 전문가 정보를 담은 상대 표식**을 사용한 첫 프레임 상태 진단이다. 독립 정답 집합의 일반화 성능, 사람 정답의 정확성, 실제 진입 시점 또는 공식 S2 개선을 입증하지 않는다. 기존 12개 후보에서 진입 시점을 다시 선택하지 않았고 제출 코드를 변경하지 않았다.

다음 실험은 확대 입력의 첫 프레임 상태 판정을 실제 진입 선택에 연결했을 때 효과가 있는지 확인하는 것이 적절하다. 비교 정책은 하나로 고정한다: 첫 프레임이 INSIDE이면 시작 프레임을 선택하고, OUTSIDE/UNCERTAIN이면 기존 진입 선택을 유지한다. 사고 단위로 분리한 새 검수 자료에서 시작 전 진입과 영상 중 진입을 함께 평가해 잘못된 시작 프레임 선택을 반드시 집계해야 한다. 상태 INSIDE와 대회의 진입 시점 정의가 일치하는지 검수하는 절차가 선행되어야 한다. 현재 전문가 표식을 쓰는 한 진단 실험이며, 제출 개선을 주장하려면 자동 상대 지정까지 포함한 검증이 별도로 필요하다. 이번 작업에서는 이 추가 실험을 실행하지 않았다.

## 근거 파일

- [사전 프로토콜](/Users/hyeongi/projects/blackbox-accident-ai/artifacts/stage2_state_resolution_20260920/protocol.json), [동결 manifest](/Users/hyeongi/projects/blackbox-accident-ai/artifacts/stage2_state_resolution_20260920/freeze.json), [입력 검사](/Users/hyeongi/projects/blackbox-accident-ai/artifacts/stage2_state_resolution_20260920/input_checks.json)
- [평가 원문](/Users/hyeongi/projects/blackbox-accident-ai/artifacts/stage2_state_resolution_20260920/evaluation.json), [실행 기록](/Users/hyeongi/projects/blackbox-accident-ai/artifacts/stage2_state_resolution_20260920/run/report.json)
- [독립 검증](/Users/hyeongi/projects/blackbox-accident-ai/artifacts/stage2_state_resolution_20260920/independent_verification.md), [검증 세부 JSON](/Users/hyeongi/projects/blackbox-accident-ai/artifacts/stage2_state_resolution_20260920/independent_verification.json)
- [시각 검수 A](/Users/hyeongi/projects/blackbox-accident-ai/artifacts/stage2_state_resolution_20260920/visual_review_a.md), [시각 검수 B](/Users/hyeongi/projects/blackbox-accident-ai/artifacts/stage2_state_resolution_20260920/visual_review_b.md), [사후 시각 분석](/Users/hyeongi/projects/blackbox-accident-ai/artifacts/stage2_state_resolution_20260920/post_resolution_analysis_a.md)
- 196 실제 입력: [축소](/Users/hyeongi/projects/blackbox-accident-ai/artifacts/stage2_state_resolution_20260920/inputs/CCD_000196_low.png), [단순 확대](/Users/hyeongi/projects/blackbox-accident-ai/artifacts/stage2_state_resolution_20260920/inputs/CCD_000196_upsampled.png), [원본 세부 사용](/Users/hyeongi/projects/blackbox-accident-ai/artifacts/stage2_state_resolution_20260920/inputs/CCD_000196_high.png)
