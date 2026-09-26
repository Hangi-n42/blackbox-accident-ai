# Stage2 CCD540 동일 크기 원본 세부 비교

2026-09-20 KST. **1152×768 크기를 고정해도 원본 세부 보존은 이번 바퀴 상태 판별을 개선하지 못했다.** 두 조건 모두 f31·35·39에 OUTSIDE라고 답했다. 판정 가능한 AI 참조 두 프레임의 일치율은 각각 1/2이며, 개선 0건·손실 0건이다. INSIDE 참조인 f39의 오류가 유지됐다.

## 무엇을 비교했는가

직전 실험에서는 384×256 축소본을 1152×768로 단순 확대해도 f39 오류를 회복하지 못했다. 이번에는 입력 크기와 이미지 토큰 수를 고정하고, 원본에서 가져온 더 세밀한 장면 픽셀이 도움이 되는지 확인했다.

| 조건 | 입력 구성 |
|---|---|
| 단순 확대 | 이전 384×256 입력 전체를 3배 최근접 확대했던 1152×768 파일 그대로 사용 |
| 원본 세부 보존 | 원본 1280×720을 1152×648로 BICUBIC 축소하고, 같은 1152×768 캔버스의 장면 영역에 삽입 |

장면 범위, 상대 SUV, 원본 프레임, 노란 표식의 정확한 픽셀, 헤더·여백, 바퀴 상태 질문, 모델, 40토큰 출력 한도, 엄격한 JSON 파서, Mac 실행 설정을 고정했다. 원본 세부 보존본은 원본 너비의 90%이므로 원본 해상도 그대로는 아니다. 세부·흐림·계단 현상은 함께 바뀐다.

질문은 표시된 차량의 **최소 한 바퀴가 자차 차로 경계에 닿거나 안쪽에 있는가**였다. 모든 바퀴가 접촉 없이 바깥이면 OUTSIDE, 접촉하거나 안쪽이면 INSIDE, 판별 불가이면 UNCERTAIN이다. 점선 공백에서는 보이는 차선 경계를 연장하도록 했다.

AI 전문가가 실행 전에 실제 입력 6장과 원본 3장을 확인했다. 같은 상대·장면 범위를 유지했고 표식이 바퀴나 차선을 추가로 가리지 않았다. 기존 전문가 참조를 파일 바이트까지 동일하게 유지했으며, 이번 응답을 보고 재라벨링하지 않았다. 각 프레임·조건은 새 worker에서 한 번씩 질의했다.

## 결과

| 원본 프레임 | 고정 AI 참조 | 단순 확대 | 원본 세부 보존 | 채점 |
|---|---|---|---|---|
| f31 / 3.1초 | OUTSIDE | OUTSIDE | OUTSIDE | 둘 다 참조 일치 |
| f35 / 3.5초 | UNKNOWN | OUTSIDE | OUTSIDE | 제외 |
| f39 / 3.9초 | INSIDE | OUTSIDE | OUTSIDE | 둘 다 오답 |

확정 참조 2개에서 각각 **1/2**, 차이 **0%p**다. 두 조건 모두 UNCERTAIN과 잘못된 JSON은 0건이다. f35는 바퀴·경계 관계가 불명확한 참조이므로 정답이나 오답으로 세지 않았다. 과거 진입 구간 안의 후보라는 사실과 그 단일 프레임의 상태 참조를 혼동하지 않았다.

단순 확대 대조군 3개의 원응답·판정과 전처리 텐서는 직전 실험과 정확히 일치했다. 따라서 이번 조건에서 기존 오류를 재현한 뒤 원본 세부 추가 효과를 비교했다. 사전에 정한 ‘확정 참조 개선 1건 이상·손실 0건’ 기준은 개선이 없어 통과하지 못했다.

## 해석과 한계

**이 사례와 질문에서는 원본 세부를 더 보존하는 것만으로 f39 오류를 해소하지 못했다.** 큰 캔버스에 새 세부 없이 확대했던 직전 결과에 이어, 같은 크기의 원본 세부 보존본에서도 오류가 관측됐다. 현재 근거는 해상도나 세부 보존만 반복해서 바꾸는 다음 실험을 지지하지 않는다.

그러나 정확한 실패 원인은 **확인 불가**다. 모델이 표시된 SUV를 실제로 대상으로 삼았는지, 바퀴 접지점을 찾았는지, 자차 차선 경계를 올바르게 연장했는지, ‘최소 한 바퀴’ 규칙을 적용했는지는 상태 JSON 하나로 분리할 수 없다. 원본 자체의 압축·가림도 남는다. 세부가 모든 사고에서 무용하다는 결론이나, 원래 다중 후보 Q3의 f44 선택 원인을 규명했다는 결론은 성립하지 않는다.

한 번 노출된 사고 1건, 확정 AI 참조 2프레임, 전문가가 지정한 상대 표식에 의존한다. 사람 확정 정답이나 독립 검증 집합의 성능이 아니며 공식 S2도 아니다. 진입 시점과 다른 세 출력, 제출 코드는 변경하지 않았다.

## 실행 후 독립 검증

Mac 실제 질의 6회와 모든 worker의 exit 0을 확인했다. 독립 검증기는 모델을 다시 로드하거나 질의하지 않고 CPU에서 6개 입력 전처리를 재생해 실제 호출의 텐서 해시와 모두 일치함을 확인했다. 동결 파일 1,866개의 실행 전후 해시가 일치하며, 그 안의 이전 동결 1,829개도 보존됐다. 원응답의 엄격한 파싱과 점수 계산도 독립적으로 재계산해 통과했다.

두 조건 모두 실제 이미지 격자 `[1,48,72]`, 이미지 토큰 **864개**, 전체 입력 토큰 **985개**다. 각 쌍에서 텍스트·attention mask·이미지 격자와 프롬프트가 같고 **pixel_values만 다르다**. 따라서 이번 무개선 결과를 입력 크기나 토큰 노출량 차이로 설명할 근거는 없다.

6개 부모 측 worker 경과 시간 합은 97.637초, 양군 최대 MLX 메모리는 각각 5.511GB다. 시간은 프로세스 시작·모델 로드·질의·저장을 포함하고 자료 검수와 독립 검증은 제외한다. 조건별 합은 확대 56.524초, 세부 보존 41.113초지만 단일 실행이므로 속도 개선으로 해석하지 않는다.

## 다음 단일 진단 제안 — 미실행

같은 세부 보존 입력 f31·f39에서 **표식 차량의 상자, 보이는 바퀴 접지점, 자차 차선 경계와 안쪽 방향을 좌표로 지목하는 질문**을 한 번씩 하는 것이 다음 제안이다. 새로운 좌표 참조·허용오차를 응답 전에 고정하고, 숨겨진 지점은 미상으로 남긴다. 응답을 이미지 위에 표시해 대상·바퀴·차선의 불일치를 각각 검수하면 현재 범주 응답보다 실패 구성요소를 좁힐 수 있다.

이는 새로운 출력 과제이므로 좌표 생성 실패와 기존 상태 판별 실패를 완전히 분리하지 못한다. 모든 좌표가 맞아도 원래 Q3 오류의 원인 규명은 아니다. 추가 해상도 변형이나 답을 본 뒤 재질문 없이 최대 2회로 제한하는 제안이며 이번에는 실행하지 않았다. 전문가 A의 사후 해석과 보고서 사실 대조도 완료했다.

## 기록

- [사전 프로토콜](/Users/hyeongi/projects/blackbox-accident-ai/artifacts/stage2_wheel_detail_540_20260920/protocol.json) · [실험 코드](/Users/hyeongi/projects/blackbox-accident-ai/artifacts/stage2_wheel_detail_540_20260920/experiment.py) · [동결 manifest](/Users/hyeongi/projects/blackbox-accident-ai/artifacts/stage2_wheel_detail_540_20260920/freeze.json)
- [입력 시각 검수](/Users/hyeongi/projects/blackbox-accident-ai/artifacts/stage2_wheel_detail_540_20260920/input_peer_review.md) · [사전 실행 검증](/Users/hyeongi/projects/blackbox-accident-ai/artifacts/stage2_wheel_detail_540_20260920/preflight_review.md)
- [고정 AI 참조](/Users/hyeongi/projects/blackbox-accident-ai/artifacts/stage2_wheel_detail_540_20260920/references.json) · [평가 결과](/Users/hyeongi/projects/blackbox-accident-ai/artifacts/stage2_wheel_detail_540_20260920/evaluation.json) · [실행 기록](/Users/hyeongi/projects/blackbox-accident-ai/artifacts/stage2_wheel_detail_540_20260920/run/report.json)
- f39 실제 입력: [단순 확대](/Users/hyeongi/projects/blackbox-accident-ai/artifacts/stage2_wheel_detail_540_20260920/inputs/f39_upsampled.png) · [원본 세부 보존](/Users/hyeongi/projects/blackbox-accident-ai/artifacts/stage2_wheel_detail_540_20260920/inputs/f39_source_detail.png)

- [실행 후 독립 검증](/Users/hyeongi/projects/blackbox-accident-ai/artifacts/stage2_wheel_detail_540_20260920/independent_verification.md)

- [전문가 사후 해석과 보고서 대조](/Users/hyeongi/projects/blackbox-accident-ai/artifacts/stage2_wheel_detail_540_20260920/post_result_review.md)
