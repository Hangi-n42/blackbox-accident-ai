# Stage2 CCD540 차량·바퀴·차선 좌표 진단

2026-09-20 KST. **차량 상자의 좌표 단위 불일치와, 단위를 환산해도 남는 바퀴·차선 지목 실패를 확인했다.** f31은 바퀴·차선을 지목하지 않고 UNCERTAIN, f39는 부정확한 바퀴점과 차량 상자의 대각선을 차선으로 반환하면서 INSIDE라고 답했다. f39의 상태 응답이 이전 OUTSIDE에서 INSIDE로 바뀌었지만 올바른 공간 판단이나 점수 개선으로 인정할 근거는 없다.

## 실험 조건과 참조

이전 실험의 `f31_source_detail.png`, `f39_source_detail.png`를 경로·파일 바이트까지 그대로 사용했다. 1152×768 전체 이미지, 기존 노란 상대 표식, Qwen3-VL-4B-Instruct MLX 4bit 모델, Mac native/sync/deepstack 설정을 유지했다. 새 질문 하나에서 차량 상자, 보이는 바퀴 접지점, 상대쪽 차선 중심선, 차로 안쪽 기준점과 최종 상태를 JSON으로 요구했다. 각 프레임은 새 worker에서 한 번만 호출했다.

새 구조화 출력 때문에 질문과 출력 상한을 40→768토큰으로 바꿨다. 따라서 기존 상태 질문의 내부 중간 판단을 직접 관측하는 검사는 아니다. 실제 생성은 58·187토큰이며 모두 정상 EOS로 끝났다. 길이 제한 때문에 좌표가 잘린 결과는 아니다.

전문가 AI 두 명이 서로의 참조와 새 예측을 보지 않고, 정확한 모델 입력과 검수용 이미지에서 좌표·수동 오차 범위를 먼저 동결했다. 사람 확정 정답은 아니며 두 명 모두 이 사고의 과거 상태와 실험 맥락을 알고 있었다.

- f31: 두 전문가가 차량과 가까운 뒷바퀴를 지목했다. 앞바퀴는 A만 지목했고 B는 불명으로 남겼다. 보이는 흰 점선과 제한된 연장을 별도로 기록했다.
- f39: 차량과 가까운 바퀴 두 점은 지목했지만, 두 전문가 모두 차체·그림자·점선 공백 때문에 관련 차선의 신뢰할 좌표를 만들 수 없었다.
- **기존 f39 INSIDE 참조는 이번 좌표 검수로 독립 재확인되지 않았다.** 기존 파일을 변경하거나 그 참조가 틀렸다고 확정하지는 않았다. 그러나 이후 이 단일 프레임의 상태를 확정 정답처럼 활용할 근거는 제한된다.

## 사전에 정한 픽셀 좌표 기준 결과

질문에는 헤더를 포함한 전체 이미지의 절대 픽셀 좌표와 ‘정규화 좌표를 사용하지 말 것’을 명시했다. 엄격한 JSON 형식은 두 응답 모두 유효했지만, 그 좌표를 그대로 픽셀로 해석하면 차량 상자는 두 전문가 참조와 모두 IoU 0이었다.

| 항목 | f31 | f39 |
|---|---|---|
| 원응답 차량 상자 | `[565,582,645,665]` | `[544,562,725,700]` |
| 바퀴점 | 빈 목록 | `[608,658]`, `[648,658]`, 둘 다 rear |
| 차선 | 빈 목록 | 차량 상자의 좌상·우하 좌표와 정확히 같은 두 점 |
| 상태 | UNCERTAIN | INSIDE |
| 픽셀 기준 바퀴 검수 | 미지목 | 참조 오차 약 136~142px, 허용 범위 밖 |

f31의 빈 목록은 좌표 미지목으로 기록했다. f39의 차선은 참조가 미상이므로 정답 선과의 거리 점수를 만들지 않았다. 차로 안쪽 기준점의 방향 비교는 f31의 응답이 null이고 f39에는 신뢰할 경계 참조가 없어 미채점으로 남겼다. 좌표 참조가 부족한 항목을 정답·오답으로 강제하지 않았다.

## 원인 분석 1: 좌표 단위가 맞지 않았다

실행 후 상자를 확인한 결과, 두 응답을 0~1000 정규화 좌표로 해석하면 같은 상대 SUV에 맞았다. 공식 Qwen3-VL 2D grounding 예제는 기본 좌표계를 0~1000으로 설명하고 `x/1000×width`, `y/1000×height` 변환을 사용한다. 이번 입력에서는 각각 1.152와 0.768을 곱한다. [공식 Qwen3-VL 예제](https://github.com/QwenLM/Qwen3-VL/blob/main/cookbooks/2d_grounding.ipynb)

| 프레임 | 공식 방식으로 환산한 상자 | 전문가 A/B와 IoU |
|---|---|---|
| f31 | `[650.88,446.976,743.04,510.72]` | 0.872 / 0.899 |
| f39 | `[626.688,431.616,835.2,537.6]` | 0.950 / 0.952 |

이는 **이 상자 출력의 큰 위치 오차가 단위 불일치로 설명된다는 근거**다. 다른 차량을 인식했다는 결론은 지지하지 않는다. 모델 내부 표현을 읽은 결과는 아니다.

이번 설계에서도 개선할 점이 확인됐다. 기본 좌표 규약을 먼저 확인해 질문·파서와 맞췄어야 했는데, 픽셀 좌표 지시만으로 출력 규약을 고정할 수 있다고 가정했다. 실제 응답은 그 지시를 따르지 않았다. 다만 수신 좌표를 절대 픽셀로 채점한 사전 평가 자체는 그대로 보존했다.

이 환산은 **응답을 본 뒤 수행한 별도 원인 진단**이다. 임의 오프셋이나 최적 배율을 맞추지 않고 공식 변환식 하나만 적용했다. 추가 모델 호출·원응답 수정·원평가 교체는 없으며 새로운 사전 계획 실험의 성공으로 계산하지 않았다.

## 원인 분석 2: 단위를 고쳐 읽어도 공간 증거는 맞지 않았다

f39의 환산 바퀴점은 `[700.416,505.344]`, `[746.496,505.344]`이다. 전문가 참조는 앞바퀴 약 `[640~641,520~526]`, 뒷바퀴 약 `[720~721,536~537]`이다. 일대일 대응 오차는 A 기준 **62.90·40.65px**, B 기준 **62.17·40.52px**로, 사전에 정한 6~7px 범위를 벗어났다. 한 점씩 가장 가까운 참조를 중복해서 사용하는 방식으로 성적을 높이지 않았다.

차선 출력 두 점은 차량 상자의 좌상·우하 좌표를 정확히 반복한다. 환산 후 선은 차량 차체를 가로지르는 대각선이다. 실제 도색 차선을 지목한 근거가 아니며, 두 점에 붙인 `visible`도 이미지와 맞지 않는다. f39의 정확한 참조 경계가 없다는 사실과, 출력 선이 도색 대신 차량 대각선이라는 관측은 구분된다.

두 바퀴 관계를 INSIDE라고 쓰고 최종 상태도 INSIDE라고 써서 범주끼리는 일관된다. 반환한 선의 바퀴 높이에서 x는 약 771.74이고, 반환한 두 바퀴 x 700.416·746.496과 안쪽 기준점 x 626.688은 모두 그 왼쪽이다. **자기가 제시한 잘못된 선과는 맞는 판정을 만들 수 있지만, 그것이 실제 영상의 차로 관계를 검증하지는 않는다.** 이 계산 역시 출력들 사이의 관계이며 모델 내부 추론 순서의 증거는 아니다.

f31은 참조로 지목 가능한 뒷바퀴와 관찰 점선이 있는데도 둘 다 비웠다. 따라서 이번 과제에서는 미지목도 관측됐다. 더 작은 차량 영상, 그림자·압축은 참조 자체의 제약이지만, 그것이 모델의 빈 출력을 직접 유발했는지는 확인하지 못했다.

아래 그림은 사후 공식 단위 환산 후의 출력이다. 빨강은 모델 차량 상자, 청록 점은 모델 바퀴점, 자홍 선은 모델 차선이다. 주황·흰 원은 두 전문가의 바퀴 참조와 오차 범위다. 입력으로 재사용하지 않은 검수용 그림이다.

![f39 공식 좌표 환산 후 지목과 전문가 바퀴 참조](/Users/hyeongi/projects/blackbox-accident-ai/artifacts/stage2_spatial_grounding_540_20260920/f39_normalized_hypothesis_overlay.png)

## 어디까지 결론낼 수 있는가

현재 출력에서 **단위 지시 불이행, 바퀴 지목 오류·미지목, 차선 필드에 차량 대각선을 반환한 의미 오류**를 구분해 확인했다. 특정 차량을 전혀 찾지 못한다는 해석은 환산 상자 결과와 맞지 않는다. ‘상태 문구가 INSIDE로 바뀌었다’는 것과 ‘바퀴·차선 증거로 올바르게 판정했다’는 것은 다르다.

왜 모델이 접지점 대신 그 점을 내고 상자 대각선을 차선 필드에 넣었는지에 대한 내부 원인은 확인 불가다. 시각 인식 능력, 다중 출력 질문의 부담, 규칙 적용, 양자화 중 하나로 단정할 실험을 하지 않았다. 현재 확인된 것은 출력과 관측 영상의 불일치다. 원래 Q3의 f44 선택 원인이나 공식 S2 개선을 입증하지 않는다.

다음에는 먼저 **질문·파서를 공식 0~1000 좌표계에 맞춘 동일 입력의 대조**로 좌표 계약을 정리해야 한다. 이것은 아직 실행하지 않았다. f39 차선 관계를 채점하려면 별도로 신뢰할 경계 참조가 필요하다. 이 참조 없이 상태 정확도 개선 실험을 반복하지 않는 것이 타당하다.

## 실행과 검증 기록

Mac 실제 호출 2회, 모든 worker exit 0, 네트워크 시도 0회다. 입력은 각각 이미지 토큰 864개·전체 1,245개이며 이전 실험과 이미지 픽셀 텐서·격자가 정확히 같다. 독립 검증자가 CPU에서 두 입력을 재생하고 엄격한 JSON 파싱, 원좌표 점수 산술을 확인했다. 동결 1,896개 파일과 이전 1,866개 보존을 확인했다. 부모 worker 경과 시간 합은 12.656초, 최대 MLX 메모리는 5.511GB다. 속도 비교 실험은 아니다.

- [사전 프로토콜](/Users/hyeongi/projects/blackbox-accident-ai/artifacts/stage2_spatial_grounding_540_20260920/protocol.json) · [실행 코드](/Users/hyeongi/projects/blackbox-accident-ai/artifacts/stage2_spatial_grounding_540_20260920/experiment.py) · [동결 기록](/Users/hyeongi/projects/blackbox-accident-ai/artifacts/stage2_spatial_grounding_540_20260920/freeze.json)
- [독립 좌표 참조 A](/Users/hyeongi/projects/blackbox-accident-ai/artifacts/stage2_spatial_grounding_540_20260920/review_a.md) · [독립 좌표 참조 B](/Users/hyeongi/projects/blackbox-accident-ai/artifacts/stage2_spatial_grounding_540_20260920/review_b.md)
- [사전 실행 검증](/Users/hyeongi/projects/blackbox-accident-ai/artifacts/stage2_spatial_grounding_540_20260920/preflight_review.md) · [원평가](/Users/hyeongi/projects/blackbox-accident-ai/artifacts/stage2_spatial_grounding_540_20260920/evaluation.json) · [실행 후 독립 검증](/Users/hyeongi/projects/blackbox-accident-ai/artifacts/stage2_spatial_grounding_540_20260920/independent_verification.md) · [원평가 산술 검증](/Users/hyeongi/projects/blackbox-accident-ai/artifacts/stage2_spatial_grounding_540_20260920/raw_score_review.json)
- [사후 좌표 단위 진단](/Users/hyeongi/projects/blackbox-accident-ai/artifacts/stage2_spatial_grounding_540_20260920/coordinate_unit_audit.json) · [재현 코드](/Users/hyeongi/projects/blackbox-accident-ai/artifacts/stage2_spatial_grounding_540_20260920/coordinate_units.py) · [공식 예제 발췌 기록](/Users/hyeongi/projects/blackbox-accident-ai/artifacts/stage2_spatial_grounding_540_20260920/coordinate_source_excerpt.json)

- [사후 단위 변환 산술 검증](/Users/hyeongi/projects/blackbox-accident-ai/artifacts/stage2_spatial_grounding_540_20260920/coordinate_unit_review.json) · [전문가 사후 시각·해석 검토](/Users/hyeongi/projects/blackbox-accident-ai/artifacts/stage2_spatial_grounding_540_20260920/post_result_review_a.md)
