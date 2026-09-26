# Stage2 전문가 검수와 Mac 실측 결과

> 2026-09-20 후속 검증 완료: 새 CCD 12건·전문가 검수·Mac 48회 실행 결과, 기존 단일 후보를 미채택으로 종료했다. 합의한 준비·진단·검증 작업은 완료했고 점수 상승은 미입증이다. [최신 결과](/Users/hyeongi/projects/blackbox-accident-ai/docs/stage2-goal-results-20260920.md). 아래는 당시 기록이다.

2026-09-19. **전문가 에이전트 검수와 Mac 실행을 완료했다. 신규 6건에서는 기준선과 후보의 네 출력이 모두 같았고, 정확한 접촉 정답은 확보되지 않았다. 따라서 추가 점수 개선을 입증하지 못했다.** 기존 단일 후보를 연구 상태로 보존한다.

사용자의 변경 지시에 따라 사람 검수나 Linux/CUDA 제공을 기다리지 않고 수행했다. AI 검수를 사람·공식 정답으로 표기하지 않았고, Mac 결과의 범위를 CUDA 실측으로 확대하지 않았다. 기존 V6/V7 제출 코드·가중치는 변경하지 않았다.

## 1. 전문가 검수와 평가 묶음

문맥을 분리한 시각 검수자 A/B는 이전 모델 예측·기존 screening·서로의 판단을 보지 않고 00024~00029를 검수했다. 전체 시간축 2Hz 표본과 세부 원본을 포함하여 A는 고유 772프레임, B는 747프레임을 판독했다. 제3 조정자는 두 기록과 핵심 원본 PNG 25장을 대조했다. 전체 원본 6,572장을 사람이 연속 재생한 검수가 아니다.

| 원본 | 확인한 차량 관측 | 최종 검수 판정 |
|---|---|---|
| 00024 | 노란 택시, 화면 LEFT | 후드 아래 접촉면·실제 사고 상대 미확정 |
| 00025 | 검은 세단, 화면 LEFT | 회전 중 접촉면·첫 바퀴 경계 접점 미확정 |
| 00026 | Prime 트레일러, 화면 RIGHT | 상향 시야·하단 흐림으로 바퀴·노면·접촉면 미확정 |
| 00027 | 근접 장면의 전방 승용차 | 시작 차량과의 동일성·실제 접촉 미확정 |
| 00028 | 시작부터 동일 차로 전방 SUV | 이 SUV가 실제 상대라면 규칙 진입 f0; 실제 접촉 미확정 |
| 00029 | 은색 차량, 화면 RIGHT | 실제 접촉 미확정; 진입 구간 추론은 검수자 간 불일치로 미채택 |

엄격한 collision/entry/direction/space 정답은 각각 0건이다. 이는 0% 정확도나 비충돌 판정이 아니라 **채점 가능한 분모가 0**이라는 뜻이다. 조건부로 유지한 것은 00028의 규칙 진입값 1개와 화면 위치 관측 4개다. 화면 위치를 최초 진입 방향 정답으로 바꾸지 않았다.

00029의 f570~585(19.0~19.5초)는 B가 제시한 추론이다. A와 조정자는 양끝을 진입 전후로 확정하지 못했다. 원래 제안을 불일치 기록으로 보존하고 정답 구간·거리 채점에 사용하지 않았다. 근접하거나 흔들린 구간도 물리적 접촉 구간으로 사용하지 않았다.

원본 SHA별 그룹은 구분했지만 사고·주행 원천 단위 독립성은 인증하지 않았다. 새 6건은 `new_source_diagnostic`이며, 기존 공개5는 접촉 회귀, 사람초안9는 개발 진단이라는 역할을 유지한다.

- [독립 검수 A](/Users/hyeongi/projects/blackbox-accident-ai/artifacts/stage2_goal_20260919/expert_mac_review/reviewer_a/review.md)
- [독립 검수 B](/Users/hyeongi/projects/blackbox-accident-ai/artifacts/stage2_goal_20260919/expert_mac_review/reviewer_b/review.md)
- [최종 조정 판정·근거](/Users/hyeongi/projects/blackbox-accident-ai/artifacts/stage2_goal_20260919/expert_mac_review/adjudication/records.json)
- [평가·해석 규약](/Users/hyeongi/projects/blackbox-accident-ai/artifacts/stage2_goal_20260919/expert_mac_review/protocol.json)

## 2. Mac 실행과 독립 검증

Apple M4 Max, 메모리 36GiB, macOS 27.0, Python 3.12.14, MLX 0.29.3/MLX-VLM 0.3.4에서 실행했다. 기존 `sync/native/deepstack_fix` 설정과 모델을 유지했다. 영상마다 새 worker가 원본 전부의 움직임을 계산하고 4회 실제 VLM 질의를 수행했다. 같은 질의 결과를 공유하여 마지막 접촉 점수의 residual·appearance 보조합 제거만 비교했다. 두 모델을 각각 실행한 속도 비교는 아니다.

원본 번호·native PTS가 대응된 무손실 RGB PNG 6,572장(5,848,190,424바이트)을 사용했다. 검수·근거·코드·모델 관련 1,182파일은 **17:14:40 KST에 동결**, 실제 모델 실행은 **17:14:57 이후** 시작했다. 모델 worker에는 ID·이미지 경로·해시만 전달했고 정답·PTS는 전달하지 않았다.

| ID | 기준선=후보 접촉 원본번호 / 초 | 진입 원본번호 / 초 | 방향 | 공간 |
|---|---:|---:|---|---:|
| 00024 | 613 / 20.433333 | 502 / 16.733333 | LEFT | 0 |
| 00025 | 568 / 18.933333 | 180 / 6.000000 | LEFT | 0 |
| 00026 | 301 / 10.033333 | 324 / 10.800000 | LEFT | 0 |
| 00027 | 591 / 19.700000 | 158 / 5.266667 | LEFT | 0 |
| 00028 | 613 / 20.032680 | 446 / 14.575163 | LEFT | 0 |
| 00029 | 599 / 19.966667 | 255 / 8.500000 | RIGHT | 0 |

시간은 원본 PTS다. 특히 00028을 일괄 30FPS로 환산하지 않았다.

- 6 worker 모두 exit 0, 실제 질의 총 24회.
- 최종 충돌 변경 0/6, 다른 세 출력 동일 6/6.
- worker wall time 합계 **243.312715초**, 영상별 24.602~48.946초. 프로세스 시작·입력 해시·모델 로드·광류·질의·저장을 포함하며 PNG 추출 시간은 제외한다.
- MLX peak 최대 **6.092430486GB**, 프로세스 RSS 최대 **4,172,939,264바이트**. 서로 다른 메모리 계측이며 합산하지 않는다.
- HuggingFace 오프라인 설정과 Python socket 차단을 적용했고 접속 시도는 0회다. OS 전체 네트워크 패킷을 계측한 결과는 아니다.

별도 실행 검증 전문가가 입력 PNG 6,572개·동결파일 1,182개 SHA, 원본 번호/PTS, 18개 점수 배열의 bytes와 argmax를 확인했다. 저장 응답을 반환하는 CPU 재생에서는 24개 질문·토큰 예산·시트 RGB 해시·전체 diagnostics가 일치했다. 검증용 새 모델 호출은 0회이며 광류·processor tensor를 재추론한 검사는 아니다.

[실제 실행 원기록](/Users/hyeongi/projects/blackbox-accident-ai/artifacts/stage2_goal_20260919/expert_mac_review/mac_run/report.json) · [예측·PTS 분석](/Users/hyeongi/projects/blackbox-accident-ai/artifacts/stage2_goal_20260919/expert_mac_review/prediction_analysis.json) · [독립 실행 검증](/Users/hyeongi/projects/blackbox-accident-ai/artifacts/stage2_goal_20260919/expert_mac_review/runtime_independent_review.md) · [실행 전 동결](/Users/hyeongi/projects/blackbox-accident-ai/artifacts/stage2_goal_20260919/expert_mac_review/review_lock.json)

## 3. 확인한 문제와 다음 순서

**기존 단일 실험의 판단은 유지한다.** 공개 접촉4/5 유지, 기존 사람초안4/9→5/9라는 결과는 가설을 만든00007 한 건의 개선이다. 이번 새6건은 출력 변화가0이고 정답 분모가0이므로 독립 성능·공식 S2 개선 근거를 추가하지 않는다. 새 가중치 탐색이나 접촉 재질문 실험은 실행하지 않았다.

**진입 문제에는 후보 수 증가만으로 해결되지 않는 증거가 있다.** 00028은 조건부 규칙값 f0가 내부 접촉 이전 구간과 12개 후보에 모두 남지만 모델은 f446을 출력했다. 관측 SUV를 같은 사고 상대로 전제할 때의 선택 불일치다. 실제 상대가 미확정인 만큼 엄격 진입 오답1건으로 집계하지 않는다. 기존 개발 자료의 첫 프레임 후보가 있어도 선택하지 않은 사례와 함께, 상대 동일성·시작부터 진입한 상태의 판정을 먼저 점검해야 한다.

**네 출력의 시간 참조 문제도 남았다.** 00026은 최종 접촉10.033333초보다 진입10.8초가 뒤에 있다. 내부 VLM 접촉은16.966667초다. 새6건 중00025/26/27/29의4건은 내부 VLM 접촉과 최종 접촉이 다르다. 00029는31.133333초와19.966667초로11.166667초 차이가 난다. 이는 기록상 참조 시점의 차이이며, 실제 상대 오식별이나 공간 오답을 확정한 숫자는 아니다. 정답 없이 진입을 강제로 잘라 수정하지 않는다.

다음 작업의 우선순위는 다음과 같다.

1. **정확도용 자료의 적격성부터 바꾼다.** 동일 상대·실제 접촉 근거·진입 경계를 확인할 수 있는 사고를 먼저 선별한다. collision/near-miss가 섞인 공급자 이벤트 시간과 화면 흔들림만으로 정답을 만들지 않는다. 새6건은 실행·불확실성 진단에 유지하고 정확도 확인셋에서는 제외한다.
2. **같은 상대의 시작 상태를 검증한다.** 예측 이전 전문가 검수 방식은 그대로 유지한다. 시작부터 차로 안에 있던 사례는 물리 진입 시점 미상과 규칙상 첫 프레임을 구분한다. 서로 다른 편집본·주행 원천도 확인한 뒤 사고 단위로 분리한다.
3. **그 자료에서 단일 후보의 접촉 개선을 확인한 뒤 다음 변경 하나를 결정한다.** 현재 jerk-only 후보를 추가 조정하지 않는다. 이후 실험에서는 시작 상태 판정과 최종 접촉에 대한 문맥 정렬을 동시에 바꾸지 않는다. 이번 관측만으로 어느 변경이 점수를 올린다고 단정할 수 없다.

요청한 전문가 검수·Mac 실측은 완료했고, 신뢰할 수 있는 새 접촉 정답 및 독립 점수 개선의 입증은 미완료다. 미완료 이유는 사람/CUDA 부재가 아니라 현재 6개 원본의 관측 한계와 사고 단위 독립성 미확정이다.

## 재현

이미 고정된 review_lock을 검증하고 새 결과 폴더에서만 실행한다. 기존 산출물을 덮어쓰지 않는다.

```sh
cd /Users/hyeongi/projects/blackbox-accident-ai
PYTHONDONTWRITEBYTECODE=1 artifacts/mac_experiments/stage2_mlx/.venv/bin/python \
  artifacts/stage2_goal_20260919/expert_mac_review/run_paired_mac.py \
  --output artifacts/stage2_goal_20260919/expert_mac_review/mac_run_recheck
```

실행 전에 `freeze_reviews.py`를 다시 호출할 필요는 없다. 기존 검수 동결과 모델을 그대로 사용한다. 이전 진행 문서와 STATUS는 [변경 전 기록](/Users/hyeongi/projects/blackbox-accident-ai/artifacts/stage2_goal_20260919/expert_mac_review/prior_status_snapshot/STATUS.json)에 보존했다.
