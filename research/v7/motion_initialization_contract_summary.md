# 초기화 후보 CPU 계약 검토

`research/v7/solution/stage2_motion_init_v7.py`를 실제 V6 제출 모듈에 연결하고 합성 feature로 검사했다. **12개 검사, 고정 seed의180개 속성 사례 모두 통과**했다. 기존 코드·모델은 수정하지 않았으며 source SHA가 유지됐다. 실제 영상·정답·GPU·후보 정확도 평가는 사용하지 않았다.

| 계약 | 확인 결과 |
|---|---|
| 유효1프레임 | 점수0/원본 번호 유지, index1 범위 오류 없음 |
| 유효2프레임, jerk만 존재 | index1의 기여 제거 후 기존 argmax tie에 따라 첫 원본 번호 |
| 유효2프레임, residual/appearance 존재 | 두 번째 유효 프레임이 여전히 승자 가능 |
| 경계 residual/appearance가 매우 큼 | index1 점수8.5 유지, 첫 접촉 후보를 전면 제외하지 않음 |
| 경계의 uncapped jerk가 매우 큼 | 합계에서 큰 값을 빼지 않고 작은 appearance 기여0.25를 보존 |
| 다른 score 원소 | dtype·shape·bytes 동일, 입력 feature/base/new score 변경 없음 |
| 기존 승자가 index1이 아님 | 최종 argmax 유지 |
| 원본 번호와 유효 위치 구분 | 선행 무효 경로를 제외한 원본40/100 중 두 번째 유효 위치만 처리 |
| 호출 연결 | 기존 predict 함수에 V6 base/new score 그대로 전달, 모델 인스턴스1회, 다른3필드 유지 |
| 입력 검사 | 빈 feature, 비유한 feature/score 거부 |

차단 결함은 발견하지 않았다. `changed_score_indices=[1]` 로그는 실제 변경된 인덱스가 아니라 **수정 연산 대상**을 나타낸다. 원래 정규화 jerk 기여가0이면 bytes 변화 없이도 `[1]`을 기록한다. 이 로그만으로 실제 변경 파일 수를 집계하면 안 된다. 필요하면 추후 진단 이름을 `target_score_indices`로 명확히 하거나 실제 bytes 차이를 별도로 기록할 수 있으나 점수/출력 결함은 아니다.

모델·광류 호출은 대체 함수로 연결 계약만 검사했다. 따라서 실제 네 VLM 답변의 실행 재현성, 비정상 이미지의 실제 decoder 동작, 숨은 평가 성능을 검증한 것은 아니다. synthetic core 함수 검사는 V6의 실제 `_scores_from_features`와 `_robust_scale`을 사용했다.

근거: `test_motion_initialization_contract.py`, `motion_initialization_contract_results.json`의 검사/피검사 SHA 및 결과. 과학적 해석과 경계 접촉 손실 위험은 `motion_initialization_review.md`에 별도 기록했다.
