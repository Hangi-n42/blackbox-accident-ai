# DADA 외부 추론 미실행 결정

상태: `not_run_due_to_upstream_candidate_rejection`.

Root의 최종 결정에 따라 V5-simple은 공개 known-regression gate 실패로 미채택되었으며, DADA 외부 추론은 실행하지 않았다. 외부 PASS, FAIL, 개선 평가 또는 성능 수치는 없다. Stage2 원 V3 복원은 root 담당 결정이며 이 기록에서는 복원 구현을 검증하지 않았다. 공개 예측 결과를 이 기록 작성자가 독립 재계산한 것은 아니다.

기존 합의본·평가 계획·원본 입력 SHA binding은 변경 없이 보존했다. 실행용 모델·runner·계측 코드의 추가 binding은 진행하지 않았다. 독립 검토자는 외부 예측 파일을 열람하거나 외부 추론을 수행하지 않았다.

추가 영상을 확보했지만 공식 정확 GT는 0개다. 같은 대상을 가정한 기원측 6개 및 폭이 넓은 entry 구간 5개는 사전 선언된 가설 반증에만 사용할 수 있다. 정확한 대회 정의의 라벨 부족이라는 근본 한계는 해결되지 않았고, 일반화 성능 개선도 입증되지 않았다.

검증에서 악화한 변경을 제외하고 기존 비교 모델로 복원하는 것은 관측된 위험에 근거한 조치다. 복원 자체를 신규 성능 향상 또는 최종 목표 달성으로 표현해서는 안 된다. 이미 탈락한 후보를 살리기 위해 외부 기준을 사후 완화하지 않는다.

보존 파일 SHA256:

- `consensus_frozen.json`: `6c71ed570effa495bac9229a5ccf3e9640983e1f08f3a452ac84c97eec47cdd4`
- `external_plan_frozen.json`: `48696cfe03f7c366f05b803350b72fde06ba2fc90e8335bf31d433a8a65e10f4`
- `external_input_binding.json`: `e7bc455361995d57171ddf933867b9320ccf3ccbff7c1d19ea5f5cf6084960fe`
