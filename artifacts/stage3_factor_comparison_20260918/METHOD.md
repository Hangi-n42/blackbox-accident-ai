# 사전 고정 비교 설계

2026-09-18. 기존 날짜 분할 3개 및 mixed_budget_rav4의 저장된 학습 표본을 그대로 재사용한다. 영상, 프레임 선택, 센서 대체 정답, 공개 정답을 수정하지 않는다. 각 fold 385개 학습 표본, 같은 StandardScaler와 LogisticRegression 설정을 사용하며 계수는 각 표현에 맞춰 새로 적합한다. 따라서 분류기 계수 자체를 동결한 입력 교체 실험이 아니라, 같은 분류기 학습 방법에서 광류/특징의 영향을 비교하는 실험이다.

## 요인별 조건

|조건|광류|특징|기준 대비 변경|
|---|---|---|---|
|dis_base|DIS FAST|기존864|기준 재현|
|dis_common|동일DIS|기존864 + affine계수36|공통 성분 추가|
|dis_residual|동일DIS|기존864 + 잔차864|잔차 추가|
|dis_common_residual|동일DIS|기존864 + 공통36 + 잔차864|두 성분 결합|
|dis_all|동일DIS|위 조건 + 적합진단24|신뢰도 정보 추가|
|raft_base|RAFT-Small C_T_V2|기존864|광류 모델만 교체|

공통/잔차/신뢰도 모두 기존과 같은 5/15/31프레임 평균, ±5/±15 시각 차분 및 중간시각 보간을 적용한다. 2D affine 계수는 교정된 카메라 회전/실제 자차 속도가 아니다. 잔차에도 깊이와 동적 객체가 혼재할 수 있다. 특징 수가 늘어나는 효과까지 포함한 실용적 비교이며 물리적 인과 기여도를 분리하는 실험은 아니다.

RAFT 입력은 동일한 회색조 프레임을 3채널로 복제한다. 색 정보 변경까지 동시에 섞지 않기 위한 통제이며 공식 RGB 입력 벤치마크와는 다르다. 12회 업데이트와 정규화 [-1,1]을 사용한다. 광류의 크기는 동일한 px/s로 환산하며 이후 ROI/시간 특징은 동일 코드다. 공개 가중치는 약3.8MB로 실험 폴더에 저장한다. 광류 사전학습/미세조정은 수행하지 않는다.

## 판정

각 날짜 묶음별 전체/차량별 Macro F1, 가속↔감속 방향별 반전, 등속 오류 및 공개 정답 회귀를 보고한다. 서로 중복되는 fold를 합친 종합 점수는 만들지 않는다. 새 반전과 기존 반전 수정 수를 별도 기록한다. 알려진74오류/101정상 시각은 진단이며 독립 검증으로 표현하지 않는다.

승격 조건: 각 fold F1 상승, 전체/차량별/반전 방향별 오류 증가 없음, 공개 F1 하락 없음. 운영/제출 모델은 이번 실험에서 교체하지 않는다. Depth Anything은 깊이 정보 필요성이 별도로 확인될 때만 다음 조건부 실험이며 이번 교차 비교에 함께 넣지 않는다.

## 출처 및 이용 조건

- Torchvision0.23 RAFT-Small C_T_V2 공식 가중치: https://docs.pytorch.org/vision/0.23/models/generated/torchvision.models.optical_flow.raft_small.html
- Torchvision 코드 BSD-3-Clause: https://github.com/pytorch/vision/blob/v0.23.0/LICENSE
- 가중치 학습 출처: FlyingChairs + FlyingThings3D (공식 모델 카드). 코드 라이선스와 학습 데이터/가중치 이용 조건을 혼동하지 않는다. 본 기록은 대회 제출 승인 증명이 아니다.
- comma 출처 및 기존 주석 이력: artifacts/stage3_training_basis_20260917/cases.json의 truth_source 및 원본 경로를 그대로 따른다. 센서 대체 정답은 대회 공식 정답이 아니다.

재실행: artifacts/mac_experiments/scipy_compat/.venv/bin/python artifacts/stage3_factor_comparison_20260918/run.py dis
이후 동일 명령의 마지막 인자를 raft로 변경. 완료된 특징 캐시는 재사용한다. run.py와 freeze.json으로 설정을 확인하고 각 *_execution.log에서 실행 종료 상태를 확인한다.
