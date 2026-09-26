# Stage3 상태 학습 비교 — 2026-09-19

사용자가 승인한 순서: 기준 정렬 → DIS 고정 학습 목표/헤드 비교 → V-JEPA 동결 특징 비교 → 개선 여부 판정. 모델 교체와 특징 교체를 한 조건으로 묶지 않았다. 결과를 근거로 운영 파일을 자동 변경하지 않는다.

## 평가와 데이터

- 공식 공개 5영상 50시각, 영상별 leave-one-out. 4영상 정답만 학습에 추가한다. 가감속 4종 Macro-F1, GT STOPPED를 제외한 47시각 조향 3종 Macro-F1. S3 = 0.7 × 가감속 + 0.3 × 조향.
- 조향은 `research/stage3_oof_external.csv`의 기존 `steer_forest` OOF를 모든 조건에서 그대로 사용한다. 새 가감속만 비교하기 위한 고정 합성 OOF 점수이며, 새 전체 제출 파이프라인을 실행한 점수가 아니다.
- comma 23개 기존 영상/센서만 사용. 공개 영상과 정렬된 중복 1개는 기존 선정에서 이미 제외돼 있다. 공개 평가용 학습은 엄격 범주 1,083시각과 나머지 공개 4영상의 40정답이다.
- 별도 외부 날짜 진단은 이전의 3분할, 각 학습 385시각 그대로다. 평가 1,612/1,167/561시각. 차량별 날짜와 원천은 학습/평가 간 겹치지 않지만, 평가 날짜가 3분할 사이에 반복되므로 합산 독립 표본으로 보지 않는다.
- 공개 자료와 comma 모두 과거 개발에 노출됐다. 대회 전체/private/독립 일반화 점수가 아니다. 전체 출처 중복 가능성은 기존 검사 범위를 넘어 배제하지 못한다.
- 원본/정답은 변경하지 않았다. 엄격 범주 밖의 시각은 분류 손실에서 제외하고, 유효 센서가 있으면 회귀 목표로만 활용했다. 센서는 추론 입력에 넣지 않는다.

## 고정 조건

1. 기존 StandardScaler + LogisticRegression(C=.03, balanced) 기준.
2. 작은 시간창 분류기: t−0.5초/t/t+0.5초 입력 각각 864→32 GELU, 합쳐 96→32 GELU, 4분류. DIS의 기존 시간 요약 특징은 그대로 유지.
3. 같은 분류기에 가속도 보조회귀만 추가: 기존 정제 자료의 11점/1초 속도 기울기, 학습 센서 평균·표준편차로 표준화, SmoothL1 가중치 0.2. 공식 종가속도 정답이라고 부르지 않는다.

분류/보조 조건은 같은 센서 풀, 같은 분류 마스크, 같은 초기화, 같은 분류 가중치와 scaler를 사용한다. 회귀층은 분류 전용에도 생성해 초기화 순서를 맞춘다. 학습 scaler는 선택된 분류 학습 시각으로만 계산한다. 전체 특징 z-score를 ±8에서 제한하는 규칙은 두 신경망에 동일하다. 선형 기준은 과거 알고리즘을 그대로 유지하므로 신경망과의 차이는 비선형 구조·시간창·학습법 묶음의 효과다. 시간창 하나의 독립 효과라고 해석하지 않는다.

seed 17/42/73, AdamW lr .001, weight_decay .01, full-batch 120epoch, gradient clip1. 학습 손실과 추론은 CPU2스레드. 외부 평가 결과로 epoch/가중치/임계값을 탐색하지 않았다.

## 동결 V-JEPA 비교

공식 `facebookresearch/vjepa2` commit와 EMA 체크포인트 SHA는 `vjepa_provenance.json`에 기록. strict 로딩, encoder 전체 동결, 실제 파라미터 86,833,152개. predictor/optimizer는 추론에 쓰지 않는다.

기존 10Hz RGB에서 16프레임씩 처리한다. 공식 평가 전처리의 short-side438 → center crop384, ImageNet 정규화. PyAV와 OpenCV bilinear를 사용하는 Mac 어댑터다. EMA encoder 코드는 수정하지 않는다. 모델 hub의 localhost URL을 호출하지 않고 공식 README의 실제 체크포인트를 읽는다. 대소문자가 충돌하는 giant 모델 설정 파일은 이번 ViT-B 실행에서 읽지 않는다.

시간 tubelet 8개 각각의 공간 토큰만 평균해 768차원으로 만들고, 실제 10Hz 프레임 인덱스에 대응하는 tubelet 중심에 놓아 보간한다. 전체 시간축을 평균하지 않는다. 마지막 짧은 창만 끝 프레임으로 패딩한다. 출력은 전 시각 10Hz다. 공간 위치를 명시적으로 보존하는 방식은 아니며, 1.6초 창·공간 평균·중앙 crop이라는 제한이 있다. 이 한 설정의 성능을 V-JEPA 전체 능력으로 일반화하지 않는다.

768차원 뒤에 0을 채워 DIS와 같은 864 입력 및 같은 헤드 파라미터 수를 유지한다. 다른 encoder를 위한 별도 최적 헤드나 손실을 탐색하지 않는다. 선형↔선형, 분류↔분류, 보조회귀↔보조회귀끼리 비교한다. 영상 전처리와 표현이 함께 바뀌므로 사전학습 가중치만의 효과라고 부르지는 않는다.

`timm1.0.15`, `einops0.8.1`은 이 폴더의 vendor에만 no-deps 설치했다. 기존 venv 패키지/lock/운영 코드와 모델은 변경하지 않았다. 원본 공식 저장소의 라이선스 고지를 함께 보관한다.

## 사후 원인 점검과 추가 대조

고정120 DIS에서 학습 F1=1.0, 날짜 평가 저하를 관측했다. 이 결과 이후, 새 대조군 결과를 보기 전에 `route_early_stop/freeze.json`을 만들었다. 독립적으로 미리 계획한 최초 실험처럼 표시하지 않는다.

학습 원천만 해시 순서로 약20%(최대35%) 분리한다. 가능한 차량별 최소1원천, 학습의 모든 클래스 유지 조건을 사용한다. 내부분할에서도 scaler/회귀 정규화에 검증 원천을 넣지 않는다. 원천별 평균 class-balanced CE로 1~120epoch 중 선택하고, 새 초기화로 전체 외부학습 자료를 그 epoch만큼 다시 학습한다. 외부 평가 날짜/held-out 공개 영상은 선택에 사용하지 않는다. DIS/V-JEPA 모두 같은 내부 원천과 선택법을 적용한다. 기존120epoch 결과는 그대로 보존한다.

## 사전 채택 기준

공개 S3 평균 +0.01 이상, 모든 seed에서 비감소, 새 공개 반전 없음. 날짜별 평균 가감속 F1 비감소, 날짜 및 차량별 평균 반전율 증가 1%p 이하. 공식 점수에 반전 페널티가 추가되는 것은 아니다. 위험 기준과 공식 점수의 증감을 따로 기록한다. 통과해도 개발 후보일 뿐 운영 자동교체나 private 개선 주장이 아니다.

## 재현

프로젝트 루트에서 Python은 `artifacts/mac_experiments/scipy_compat/.venv/bin/python`을 사용한다.

```sh
PY=artifacts/mac_experiments/scipy_compat/.venv/bin/python
$PY artifacts/stage3_state_learning_20260919/run.py dis
$PY artifacts/stage3_state_learning_20260919/diagnose.py dis
$PY artifacts/stage3_state_learning_20260919/extract_vjepa.py
$PY artifacts/stage3_state_learning_20260919/run.py vjepa
$PY artifacts/stage3_state_learning_20260919/diagnose.py vjepa
$PY artifacts/stage3_state_learning_20260919/early_stop.py dis
$PY artifacts/stage3_state_learning_20260919/early_stop.py vjepa
$PY artifacts/stage3_state_learning_20260919/diagnose.py dis --inner-stop
$PY artifacts/stage3_state_learning_20260919/diagnose.py vjepa --inner-stop
$PY artifacts/stage3_state_learning_20260919/summarize.py
$PY artifacts/stage3_state_learning_20260919/clipping_control.py
$PY artifacts/stage3_state_learning_20260919/coverage_control.py
```

주 실험은 완료한 학습과 특징 파일을 건너뛴다. `coverage_control.py`는 이미 결과가 있으면 보존을 위해 중단한다. 위 명령은 실제 실행 순서의 기록이며 완료 폴더에서 전부 다시 실행하라는 지시가 아니다. 독립 재실행은 소스와 필요한 고정 입력을 새 실험 폴더에 복사해 빈 출력 경로를 사용한다. 원본 실험 결과를 지우고 재실행하지 않는다.
