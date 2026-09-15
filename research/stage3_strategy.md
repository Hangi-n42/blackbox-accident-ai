# Stage 3 전략 검토

검토일: 2026-09-11. 작성 범위: 로컬 대회 문서·학습/추론 노트북·공개 라벨과 아래 공식 웹 출처. 다운로드·학습·성능 측정은 하지 않았다. 아래 실험 설계는 가설이며 검증된 대회 성능 주장이 아니다.

## 1. 평가가 요구하는 우선순위

`대회_통합_정보.md` 2.4, 5, 7~9절 기준:

- Stage 3는 `0.70 * accel_macro_f1 + 0.30 * steer_macro_f1`; 종합점수 기여는 각각 28%, 12%다.
- 가감속 4클래스와 조향 3클래스 모두 정의된 전체 범주로 macro F1을 계산한다. 다수 클래스만 잘 예측하는 accuracy 최적화는 목표와 어긋난다.
- 정답 STOPPED 프레임만 조향 채점에서 제외된다. 예측 STOPPED로 조향 평가를 피할 수 없다.
- 평가 영상은 10Hz. N개를 디코딩하면 `sample_index=0..N-1` 전부 반환한다.
- 한 영상의 미래·과거 문맥과 영상 내 후처리는 사용 가능하다. 다른 파일의 통계·예측으로 현재 파일을 보정하면 안 된다.
- 평가 데이터에서 모델 갱신, 임계값 학습, pseudo-labeling, 온라인 adaptation은 금지한다. 모든 학습·전역 임계값·정규화 계수는 공개 학습자료로 고정한다.
- 총 60분은 세 Stage 합산이다. Stage 3의 겹치는 16프레임 클립마다 큰 모델을 다시 실행하는 방식은 비용 대비 우선순위가 낮다.

## 2. 로컬 baseline의 확인된 상태

`Baseline/[Baseline_Train]_3Stage_학습.ipynb`의 `fit_stage3()`는 MViTv2-S를 `weights=None`으로 만들고 라벨마다 16프레임을 읽는다. 분류기 두 개에 가중치 없는 cross entropy를 더한다. 검증 점수로 체크포인트를 선택하는 로직이 없다.

`Baseline/data/stage3/labels.csv`는 5영상 × 10라벨이다. 가감속 분포는 CONSTANT 30, ACCELERATING 9, DECELERATING 8, STOPPED 3. 조향은 STRAIGHT 39, LEFT 6, RIGHT 5다. 프레임 120이 6초이므로 공개 영상 라벨의 시간축은 20Hz를 가리킨다. 이는 평가 10Hz와 별도로 처리해야 한다. 실제 영상 metadata 시간을 정답으로 가정하면 안 된다.

현재 라벨만 이용할 경우 5-fold leave-one-video-out 외에 신뢰할 큰 검증셋이 없다. 인접 프레임을 서로 다른 fold로 나누지 않는다. 특히 한 fold에 일부 클래스가 없을 수 있으므로 fold 평균만 제시하지 말고 전체 out-of-fold 예측을 합친 macro F1과 클래스별 F1/건수를 같이 기록한다.

## 3. 우선 실행할 가설

### H1. 영상의 배경 운동을 먼저 추정하는 작은 모델이 현재 무작위 대형 영상모델보다 실용적이다

이유는 라벨 수와 목표 변수의 성격이다. 목표는 사물의 정체보다 카메라 자차운동과 그 시간 변화에 가깝다. 단, 배경 optical flow가 CAN 값과 동일한 것은 아니다.

실행 순서 제안:

1. 프레임을 한 번만 순차 디코딩한다. 공개 20Hz 영상은 라벨의 시간축을 보존한 채 10Hz frame pair로 정규화한다. 실제 제출 영상에는 다시 시간축 변환하지 않는다.
2. 작게 축소한 회색영상에서 OpenCV DIS 또는 Farneback dense optical flow와 sparse LK + RANSAC motion을 추출한다. 학습된 외부 가중치 없는 구현을 첫 비교군으로 둔다. 해당 API는 OpenCV 공식 문서에 제공된다.[1]
3. 하늘·보닛·시간 자막을 제외한 복수 ROI의 flow 크기/수평/수직 성분, 분산, quantile, forward-backward consistency, RANSAC inlier 비율 등을 특징으로 구성한다. 도로 중앙과 화면 측면을 따로 집계한다.
4. 방사형 팽창 성분, 배경 수평 운동, 움직임 잔차, 시간 미분과 0.5~3초 고정 시간창 평균/분산을 만든다. 창 길이는 실험 후보이며 대회 공식 값이 아니다.
5. 공개 50라벨로는 정규화 logistic regression 및 작은 tree ensemble을 비교한다. 원시 특징이 영상 ID나 촬영 기기를 암기하지 않는지 영상 단위 검증한다. 특징 전처리와 분류기를 전체 fold 훈련에 맞춰 각각 학습한다.

한계: 가까운 벽/차량, 곡선, 언덕, 흔들림, 비·야간, 노출 변화가 flow를 바꾼다. 'flow magnitude 증가 = 가속'만으로 판정하면 장면 깊이 변화와 회전을 가속으로 오인한다. 정지 중 움직이는 전방 차량도 자차 출발로 오인할 수 있다. 장면 전체 median 하나로 해결하지 말고 배경 합의도와 여러 ROI를 사용한다.

### H2. 공개 CAN 동기 영상으로 연속량을 학습한 뒤 대회 라벨을 보정하는 방식이 확장 경로다

가장 먼저 검토할 데이터는 comma2k19다. 공식 자료에는 전방 영상, CAN 속도·조향각, 시간별 camera pose가 있다. 데이터셋은 33시간 이상, 2019개 1분 세그먼트이며 캘리포니아 특정 고속도로 구간과 두 차종으로 구성되어 도메인 다양성이 제한된다.[2]

공식 Hugging Face 계정은 MIT를 표시하며 raw ZIP 10개를 게시한다. `raw_data` 전체는 페이지 표기 94.6GB, Chunk_1은 8.73GB다. HF `load_dataset` 기본 demo는 64행의 preview/log이므로 이를 2019개 전체 영상 접근으로 오인하지 않는다. 영상 포함 raw ZIP 또는 별도 파일을 실제 확인해야 한다.[3][4]

학습 제안:

- CAN 속도와 영상 timestamp를 정렬하고 평활한 속도 미분을 가속 보조 정답으로 만든다. CAN 조향각은 차량별 부호·중립 오프셋을 검증한 뒤 사용한다. 물리 좌표계를 확인하지 않고 LEFT/RIGHT를 붙이지 않는다.
- flow + 작은 RGB encoder + temporal convolution으로 속도, 가속도, 조향각을 회귀하고, 별도 범주 헤드를 학습한다. 원본 CAN은 학습 정답으로만 사용하며 제출 입력으로 기대하지 않는다.
- 외부 데이터의 범주화 임계값은 대회 비공개 임계값과 같다고 주장하지 않는다. 대회 공개 라벨로 calibrator를 학습하되 label 수가 적으므로 지나치게 세분화한 threshold search를 피한다.
- route 또는 수집 session 단위로 분할한다. 같은 route의 인접 1분 segment를 train과 validation에 나누지 않는다. 공개 baseline과 외부 자료의 중복 여부도 공개 자료끼리만 점검하고 중복은 평가 fold에서 제거한다.
- STOPPED·출발·제동·좌우 조향 구간을 CAN으로 찾아 샘플링한다. 고속도로 데이터에서 정지가 부족하면 전체 시간만 늘려도 macro F1 문제가 남는다.

가설 채택 조건: H1과 같은 공개 holdout에서 accel macro F1이 개선되고, minor class F1 붕괴가 없어야 한다. 단순 외부 데이터 regression MAE 개선을 대회 분류 점수 개선으로 대신하지 않는다.

### H3. 고정된 시간 문맥과 class calibration이 급격한 프레임별 오분류를 줄일 수 있다

프레임별 확률에 영상 내 고정 길이 평균, 작은 temporal network, 또는 공개 train에서 학습한 상태전이 제약을 비교한다. 다수 클래스에 과도하게 끌리지 않도록 class-balanced sampling/weighted loss와 고정 class bias를 검증한다.

후처리는 실제 짧은 가감속과 조향 전환을 지울 수 있다. 따라서 평활 유무를 class별 F1로 비교한다. 다수 클래스 비율을 평가 영상들의 통계로 맞추는 보정은 규칙 위반이므로 하지 않는다. 단일 영상의 정규화도 모델 갱신처럼 작동하는 학습 절차를 넣지 않고 사전에 정의한 결정론적 전처리만 사용한다.

## 4. 외부 자원 접근·라이선스 확인

| 자원 | 확인된 접근·용도 | 라이선스·제한 판단 |
|---|---|---|
| commaai/comma2k19 | 공식 GitHub 설명과 HF raw ZIP 목록을 로그인 없이 읽음. 실제 다운로드 속도/완료는 미확인 | 공식 HF 데이터 카드 MIT + 공식 GitHub MIT 원문 확인. 저작권·허가문을 보존한다. 가장 직접적인 1순위 데이터 후보.[2][3][4][5] |
| nuScenes CAN bus | 공식 튜토리얼에 속도/가속/steering 신호와 다운로드 절차. 영상+CAN 추가 자원 후보 | 공식 약관 CC BY-NC-SA 4.0 + 추가 약관. 공식 페이지는 등록·로그인 후 다운로드 안내. 대회가 비영리 허용 자원을 허용하더라도 수상 산출물 독점 귀속/재배포와 조건을 별도로 기록해야 한다. 본 조사에서 로그인 다운로드는 수행하지 않음.[6][7][8] |
| RAFT Small | torchvision 공식 소스에 `C_T_V1`/`C_T_V2` 다운로드 URL과 990,162 파라미터 명시. GPU 사용 가능한 flow 대안 | RAFT/torchvision 코드 BSD-3-Clause 원문 확인. 코드 라이선스만으로 학습 데이터와 모든 가중치 이용조건을 일괄 단정하지 않는다. 사용한다면 모델 weight 출처·훈련데이터 고지·라이선스를 별도 보존. 최초 구현은 weights 없는 OpenCV 방식이 절차상 간단.[9][10][11] |
| comma speedchallenge | 공식 repo에 20,400프레임 train 영상과 프레임별 speed가 설명됨 | 확인한 repo root에는 LICENSE가 보이지 않는다. 공개 repo라는 이유만으로 활용 허용을 단정할 수 없어 현 단계 학습 제외.[12] |

RAFT Small을 시험할 경우 가중치는 개발 중 받아 ZIP에 포함하고 `weights=None`으로 instantiate 후 로컬 load한다. 평가 중 URL 다운로드를 호출하지 않는다. 공식 구현은 작은 RAFT가 큰 모델보다 빠른 대신 정확도를 절충한다고 설명한다. 이 대회의 점수·실행시간은 별도 측정 대상이다.[13]

## 5. 즉시 구현에 필요한 검증 명세

- 디코딩 N프레임과 반환 N행, 중복/누락 sample_index, 두 라벨의 허용 범주 검증.
- 공개 라벨 비교는 `frame_index` 위치 기준이며 평가 형식의 sample_index와 혼동하지 않는다.
- 의미 있는 검증: 동일 영상 파일을 단독 및 다른 영상과 함께 넣었을 때 그 파일의 예측이 동일해야 한다. 파일 순서 변경에도 동일해야 한다.
- 전체 validation out-of-fold macro F1, 4/3개 class별 precision/recall/F1, 정답 STOPPED를 제외한 steer 점수를 기록한다.
- 모델이 frame 번호/ID/파일명으로 정답을 찾는 lookup을 포함하지 않아야 한다.
- 긴 영상에서 시간과 peak memory를 측정하고 순차 decoding 또는 작은 고정 buffer로 처리한다. 전체 frame RGB를 GPU에 올리지 않는다.
- 개발 환경·seed·라이선스·훈련 split·hyperparameter·checkpoint hash를 보존한다.

## 6. 판단

현시점 최고 점수 모델은 확인할 수 없다. 정당한 다음 단계는 H1을 빠른 재현 가능한 비교군으로 만들고 공개 영상 단위 검증을 끝내는 동시에, comma2k19 공개 raw data를 확보하여 H2를 실험하는 것이다. 무작위 MViT를 예제 50개로 학습해 제출하는 전략에는 일반화 근거가 없다. Stage 3의 불확실성을 무시한 강한 규칙 기반 출력 역시 높은 점수를 보장하지 않는다.

## 7. 후속 구현·측정 기록

연구 이후 구현을 승인받아 `solution/stage3.py`, `research/train_stage3.py`를 작성했다. OpenCV DIS 256px, 고정 12개 ROI와 시간 특징 864차원, ExtraTrees/RandomForest/regularized logistic regression을 비교했다. 공개 5영상의 각 영상을 통째로 제외하여 총 50개 out-of-fold 예측을 생성했다. 조향 학습·검증 모두 정답 STOPPED 행을 제외했다.

| 공개 실험 | 가감속 Macro F1 | 조향 Macro F1 | 가중 Stage 3 |
|---|---:|---:|---:|
| DIS 특징 + 선택된 ExtraTrees | 0.636659 | 0.467532 | 0.585921 |
| 동일 모델 + 물리적으로 일관된 좌우 반사 증강 | 0.674431 | 0.641818 | 0.664647 |
| 외부 CAN 범주 proxy + 공개 라벨, 가감속 logistic / 조향 RandomForest | 0.776009 | 0.850304 | 0.798298 |

반사는 ROI 열 역순, 수평 flow 부호 반전 및 quantile 순서 교환, LEFT/RIGHT 교환으로 구성했다. 속도/가감속 정답은 유지한다. 증강은 fold 학습 자료에서만 생성한다. 동일한 OOF 결과로 모델을 선택하므로 별도 독립 검증 성능으로 해석하면 안 된다. 공개 라벨 50개의 작은 표본도 한계다. 상세 confusion matrix와 class F1은 `stage3_validation_public*.json`, 개별 예측은 `stage3_oof_public*.csv`에 기록했다.

`validate_stage3_contract.py`에서 31프레임 전부 반환, 인덱스 연속성, 라벨 범주, 파일명 변경 및 다른 파일 추가 시 해당 영상의 예측 동일성을 실제 검증했다. 이는 성능 시험과 별개인 제출 함수 규격 검사다.

comma2k19는 전체 ZIP 다운로드 대신 HTTP Range로 일부 압축 멤버만 읽는다. 다운로드 스크립트는 영상·CAN·frame_times만 추출하고 원본 URL/선택 segment 목록을 manifest에 저장한다. 공개 예제와의 영상 중복을 감지해 외부학습에서 제외한다. 실제 외부 학습 결과는 후속 JSON 실험 기록을 우선한다.

최종 선택은 외부 CAN proxy 학습 모델이다. 24개 서로 다른 route segment를 내려받고 공개 OPEN_001과 정렬된 영상의 thumbnail RMSE=0인 1개를 제외하여 23 route를 사용했다. 내려받은 파일·manifest·license의 합계는 inventory 생성 시 905,263,762바이트였다. 모든 source 파일 SHA256은 `external_data/comma2k19/inventory.json`에 기록했다. 외부 학습 표본 2,761개의 가감속 proxy 분포는 가속 521, 감속 570, 등속 1,393, 정지 277이다. 같은 route의 다른 segment가 public에 포함되어 있는지까지 완전히 배제한 것은 아니다.

외부 CAN 범주화는 속도 <0.3m/s를 정지, 평활 속도의 시간미분 절댓값 0.25m/s²를 가감속 경계, 조향각 ±2도를 방향 경계로 사용했다. 이는 공식 대회 임계값이 아니다. 실제 공개 영상/CAN 대조에서 일부 기준 불일치도 확인했다. 외부 범주 proxy와 공개 라벨을 함께 학습한 모델의 OOF 결과로 선택했으며, CAN 연속량 회귀 후 공개 라벨로 분류하는 대안도 실험했다. 후자의 조향 F1은 0.686186이었고 가감속은 저하되어 최종 선택하지 않았다.

최종 `model/stage3/motion_model.joblib`의 SHA256은 `c52ecd15cf68856f2a7b0b37b120c7a961b3bd8889a33af4ec47f8b9719bb3b3`다. 이 모델로 10Hz 변환 공개 영상 5개를 실제 `predict_stage3`에 넣었다. 별도 PyAV 디코딩으로 확인한 600/601/599/599/599프레임과 반환 행 수가 모두 같았고, 인덱스·라벨 검사를 통과했다. 로컬 측정 시간은 50.75초였다. 이 입력은 학습에 사용한 공개 자료이므로 해당 실행을 정확도 검증 점수로 제시하지 않는다. 전체 비공개 평가의 60분 통과 여부는 이 소규모 실행으로 보장할 수 없다.

재현 순서(가상환경의 Python 사용): `research/acquire_comma_subset.py --chunk 1 --segments 12`, 동일하게 chunk 3, `research/train_stage3.py --external`, `research/select_stage3.py`. 선택 비교 전체를 재현하려면 public 기본·`--mirror` 실험과 `train_stage3_transfer.py`도 실행한다. 최종 체크포인트의 두 분류기는 sklearn 객체이며 사전학습 외부 가중치가 필요 없다. `research/finalize_stage3.py`는 이전 중간 조합의 실험 기록이므로 최종 선택용 실행은 `select_stage3.py`를 사용한다.

## 출처

1. OpenCV 4.10 공식 optical flow API: https://docs.opencv.org/4.10.0/dc/d6b/group__video__track.html
2. comma.ai 공식 comma2k19 설명: https://github.com/commaai/comma2k19
3. comma.ai 공식 HF 데이터 카드: https://huggingface.co/datasets/commaai/comma2k19
4. 공식 raw ZIP 목록: https://huggingface.co/datasets/commaai/comma2k19/tree/main/raw_data
5. comma2k19 MIT 원문: https://raw.githubusercontent.com/commaai/comma2k19/master/LICENSE
6. nuScenes CAN 공식 튜토리얼: https://www.nuscenes.org/tutorials/can_bus_tutorial.html
7. nuScenes 다운로드/로그인 안내: https://www.nuscenes.org/nuscenes?tutorial=can-bus
8. nuScenes 공식 이용약관: https://www.nuscenes.org/terms-of-use
9. torchvision RAFT 공식 구현/가중치 메타데이터: https://github.com/pytorch/vision/blob/main/torchvision/models/optical_flow/raft.py
10. RAFT 라이선스: https://raw.githubusercontent.com/princeton-vl/RAFT/master/LICENSE
11. torchvision 라이선스: https://raw.githubusercontent.com/pytorch/vision/main/LICENSE
12. comma.ai speedchallenge 공식 repo: https://github.com/commaai/speedchallenge
13. torchvision 공식 RAFT 튜토리얼: https://docs.pytorch.org/vision/master/auto_examples/others/plot_optical_flow.html
