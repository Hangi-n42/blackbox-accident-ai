# V7 검증·자원 계획: 모든 Stage의 개선 가능성과 중단 조건

작성 범위는 1차 논문·공식 코드·기존 감사의 읽기와 설계다. 모델 실행, 가중치 다운로드, 기존 코드/모델 변경은 하지 않았다. 아래 수치 예산과 후보 구조는 **실행 전 제안**이며 측정 성능이 아니다.

## 1. 현재 확인된 병목

Root가 전달한 V6 공식 Stage2 점수는 0.2817803332이고, 새 사람 접촉 참조 3개에서는 0/3→1/3이었다. 이 표본에서의 증가가 전체 Stage2 성능이나 다른 세 항목의 정확도를 입증하지는 않는다. V6에 사용한 9개 Nexar 원천은 이후 후보의 미사용 검증 자료로 재사용할 수 없다.

기존 6개 전체 프레임 진단은 fine 접촉 후보 포함 4/6인데 내부 선택 0/6이었다. 진입은 알려진 5개 중 범위 제한 1, 균등 후보 누락 1, 후보가 있는데 선택 실패 3으로 나뉜다. 36개 coarse/fine/entry 선택 모두 제공 번호를 수락해 파서 fallback이 설명이 되지 않았다. 따라서 프레임 수 증가나 JSON 강제만으로 해결된다는 설명은 반증됐다. 근거: `research/v6_stage2/v5_fullframe_error_decomposition_v2/execution_notes.md:7`.

Stage3는 원래 144개 ROI 분위수 특징에 평균 창 5/15/31과 시간 차분 2개를 붙인 864차원이다. 시간 정보가 전혀 없는 모델이 아니다. 다만 ROI 안에서 픽셀 대응·객체별 이동·깊이 정보가 없어지고, 고정 평균/차분 밖의 순서 패턴은 직접 학습하지 않는다. radial 중심도 고정 좌표다. 근거: `solution/stage3.py:18–69`.

Stage3의 정확히 대응하는 공개–CAN 감사는 **10개 가감속 GT 전부 CONSTANT**다. 전체 공개 라벨 50개까지 전부 CONSTANT라는 뜻은 아니다. 비교 가능한 10개로 가속·감속·정지의 proxy 신뢰도는 검증하지 못한다. 차속 미분 proxy와 공식 종가속도 신호의 생성 차이는 확인되지만, 공식 비공개 임계값이나 저성능의 단일 원인은 확인 불가다. `research/v6_stage3/summary.md` 참조.

Stage1에는 **주행 영상의 실제 화면 재촬영 대응쌍 0쌍**이라는 제약을 유지한다. DLC 문서 재촬영 자료와 합성 변화가 실제 블랙박스 재촬영 대응쌍을 대신하지 않는다. Stage1의 별도 forensic 전문가 실험은 source/display/capture-device 분할과 물리적 재촬영 검증의 유무를 기록해야 한다.

## 2. 주축 논문 3개와 보조 calibration 근거

### A. SlowFast Networks for Video Recognition — ICCV 2019

저빈도 공간 의미 경로와 고빈도 운동 경로를 분리하고 결합한 행동 인식 모델이다. 운동의 시간 해상도를 공간 처리량과 분리할 수 있다는 근거다. 논문의 행동 인식 결과는 CAN 범주나 0.3초 접촉 시점 정확도의 증거가 아니다. [CVF 논문](https://openaccess.thecvf.com/content_ICCV_2019/html/Feichtenhofer_SlowFast_Networks_for_Video_Recognition_ICCV_2019_paper.html)

**적용 추론:** Stage3의 낮은 비용 광류 시계열과 Stage2의 희소 전체 장면 의미를 구분해 계산 예산을 배분한다. 고빈도 경로를 모든 프레임의 대형 VLM 호출로 구현하지 않는다. 공식 Kinetics 모델 링크는 공개되어 있지만 행동 분류 가중치를 CAN head로 직접 바꿀 수 없다. 코드 라이선스는 Apache 2.0이고 [공식 model zoo](https://raw.githubusercontent.com/facebookresearch/SlowFast/main/MODEL_ZOO.md)에 checkpoint URL이 있다. 가중치 파일 내부 고지·바이트 취득·로컬 실행은 이번에 확인하지 않았다. [코드 라이선스](https://raw.githubusercontent.com/facebookresearch/SlowFast/main/LICENSE)

### B. Deep Patch Visual Odometry — NeurIPS 2023

희소 patch 대응과 recurrent update, bundle adjustment로 카메라 pose를 추정한다. 논문은 RTX3090에서 기본 설정 60FPS/4.9GB, 빠른 설정 120FPS/2.5GB를 보고한다. 이 숫자는 해당 논문의 EuRoC 조건이며 RTX2080 SUPER나 대회 처리량으로 환산하지 않는다. 학습은 TartanAir 합성 자료이며 궤적 평가는 scale 정렬을 사용한다. [논문](https://proceedings.neurips.cc/paper_files/paper/2023/file/7ac484b0f1a1719ad5be9aa8c8455fbb-Paper-Conference.pdf)

**적용 추론:** 픽셀 분위수 대신 지속되는 대응점의 회전·이동 일관성을 보조 표현으로 얻는 연구 방향이다. monocular pose 이동량은 절대 CAN 속도·종가속도와 동치가 아니며, 정차/저속·와이퍼·근접 이동차량·모션 블러에 대한 실제 주행 검증이 필요하다. depth/pose를 쓰는 접근은 이전 2D global/residual 후보와 구조적으로 다르지만 개선은 미확인이다.

획득 판단: [공식 저장소](https://github.com/princeton-vl/DPVO)는 MIT 코드와 모델 링크를 제공한다. [다운로드 스크립트](https://raw.githubusercontent.com/princeton-vl/DPVO/main/download_models_and_data.sh)는 models.zip 공개 URL을 포함한다. 모델 payload의 현재 다운로드 성공·SHA·내부 별도 이용조건은 확인하지 않았다. 코드 MIT를 checkpoint의 모든 권리로 확대 해석하지 않는다. Ubuntu/CUDA build와 calibration 파일이 필요하고 [공식 환경](https://raw.githubusercontent.com/princeton-vl/DPVO/main/environment.yml)은 Python3.10/PyTorch2.3.1/CUDA12.1을 지정한다. 현재 Windows/Python3.12 환경에 바로 추가 가능한 자산으로 보지 않는다. [MIT 원문](https://raw.githubusercontent.com/princeton-vl/DPVO/main/LICENSE)

### C. TimeChat — CVPR 2024

프레임에 시각을 연결하는 encoder와 sliding video Q-Former를 사용하고 temporal grounding 등의 작업을 평가한다. [CVF 논문](https://openaccess.thecvf.com/content/CVPR2024/html/Ren_TimeChat_A_Time-sensitive_Multimodal_Large_Language_Model_for_Long_Video_CVPR_2024_paper.html)

**적용 추론:** Stage2의 원본 번호와 시간 위치를 명시하고 전체 문맥과 짧은 연속 구간을 일관되게 연결하는 원칙이 유효하다. 현재 영상 옆에 시간 문자열을 붙이는 것만으로 논문의 학습된 timestamp-aware encoder를 재현했다고 주장하지 않는다. TimeChat7B는 EVA/BLIP Q-Former/Video-LLaMA/LLaMA2 등 여러 자산이 필요하다. 공식 checkpoint 페이지는 공개지만 관련 모델별 조건과 총 메모리·설치를 별도로 확인해야 한다. 8GB에서 검증된 기존4B를 곧바로 대체할 근거가 없다. [공식 코드·모델 구성](https://github.com/RenShuhuai-Andy/TimeChat)

### 보조: On Calibration of Modern Neural Networks — ICML 2017

Temperature scaling은 고정 모델의 logits를 보정하며 별도 검증 집합이 필요하다. 양의 단일 temperature는 argmax를 바꾸지 않으므로 자체로 분류 정확도를 올리지 않는다. 논문은 train/validation/test의 같은 분포를 가정한다. [논문](https://proceedings.mlr.press/v70/guo17a/guo17a.pdf)

**제한:** 모델이 말한 “확신”, 반복 답변 일치도, optical-flow 점수를 정답 확률로 취급하지 않는다. AI 생성 라벨을 정답 삼아 같은 모델을 calibration하는 것은 독립 정답 검증이 아니다. 3개 접촉 참조로 temperature나 confidence 임계값을 맞추고 신뢰도 보정 성공을 주장하지 않는다.

## 3. Stage3: 실제 구현 가능한 단일 구조 후보

우선 후보는 **현재 광류 계산을 보존하는 작은 시간 convolution 보정 head**다. DPVO 전체 편입보다 취득/설치 비용이 낮고 기존 입력으로 시험할 수 있다. 이는 논문의 SlowFast 구현이 아니라 시간 순서를 학습한다는 원칙을 이용한 별도 제안이다.

- 입력: 이미 계산하는 raw144의 10Hz 순서, 중심 앞뒤 15개를 포함한 31개 창. 기존 864/head는 기준 출력으로 그대로 계산한다.
- 구조 사전 고정 제안: 144→32 projection, kernel3/dilation1·2·4·8의 32채널 Conv1d 네 층, 두 head 합계 7개 logit 보정. 편향을 포함하면 약 1.7만 파라미터 규모의 설계다. 이것은 새 학습이 필요하며 새 사전학습 가중치 다운로드는 필요하지 않다.
- 출력: 기존 두 classifier 확률의 log에 보정값을 더한다. zero 초기 출력으로 기준 모델에서 출발한다. 각 head에 기존 클래스 순서와 GT STOPPED 조향 제외를 유지한다. 이 log 확률을 calibration된 확률이라고 부르지 않는다.
- 학습: seed42·창31·폭32를 결과 전에 고정, 기존 외부 proxy 및 공개 GT의 행/출처 가중을 기준과 동일하게 둔다. 모델을 추가하는 효과와 시간 표현 변경이 함께 있으므로 순수 시간 인과효과로 표현하지 않는다. 효과가 관측될 때만 별도 순서 교란 진단으로 시간 순서 의존을 확인한다; 이 진단으로 후보를 다시 고르지 않는다.
- 평가: 먼저 기존 RAV4→Civic 고정 stress와 공개 그룹 OOF의 **두 head**를 기준과 비교한다. 이 분할과 공개 OOF가 이미 개발에 반복 사용됐다는 사실을 표시한다. 이전 split을 “새 독립 검증”으로 이름만 바꾸지 않는다. 같은 원천과 겹치는 영상은 전체 route를 훈련에서 제외한다.
- 로컬 연구 통과 제안: 두 head Macro-F1 무하락, S3 증가, 기존 12개 Civic 중 7개 이상 route 개선. 기존 Stage3 시간 기준과 같은 CPU 실제 paired median ≤1.25를 별도 확인한다. 이 기준을 통과해도 공식 정의의 독립 GT가 확보되지 않은 일반화 주장은 보류한다. 부족한 증거를 이유로 조용히 임계값을 바꾸지 않는다.
- 배포 계약: 창은 한 영상 안에서만 생성하고 경계 padding을 고정한다. 모든 decoded frame에 2개 라벨을 반환한다. 긴 영상은 halo15가 겹치는 chunk로 처리하고 전체 처리와 동등성을 확인한다. 파일 간 hidden state·점수 분포·온라인 파라미터 갱신은 없다.

실패 이력과 구분: 이전 GT-only, 연속 CAN 회귀→transfer, 시간 평활화, 단순 양의 배율 증강, 출처1:1 가중은 이미 검증 근거가 부족하거나 실패했다. 1128차원 global/IRLS 잔차 후보는 RAV4→Civic S3가 0.565342→0.564394로 하락했다. 이를 새 유망 후보처럼 반복하지 않는다. 이번 작은 시간 head도 proxy가 잘못된 경우 그 오류를 더 잘 학습할 위험이 있다. 근거: `research/v5_stage3/equivalent_optimization/summary.md`, `research/v4_stage3/summary.md`.

**DPVO 편입은 후순위 조건부 연구:** 실제 모델 권리와 Linux 오프라인 build가 확인되고, 알려진 calibration이 있는 외부 구간에서 pose 실패율/자원 진단이 통과할 때만 소수 pose 보조값을 시험한다. 평가 카메라 intrinsics가 없는 상태에서 임의 intrinsics를 정답처럼 쓰거나, pose를 m/s²로 변환해 공식 비공개 임계값을 추정하지 않는다. DPVO와 VLM의 동시 상주는 로컬 8GB 계획에 넣지 않는다. 장기 학습부터 시작하는 것은 이번 제한에 맞지 않는다.

## 4. Stage2: 지역 확대와 전체 문맥을 함께 남기는 두 단계 추론

제안은 모델 크기 교체보다 **최종 판단에 필요한 상대 차량과 주변 공간을 동일 시각의 전체/지역 영상으로 연결**하는 것이다. 이전 contact-sheet 대 multi-image 변경과 접촉 검증 추가 호출 실패는 보존하고, 단순 presentation 재탐색을 반복하지 않는다. 시간 범위는 기존 후보에서 결과를 보고 GT 쪽으로 옮기지 않는다.

1. 첫 단계는 전체 장면을 보며 후보 시간과 상대의 시각적 위치를 제안한다. 생성된 box는 탐색 단서이며 정답이 아니다. 유효하지 않은 box는 원래 전체 프레임으로 돌아간다.
2. 두 번째 단계는 고정된 후보 시간의 전체 프레임과 확장 crop을 함께 제공한다. crop만으로 차선·공간·진입방향을 판정하지 않는다. 전체/지역에는 같은 원본 번호를 붙이고 동일 시점을 서로 다른 시간으로 오해하지 않게 한다.
3. 접촉 결정 이후 진입·공간 문맥을 다시 연결한다. 전체 4개 기존 호출 안에서 역할을 배치하는 것을 우선하며 무조건 8회 “두 번 생각하기”로 늘리지 않는다. 최종 contact 수정에 맞추지 않은 entry/space를 그대로 둔 채 일관성이 해결됐다고 주장하지 않는다.

ROI 실패 위험: 상대 오인, 작은 box, 가림, 확대 보간으로 접촉이 발생한 것처럼 보이는 현상, 기준점 이동, 전체 맥락 축소가 있다. VLM box를 human GT로 저장하거나 그 box를 그대로 독립 검증 정답에 쓰지 않는다. 후보 회수율과 후보 중 선택 성공률을 별도로 보고, 선택 모델이 실패하면 후보를 무조건 늘리는 방향으로 반복하지 않는다.

## 5. 모든 Stage에 공통인 채택·누수 방지 기준

| 단계 | 데이터/지표 | 진행 조건 | 중단 또는 제한 |
|---|---|---|---|
| Stage1 | 실제 주행 원본↔화면 재촬영 대응쌍, source/display/camera holdout, 두 클래스 F1 | 실제 대응 검증에서 근거 확보 후 새 forensic 조합 평가 | 현재 실제 주행 대응쌍0. 문서/합성 점수만으로 재촬영 일반화 성공 주장 금지 |
| Stage2 | 새 미사용 원천의 실제 사람 contact/entry/side/space; unknown 유지 | 변경한 항목의 개선 및 다른 알려진 항목 비하락, 원본 PTS 오차·실패 사례 보고 | contact-only3개가 전체 S2를 검증하지 못함. 이미 사용한9개는 개발 전용 |
| Stage3 | 공개50 그룹 OOF+외부 차량/route stress; stop 마스크 포함 두 head | 사전 기준을 통과한 단일 후보만 독립 target GT 확보 후 재검토 | CAN 일치 공개10행이 CONSTANT뿐. proxy 검증을 공식 label 검증과 합치지 않음 |

새 Stage2 검증에서 최소한 기존 기준의 +1개 정답·기존 정답 손실0·MAE 비증가를 유지하되, 3개만으로 통계적으로 안정한 개선이라고 표현하지 않는다. 이번에 entry/side/space까지 바꾸면 각 항목의 실제 known GT가 필요하고 알려지지 않은 항목은 종합 S2 계산에서 임의 보충하지 않는다. 별도 원천에서 표본과 기준을 먼저 동결하며 결과를 본 뒤 영상 교체·다른 모델 재시도는 새 실험으로 기록한다.

AI 라벨은 학습용 약한 감독 또는 후보 설명으로만 구분하고, 평가·채택 기준의 정답에는 사용하지 않는다. 사용자가 원본을 보고 작성한 라벨도 단일 검수 초안임을 유지한다. 가능하면 모델 답을 보지 않은 별도 검수와 불일치 조정을 실시한다. 부재 시 다중 검수 완료라고 적지 않는다. source SHA·원본 decoded index/native PTS·주석 SHA·분할·코드/가중치 SHA를 연결한다.

## 6. 자원 예산과 실제 출력 검증

현재 로컬 GPU는 RTX2080 SUPER 8GB, 대회 서버는 L40S 44.7GiB/7vCPU/60GB RAM이고 **모든 Stage 합산 60분**이다(`대회_통합_정보.md:265–275`). 큰 서버 VRAM이 더 많은 호출 시간을 보장하지 않는다. 다음은 연구용 상한 제안이다.

- Stage2: 한 모델 인스턴스, batch1, 기본 4회 호출을 우선 유지. 각 호출의 총 시각 입력 예산은 기존 `pixel_budget=1,200,000`을 넘기지 않게 동결한다. crop 추가로 원본 문맥이 얼마나 축소되는지도 기록한다. `solution/vlm.py:26–45`는 이미지 수에 따라 예산을 나누므로 crop 수 증가가 무료가 아니다. 생성 토큰 상한·processor 토큰 수·prefill/decode 시간·peak allocated/reserved를 각각 측정한다. 추가 재검토를 넣을 경우 최악의 모든 파일이 재검토되는 조건으로 예산을 산정한다.
- Stage3: 새 temporal head는 CPU2, 기존 광류 캐시 재사용. 연구 1회 학습의 운영 상한을 30분으로 제안하며 이는 예상 완료시간이 아니다. 초과 시 부분 결과로 채택하지 않는다. 추론은 공개5개와 긴 외부영상에서 기준/후보를 교차 순서로 각3회 비교한다. DPVO 연구는 별도 GPU 세션에서만 진행하고 Stage2와 동시 상주하지 않는다.
- Stage1: 원본의 JPEG/주파수 단서는 공용 리사이즈·안정화 단계 전에 보존한다. Stage2 확대/광류 전처리를 Stage1에 무조건 공유하지 않는다. 원천별 I/O를 공유하더라도 Stage별 필요한 해상도·색/주파수 정보 손실을 검증한다.
- 전체: 같은 공개 입력에서 기존 V6와 후보 ZIP을 실제 추출해 순차 실행한다. 해시·출력 계약·단일 파일/파일명 변경/다른 파일 추가의 독립성·읽기 전용 모델·네트워크0·모델 해제 후 메모리를 확인한다. 지역 처리 속도비나 호출 수만으로 서버60분 통과를 보장하지 않는다. 평가 영상 수·길이가 비공개라 최악 처리량 보장은 확인 불가다.

최종 판단은 전체 Stage 목표에 맞춰야 한다. Stage1 또는 Stage3에 검증된 개선이 없으면 무근거 변경을 넣지 않되, 그 Stage의 개선 완료라고도 쓰지 않는다. 정확도 후보가 모두 탈락한 뒤 계산 동등 버전만 제출하면서 정확도 개선 회차가 완료됐다고 보고했던 V5의 한계를 반복하지 않는다.
