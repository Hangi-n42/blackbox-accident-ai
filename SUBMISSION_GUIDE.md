> 2026-09-13 갱신: V4 실제 제출과 공식 채점을 완료했다. 접수 88503, 03:18:26 KST, 파일 `submit_v4.zip`, 서버 소요 44분 33초. V4 Stage2는 0.2175214760으로 V3보다 하락했고 Stage1/3은 같다. 계산 가중합은 V4 0.39923211794, 현재 네 제출 중 최고는 V3 0.40347580682다. 공식 Private는 아직 없다. 아래 V3 승인 대기 서술은 09-11 당시 이력이다. V4 상세는 `research/v4_execution_20260913.md`, 결과는 `artifacts/submissions/submission_status_v4.json` 참조.

# 제출 모델과 검증 범위

## 목표와 평가

대회는 재촬영 판별(Stage 1), 사고 장면 분석(Stage 2), 프레임별 주행 상태 분류(Stage 3)를 평가한다. 종합 가중치는 각각 20%, 40%, 40%다. 따라서 가감속 분류가 전체의 28%, 충돌·진입 시점이 합계 28%를 차지한다.

공식 Public 대표 점수는 Stage 1 기준이다. 종료 후 각 제출물의 종합점수로 Private를 재계산하며, 서로 다른 제출물의 Stage 최고점을 합치는 방식이 아니다. 현재 로컬 검증 수치를 Private 점수로 해석하면 안 된다.

## 실제 접수한 모델

| 제출 ID | 파일 | Stage 2 모델 | 접수 시각(KST) |
|---|---|---|---|
| 87570 | `artifacts/submissions/submit_v1.zip` | Qwen3-VL-2B-Instruct, FP16 | 2026-09-11 19:24:16 |
| 87584 | `artifacts/submissions/submit_v2_nf4.zip` | Qwen3-VL-4B-Instruct, NF4 저장본·FP16 연산 | 2026-09-11 19:58:12 |
| 87666 | `artifacts/submissions/submit_v3_motion_fast.zip` | 같은 4B NF4, 최종 충돌 motion 대조 | 2026-09-11 22:44:54 |
| 88503 | `artifacts/submissions/submit_v4.zip` | 같은 4B NF4, V3 첫 질문의 출발 방향 정의 명시 | 2026-09-13 03:18:26 |

공식 점수와 최신 상태는 `artifacts/submissions/submission_status*.json`에 기록한다. 기존 ZIP은 덮어쓰지 않는다. ZIP별 manifest에는 전체 포함 파일의 크기와 SHA256이 있다. V4의 실제 추출본 오프라인 실행과 46개 파일 무결성 검사를 통과했으며, 모델 가중치는 V3와 같다.

### 확인된 공식 채점 결과

V1/V2는 2026-09-11 제출 목록에서, V3는 사용자 제공 이미지와 2026-09-13 실제 목록에서 완료를 확인했다. V4는 2026-09-13 05:28:58 KST에 점수 표시를 처음 확인했고 05:29:07 KST에 전체 행을 대조했다.

| 제출 ID | Stage 1 | Stage 2 | Stage 3 | 가중합 계산값 | 서버 소요 시간 |
|---|---:|---:|---:|---:|---|
| 87570 | 0.5078837545 | 0.1935425389 | 0.5266169416 | 0.3896405431 | 41분 41초 |
| 87584 | 0.5078837545 | 0.2204664646 | 0.5266169416 | 0.40041011338 | 46분 6초 |
| 87666 | 0.5078837545 | 0.2281306982 | 0.5266169416 | 0.40347580682 | 42분 38초 |
| 88503 | 0.5078837545 | 0.2175214760 | 0.5266169416 | 0.39923211794 | 44분 33초 |

가중합은 화면의 Stage 점수에 0.2/0.4/0.4를 곱한 계산값이며 공식 Private가 아니다. 확인된 네 제출 중 V3가 가장 높다. V4는 V3 대비 Stage2가 0.0106092222, 계산 가중합이 0.00424368888 하락했다. 하락을 만든 Stage2 하위 항목은 이 화면만으로 확인할 수 없다. `scripts/compare_submissions.py`는 완전한 제출 단위로 비교하며, 미표시 점수를 0으로 대체하거나 서로 다른 제출의 Stage 최고점을 조합하지 않는다.

Stage 1은 고정 TPO/CLIP 재촬영 판별기와 공개 예제에서 학습한 주파수 특징 분류기를 결합한다. 가중치 병합 전후 480프레임의 최대 확률 차이는 1.49e-6이었고 영상별 판정은 모두 유지됐다.

Stage 2는 각 입력 영상에서 움직임 후보를 찾고, 동결 시각언어 모델에 충돌·방향, 충돌 정밀화, 진입 시점, 회피 공간을 네 번 질문한다. 원본 파일명의 프레임 번호를 반환한다. 두 제출은 이 질문과 프레임 선택 정책을 공유한다.

Stage 3는 각 영상의 optical flow와 시간 특징 864차원을 계산하고 고정 Logistic Regression(가감속), RandomForest(조향)로 모든 디코딩 프레임에 라벨을 반환한다. 공개 5영상·희소 라벨 50행과 comma2k19 외부 23개 구간의 2,761표본을 사용했다. 확보한 외부 24개 중 공개 영상과 일치한 1개는 외부 학습에서 제외했다.

## 검증에서 확인한 것

- 두 ZIP 모두 압축 무결성, 최상위 제출 구조, 필요한 모델·코드 포함 검사를 통과했다.
- 압축 해제본만 사용하는 격리 Python 실행에서 네트워크 연결을 차단하고 세 함수를 순서대로 실행했다. Stage 1 10행, Stage 2 5행, Stage 3 2,998행의 규격 검사를 통과했다.
- 4B NF4 저장본은 별도 프로세스에서 재로딩한 뒤 공개 5영상의 최종 예측과 20회 모델 응답이 저장 전과 같았다. 네트워크 연결 시도는 없었다.
- V2 ZIP의 Stage 1·3 결과는 V1과 완전히 같았고 Stage 2는 검증된 4B 후보 결과와 같았다.
- 공개 충돌 정답에 대한 ±3프레임 검증은 최종 2B 3/5, 4B NF4 4/5였다. 공개 예제는 10FPS라서 이 비교에만 ±3프레임을 사용했다. 실제 대회는 영상별 시간 대응으로 ±0.3초를 채점한다.
- Stage 3 공개 영상별 OOF 개발 점수는 0.798298이다. 별도 시간 평활화 실험은 동일 조건에서 예측·점수가 개선되지 않아 채택하지 않았다.

## 성능 판단의 한계

### 추가 외부 진단과 미채택 가설

DLC-2021 배포물의 CC BY-SA 2.5 라이선스를 확인하고, 결과를 보기 전에 문서 식별자 해시 순서로 6개 문서를 선택했다. 각 문서의 원본/화면 재촬영 및 두 카메라를 균형화한 24개 원영상에서 배포 JPEG를 8장씩 선택 취득했다. 총 192장, JPEG payload 50.316MB이며 CRC/SHA와 출처·이용조건을 보존했다.

고정 임계값 0.5에서 TPO 단독 Macro-F1은 0.788360, forensic 단독은 0.495798, 기존 0.5 혼합은 0.733333이었다. 이는 원영상별 이미지 부분집합에 대한 문서 도메인 진단이며, 전체 영상 디코딩 평가나 블랙박스 대회 성능이 아니다. 원본은 FHD/4K가 섞였고 재촬영은 전부 4K인 해상도 차이도 있다. TPO는 인쇄 공격도 학습했으므로 물리적 문서 원본과의 의미 차이를 고려해야 한다. 이 결과만으로 모델·임계값·혼합 비율을 변경하지 않았다. 근거: `research/stage1/dlc_frozen_diagnostic_report.md`.

이어 이미 취득한 comma2k19 원본 주행영상 23구간에서 영상당 12프레임을 검사했다. TPO 단독의 원본 오탐은 4/23(17.39%), forensic 및 혼합 모델은 각각 0/23이었다. HEVC 메타데이터 오류 때문에 실제 순차 디코딩 개수로 균등 프레임을 선택했다. 원본만 있는 이 검사로 재촬영 검출률이나 전체 Macro-F1을 계산하지 않았고 DLC와 합친 점수도 만들지 않았다. 두 차량의 제한된 주행 조건이라는 한계가 있다. 문서 재촬영의 이득과 주행 원본 오탐의 손실이 함께 나타나 Stage 1은 기존 혼합 모델을 유지한다. 이는 외부 진단을 활용한 개발 판단이며 대회에서의 우위를 입증하지 않는다. 근거: `research/stage1/comma_original_diagnostic/metrics.json`.

Stage 2의 진입 시점은 기존 12개 후보가 긴 구간의 모든 가능한 정답 시점을 충분히 촘촘하게 포함하지 못하는 구조적 한계를 확인했다. 별도 `solution/stage2_entry_refine.py`에서 최대 한 번의 국소 추가 질문을 구현했고 CPU 가짜 모델 계약 8개를 통과했다. 실제 진입 정답과 정확도 근거가 없어 제출에 적용하지 않았다. 국소 구간이 틀린 경우와 충돌 상대 식별 문맥이 부족한 경우를 해결했다는 주장도 하지 않는다. 근거: `research/stage2_entry_refine_candidate.md`.

Stage 3의 특징 배율 증강은 고정 배율 0.75/1/1.25와 기존 외부 17/6 주행 분할로 별도 실험했다. 표본 수 증가의 영향을 분리하기 위해 같은 데이터를 3회 복제한 대조군을 결과 확인 전에 추가했다. 외부 개발 검증 복합점수는 비증강 0.604002, 단순복제 0.611355, 배율증강 0.618559였다. 외부 점수는 양쪽 대조군을 넘었지만 공개 영상별 OOF는 비증강 0.759552, 단순복제와 증강 모두 0.741332로 증강이 하락했다. 사전에 고정한 최종 재학습 조건을 충족하지 못해 미채택했으며, 최종 23경로 재학습·체크포인트 생성·제출을 하지 않았다. 대회 CAN 임계값과 외부 라벨 임계값의 동일성은 확인되지 않았다. 근거: `research/stage3_scale_augmentation/external_validation.json` 및 사전 고정 설정·추가 대조군 기록.

공개 예제는 본 학습 데이터셋이 아니다. Stage 1 원천 장면은 5개이며 재녹화 예제도 실제 기기 재촬영이 아닌 모사본이다. Stage 2에는 진입·방향·회피 공간 정답이 없다. Stage 3 정답은 희소하다.

공개 5영상을 반복 검토하고 모델을 선택했으므로 개발 지표에는 선택 편향이 있다. 외부 자료의 모든 원천 주행 중복까지 배제했다고 주장하지 않는다. 외부 CAN 범주화 임계값도 공식 대회 정답 임계값과 같다고 가정하지 않는다. 미공개 평가 데이터의 개수와 길이를 모르므로 로컬 실행시간만으로 서버 60분 통과를 보장할 수 없다.

## 규칙과 출처

모든 추론은 파일별로 독립 수행하고, 다른 평가 파일의 정보·예측·통계를 사용하지 않는다. 평가 중 추가 학습·튜닝·의사 라벨 학습·모델 갱신·외부 API 호출을 하지 않는다. 공개 자료에 대한 학습 및 후보 검증과 서버의 비공개 평가를 구분한다.

- 대회 규칙·출력·환경: `대회_통합_정보.md`, [공식 규칙](https://dacon.io/competitions/official/236753/overview/rules), [공식 평가](https://dacon.io/competitions/official/236753/overview/evaluation)
- TPO 가중치: [공식 TPO](https://github.com/gurayozgur/TPO), CC BY-NC-SA 4.0. 병합 변환과 출처·라이선스 고지를 함께 포함했다.
- CLIP 구현·원본 가중치: [OpenAI CLIP](https://github.com/openai/CLIP), MIT. 제출 추론에는 필요한 시각 모델 구현만 포함했다.
- Stage 2: [공식 2B](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct), [공식 4B](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct), Apache-2.0. 고정 revision과 원본 SHA를 기록했다. NF4 저장본은 직접 변환한 파생본임을 고지했다.
- 외부 주행 자료: [commaai/comma2k19](https://huggingface.co/datasets/commaai/comma2k19), MIT. 취득 목록·해시·중복 제외·실제 학습 범위를 보존했다.

## 파일 안내

### 현재 Stage 3 학습·선택 경로

현재 제출 체크포인트는 `research/train_stage3.py --external`의 실험에서 선택된 Logistic 가감속 분류기와 RandomForest 조향 분류기를 사용한다. 그 실험의 `solution/model/stage3/motion_model_external.joblib`에서 두 분류기를 골라 `research/select_stage3.py`가 현재 `model/stage3/motion_model.joblib`를 만들었다. 선택 근거와 해시는 `research/stage3_selection.json`에 있다. 공개 OOF를 여러 후보 선택에 사용했으므로 독립 최종 검증 점수는 아니다.

`research/finalize_stage3.py`는 앞선 transfer-head 실험용이며 현재 제출 모델의 재현 진입점이 아니다. 이 과거 스크립트는 현재 모델 경로를 덮어쓰므로, 이번 개선 작업에서는 실행하지 않는다. 선택 스크립트도 출력 경로를 덮어쓰는 기존 연구 도구이므로 새 후보는 별도 경로에 만들고 이미 접수한 ZIP을 유지한다.

### 미제출 속도 개선 후보

`inference_nf4_fast.py`와 `solution/stage3_fast.py`는 V2와 같은 모델을 사용하되 Stage 3의 ROI별 통계 계산을 묶는다. 좌표 캐시는 한 영상의 함수 호출 내부에만 존재한다. `stage3_fast.py`가 원본 `stage3.py`의 상수와 보조 함수를 사용하므로 둘 다 패키지에 포함한다.

공개 10Hz 영상 2,998행과 외부 5분 반복 입력 3,000행에서 최종 특징 5,182,272개가 비트 단위로 일치했고 모든 예측이 같았다. 로컬 전체 시간은 각각 48.73→23.51초, 48.16→23.07초였다. 외부 입력은 기존 1분 영상의 5회 반복이며 독립적인 정확도 평가가 아니다. 전체 측정은 각 1회이므로 서버 시간 감소율을 보장하지 않는다. 근거는 `research/stage3_fast_benchmark/report.json`이다.

`scripts/build_submission.py --variant nf4_fast`는 별도 `artifacts/submissions/candidate_nf4_fast.zip`을 생성한다. 이미 제출한 두 ZIP은 보존한다. 후보 ZIP의 실제 격리 실행 검증은 별도 결과 파일로 기록하며, 후보 생성 자체는 대회 접수를 의미하지 않는다.

후보 ZIP SHA256은 `d6289c200e34eb36125fbeb64626611146735bdee9ee6b9a00ca1c53c4bd34ff`이며, 크기는 2,914,581,099바이트다. 압축 해제본으로 네트워크 차단·격리 실행 검증을 통과했다. Stage 1/2/3 각각 10/5/2,998행이며 세 Stage의 CSV가 모두 V2와 동일했다. 이 순차 실행에서 Stage 3는 33.01초였다. 앞의 단독 벤치마크 23.51초와 측정 조건이 다르므로 혼용하지 않는다. V2와 ZIP manifest를 비교했을 때 추가 파일은 `stage3_fast.py` 하나, 변경 파일은 `inference.py` 하나이며 나머지 파일 해시는 모두 같았다. 근거: `artifacts/submissions/verify_nf4_fast_results/report.json`.

- V1 소스 진입점: `inference.py`, `requirements.txt`
- V2 소스 진입점: `inference_nf4.py`, `requirements_nf4.txt` (ZIP에서는 각각 `inference.py`, `requirements.txt`로 배치)
- 구현: `solution/`
- 빌드: `scripts/build_submission.py --variant v1` 또는 `--variant nf4` (기존 ZIP 덮어쓰기 거부)
- 실제 ZIP 실행 검증: `scripts/verify_submission.py`
- 4B 비교·재로딩 근거: `research/stage2_4b_validation.md`
- Stage 3 선택·평활화 근거: `research/stage3_selection.json`, `research/stage3_temporal_v1/summary.md`
- 전체 작업 기록: `WORK_LOG.md`

환경은 이 프로젝트의 `.venv`다. 서버 기본 패키지를 유지하며 V2만 `bitsandbytes==0.48.1`을 추가한다. 원본 데이터와 이미 접수된 ZIP을 보존한다.


### 세 번째 충돌 시점 대조 후보

`submit_v3_motion_fast.zip`은 Stage 2의 최종 충돌 프레임만 기존 움직임 점수의 최댓값에 해당하는 원본 프레임 번호로 교체한다. 기존 4B NF4 네 호출과 다른 세 항목을 보존한다. Stage 1·3 가중치도 동일하며 Stage 3는 검증된 계산 최적화를 적용한다. 배율 증강 체크포인트는 포함하지 않는다.

실제 ZIP의 오프라인 실행은 Stage 1 10행, Stage 2 5행, Stage 3 2,998행에서 출력 계약을 통과했다. Stage 1·3의 모든 출력과 Stage 2의 진입·방향·회피 공간 출력이 V2와 동일했다. 파일별 SHA 비교에서 기존 가중치·공통 소스는 모두 같고, 진입점 변경과 두 새 모듈 추가만 확인했다. 공개 충돌 예제는 두 방법 모두 4/5이며 일반화 개선 근거는 아니다. 최대 움직임과 최초 접촉의 불일치, 다른 항목의 기존 VLM 문맥 유지가 한계다.

SHA256: `2a955dc4681d5817b33835af5136a1551b406a1ab11a668e804bc88677513c4c`. 크기 2,914,582,277 bytes, 압축 해제 3,424,365,690 bytes. 검증 기록: `artifacts/submissions/verify_v3_motion_fast_results/report.json`, `ablation_comparison.json`. 접수·공식 점수 상태는 `artifacts/submissions/submission_status_v3.json`을 따른다.

V3의 초기 최종 접수 시도는 자동 승인 검토에서 거부됐으나, 이후 제출87666으로 정상 접수·평가가 완료됐다. 현재 확인된 공식 표시 점수는 Stage1 0.5078837545, Stage2 0.2281306982, Stage3 0.5266169416이며 실행42분38초다. 과거 승인 대기 상태를 현재 상태로 해석하지 않는다.

### V5 검증·제출 완료, 채점 준비

V4의 Stage2 변경을 제거해 V3 정책을 복원한다. Stage1 디코딩 오류 복구를 유지하고, Stage3는 기존864 특징·가중치·출력을 보존한 계산 최적화만 적용한다. 신규 정확도 후보는 모두 검증 실패로 제외했으며, 라벨 부족 및 일반화 문제 전체가 해결됐다는 주장은 하지 않는다.

`submit_v5.zip`:46파일,2,914,587,768 bytes, 압축 해제3,424,379,625 bytes. SHA256 `e824f562d13c1887f47ecbbbbec59ab7baf5e7d0df2b1b565a99c8275c243322`. 실제 ZIP 오프라인 검증과 세 Stage의 V3 공개 입력 출력 완전 동일성 검사를 통과했다. Stage3 별도 동등성 실험에서는 공개5+외부23영상33,569행 일치, 로컬 전체 추론 시간 약19% 감소를 확인했다. 서버 시간이나 새로운 최고점 개선 보장은 아니다.

소스 진입점은 `inference_v5.py`, 빌드는 `scripts/build_submission.py --variant v5`다. `v5_selection_frozen.json`에 고정된 전체46파일 SHA를 강제하고 기존ZIP 덮어쓰기를 거부한다. 패키지 검증 기록은 `artifacts/submissions/verify_v5_results/package_audit.json`, 분석은 `research/v5_diagnosis_and_decisions.md`다. V5는 제출 번호88887,2026-09-13 20:14:05 KST로 실제 접수됐다. 확인 당시 채점 준비이며 공식 점수는 미정이다. 현재 접수 기록은 `artifacts/submissions/submission_status_v5.json`을 따른다.

## 2026-09-14 V5 공식 결과 및 사후 감사

사용자 제공 DACON 결과 이미지에서 V5 Stage1 0.5078837545, Stage2 0.2281306982, Stage3 0.5266169416,43분19초를 확인했다. 파일명·접수시각·SHA가 기존88887 기록과 일치한다. 이미지에는 제출번호가 잘려 있으므로 ID는 이전 실제 접수 관측에 근거한다. 새로운 웹 조회나 정확한 서버 채점 완료 시각은 기록하지 않았다.

세 점수와 계산 가중합0.40347580682는 V3와 같고, 시간은41초 길다. V4 Stage2 하락은 회복했으나 V3 대비 정확도 개선은 없다. 로컬 Stage3 약19% 단축을 서버 전체 개선으로 해석하지 않는다. 신규 후보 전원 기각 후 출력 보존/복원 버전으로 제출한 한계와 독립 정답 부족을 전문가 역할 AI 에이전트 3명과 재감사했다. 구현·접수 완료와 정확도 목표 미달을 구분한다. 종합 보고 research/v5_postmortem_20260914.md, 점수 기록 artifacts/submissions/submission_status_v5.json. 이번에는 분석과 기록 갱신만 했으며 모델 변경·학습·추가 제출은 없다.
