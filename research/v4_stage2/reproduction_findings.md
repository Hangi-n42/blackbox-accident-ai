# 실제 ZIP 결과 불일치 감사 및 동일 입력 재현

기존 `D_run`과 실제 ZIP 전체3stage 검증에서 S2_002가 달랐다. 기존 결과를 허용하거나 CSV를 수정하여 통과시키지 않았다. root가 `research/v4_validation_scope_corrected.json`을 추가 재현 결과 전에 동결했고, 실제 패키지의 동일 코드·모델과 실제 검증 입력으로 V3/D를 독립 실행했다.

## 확인된 입력 차이

기존 D 실험은 `research/stage2_public/images`, 실제 ZIP 검증은 `artifacts/public_eval/stage2/images`를 사용했다. 같은 원본 영상·번호·크기였지만 **250개 전부 decoded RGB가 달랐다.** 파일 SHA와 JPEG 양자화표도 달랐다. 영상별 RGB 절대차 평균은 8bit값 기준0.452~0.644였다.

- `research/stage2_probe.py:26`: OpenCV로 해독한 RGB를 PIL.Image.save로 JPEG 저장한다.
- `scripts/prepare_public_eval.py:28`: OpenCV.imencode로 JPEG 저장한다.
- `scripts/evaluate_stage2_v4_d.py:47`: 연구용 폴더를 고정 사용했다.
- `scripts/verify_submission.py:32`: 지정된 data/stage2를 실제 패키지에 전달한다.
- 프레임별 file/RGB SHA·크기·양자화표 비교와 절대차: `input_representation_audit.json`.

따라서 기존의 정확일치 gate는 서로 다른 픽셀을 같은 입력으로 취급한 잘못된 비교였다. 같은 번호 또는 같은 원본 MP4만으로 같은 모델 입력임을 보장할 수 없다.

## logger 정적 감사

`scripts/evaluate_stage2_v4.py:57`은 입력 이미지 PNG를 저장하고,59행의 bounded 크기는 기록용 계산이다. 실제 images를 resize하거나 교체하지 않고70행에서 원래 images/prompt/max_new_tokens를 그대로 model.ask에 전달한다. 실제 resize는 frozen `solution/vlm.py` 내부에서 수행한다. 동기화와 peak-memory 기록 외에 모델 가중치/질문/출력을 변경하는 코드는 발견하지 못했다.

기존 평가 helper를 import하면 workspace solution이 먼저 로드될 수 있으므로 재현에는 그 helper를 사용하지 않았다. 새 `scripts/reproduce_stage2_v4_package.py`는 독립 recorder를 사용하고, 패키지 code 경로를 우선 추가한 뒤 solution 모듈을 읽었다. 시작 및 종료에 solution.__file__들이 추출 ZIP 아래에 있는지 확인했고 sys.dont_write_bytecode로 패키지에 pycache를 추가하지 않았다. 실행은 -I 없이 시작했지만 실제 로드 경계를 assert했다. root의 최초 전체3stage 실행과 이번 Stage2 단독 실행 사이에는 실행 이력 차이가 있으므로, 이를 무시하고 수치 결정론을 일반적으로 주장하지 않는다.

## 동일 실제 입력의 재현 결과

패키지: `artifacts/submissions/verify_v4`. 입력: `artifacts/public_eval/stage2/images`. 같은 저장 NF4 모델로 V3전체5개 → D전체5개, 총40질문만 실행했다. 새 프롬프트·모델 변경·학습은 없다.

**새 `repro_run/D_predictions.csv`는 기존 실제 ZIP 전체3stage 실행의 `artifacts/submissions/verify_v4_results/stage2.csv`와5행 정확히 일치했다.** 원본 모델 및 패키지 source SHA도 전후 동일이고 네트워크 시도는0이다. 세션33577 exit0을 확인하고 GPU를 해제했다. 기존 D_run, 최초 실패 결과, 원래 선택 CSV는 보존했다.

| ID | 동일 실제입력 V3 진입/방향 | 동일 실제입력 D 진입/방향 | 변화 |
|---|---|---|---|
| S2_001 | 24 / LEFT | 24 / RIGHT | 방향 |
| S2_002 | 36 / LEFT | 4 / RIGHT | 진입·방향 |
| S2_003 | 23 / LEFT | 23 / LEFT | 없음 |
| S2_004 | 4 / RIGHT | 4 / RIGHT | 없음 |
| S2_005 | 23 / RIGHT | 23 / RIGHT | 없음 |

충돌과 공간은 모든 쌍에서 동일했다. 기존 연구 JPEG에서 관찰한 “D가 방향 하나만 바꾸고 후속 질문/출력이 전부 같다”는 결과는 **그 JPEG 입력에 한정**된다. 실제 검증 RGB에서는 S2_002의 진입도 달라진다.

S2_002의 차이는 모델 원응답으로 추적했다:

- 연구 JPEG D: overview collision44/LEFT → fine collision44 → entry36 → 최종 motion collision47.
- 실제 검증 JPEG D: overview collision38/RIGHT → fine collision45 → entry4 → 최종 motion collision47.

첫 질문의 문구와 원본 프레임 번호는 동일하지만 이미지 RGB 해시는 다르다. 최초 판단 차이가 후속 충돌 후보와 진입 질문의 시간 구간으로 전파됐다. 이번 불일치는 같은 입력의 비결정성으로 단정할 근거가 없으며, 동일 실제입력에서 정확히 재현됐다. 확인한 한계는 JPEG 표현 차이에 대한 예측 민감성이다. 경고를 제거하기 위해 tokenizer 설정을 바꾸지도 않았다.

## 선택과 검증의 한계

S2_001의 RIGHT는 root/별도 reviewer의 AI 시각 관찰과 일치하지만 공식 방향 GT가 아니다. S2_003 LEFT는 미해결이다. S2_002의 원영상 시작 방향은 불확실하고 진입4도 공식 정답으로 확인되지 않았으므로 새 변화를 정확도 향상으로 계산하지 않는다. 원본 접촉 라벨의 시각적 의미 불확실성도 그대로다.

수정된 gate는 실제 동일 입력의 독립 실행 CSV를 대상으로 정확일치를 요구한다. 이것은 입력 mismatch를 단순 허용한 것이 아니며, 일반화 성능을 입증하는 것도 아니다. 최종 제출 결정은 이 민감성과 하위 GT 부재를 포함하여 root가 판단한다.

근거: `repro_run/report.json`, `repro_run/{V3,D}/{ID}/call_*/result.json`과 PNG, `repro_run/input_effect_summary.json`, `repro_run/V3_predictions.csv`, `repro_run/D_predictions.csv`.
