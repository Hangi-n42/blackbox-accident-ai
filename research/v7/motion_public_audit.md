# 공개 접촉 라벨과 V7 초기화 보정의 CPU 감사

2026-09-15. 결과는 **기존 공개 5개에 대한 공식 부분 라벨 대조**다. 새로운 독립 검증셋을 확보한 결과가 아니다. GPU·모델 호출·다운로드·프롬프트 수정은 수행하지 않았다. V6 모델과 제출 파일은 변경하지 않았다.

## 결론

원본·번호·시각·입력 픽셀 대응을 검증했으므로 이 5개에서 제공 접촉 라벨과 보정 결과를 대조할 수 있다. frozen V6 접촉 선택과 초기화 보정의 결과는 **모든 영상에서 동일하다. ±0.3초 일치 4/5, MAE 0.46초가 그대로다.** 초기화 보정에 유리한 추가 증거는 얻지 못했다.

이미 실패한 사람 검수 개발 9개 gate(접촉 4/9→4/9, gained=0)는 변경하지 않는다. 이 공개 5개는 이전 개발에서 반복 사용되었으므로 독립 holdout으로 인정하거나 합쳐 독립 14개라고 계산하지 않는다. 진입·방향·회피 공식 라벨은 모두 -1이므로 전체 S2는 계산할 수 없다.

## `32`는 무엇의 단위인가

- `Baseline/data/stage2/labels.csv:1`의 열 이름은 `t_collision`이며, 2~6행 값은 32/30/31/41/30이다. 열 이름만으로 초라고 해석하면 안 된다.
- `Baseline/[Baseline_Train]_3Stage_학습.ipynb:73`의 `_video_frames`는 `cap.read()`로 영상을 순차 해독한다. 같은 파일 175행은 `int(r.t_collision)`을 전체 프레임 시퀀스의 target index로 넣고, 이후 temporal collision logits에 대한 cross entropy target으로 사용한다. 즉 baseline의 실행 가능한 의미는 **배포된 영상의 0부터 시작하는 프레임 인덱스**다.
- `Baseline/[Baseline_Inference]_3Stage_추론및ZIP생성.ipynb:607`은 smoke용 영상 추출 번호를 0으로 시작하고, 612~613행에서 각 해독 프레임을 `frame_{frame_index:06d}.jpg`로 저장한 뒤 번호를 증가시킨다. 이는 제공 예제의 전체 프레임 입력 계약을 뒷받침한다. 비공개 영상의 FPS·JPEG 품질·원천 재표본화 방식을 증명하는 코드는 아니다.
- 이번에 5개 MP4를 모두 다시 해독했다. 각 50프레임, native time base 1/10240, PTS 간격 1024, 시각 0.0~4.9초, stream duration 5초, average rate 10이다. 따라서 라벨 32는 이 제공 클립에서 3.2초에 해당한다. **32초가 아니다.** 과거 설명에서 32/30 등을 초로 불렀다면 그 표현은 잘못이다. 기존 부분 GT 변환기 `research/v6_stage2/prepare_public_partial.py:59`는 이미 프레임 인덱스를 native PTS로 변환하고 있었으므로 이번 단위 확인을 기존 평가의 인과적 버그 발견이라고 해석하지 않는다.
- 학습 노트북은 공개 CCD 5건이라고 주석을 달고 있다. 그러나 이 감사가 검증한 원본은 `Baseline/data/stage2/videos/*.mp4`까지다. 그보다 앞선 긴 촬영 원본과 50프레임 시퀀스 사이의 추출·재번호화 대응은 확인하지 않았다. **이 영상을 우리가 40초 원본에서 10Hz로 만든 Nexar 입력과 동일한 처리 결과라고 부르거나, 숨은 평가 영상도 50프레임/10FPS라고 일반화할 근거는 없다.**

제공 클립의 PTS에서 ±0.3초를 적용하면 여기서는 ±3프레임과 같지만, 구현은 native rational PTS 차이를 3/10초와 직접 비교했다. 원본 FPS나 시각 대응이 없는 다른 입력에 이 환산을 적용할 수 없다.

## 대조 결과

| ID | 제공 라벨 index | 제공 클립 시각 | frozen V6 index | 보정 index | 두 방식의 절대 시각 오차 | ±0.3초 |
|---|---:|---:|---:|---:|---:|---|
| S2_001 | 32 | 3.2초 | 35 | 35 | 0.3초 | 일치 |
| S2_002 | 30 | 3.0초 | 43 | 43 | 1.3초 | 불일치 |
| S2_003 | 31 | 3.1초 | 33 | 33 | 0.2초 | 일치 |
| S2_004 | 41 | 4.1초 | 43 | 43 | 0.2초 | 일치 |
| S2_005 | 30 | 3.0초 | 33 | 33 | 0.3초 | 일치 |

이 표는 제공 라벨을 수정·제외하지 않은 계산이다. 앞선 AI 관찰에서 제기된 S2_002 라벨과 최초 실제 접촉의 의미상 불확실성은 해소하지 않았다. 제공 공식 라벨과의 일치와 시각적 물리 접촉 정의의 독립 검증은 별개다.

현재 보정은 input index 1의 초기 shift-difference 기여만 낮춘다. 공개 5개 모두 V6 최대점이 index 1이 아니므로 최종 argmax가 유지되었다. 실제 index 1 점수는 S2_001 0.379982→0, S2_002 1.236008→1.147614, S2_003 0→0, S2_004 5.643748→1.563683, S2_005 0.535522→0.535522다. 보정 자체가 실행되지 않은 것이 아니다.

## 계산 전에 통과한 근거 검사

1. `artifacts/submissions/submit_v6.manifest.json`의 Stage2 code 전체 SHA와 실제 `verify_v6/model/stage2/code/` 파일을 대조했다. V6 모듈 SHA는 `3e86e5117230fb4680c1d4af630cb02dd7e9aca99b7c4184d7baed13f479d64e`다. 보정 모듈·protocol SHA는 기존 `research/v7/motion_init_dev/freeze.json`과 일치한다.
2. `research/v5_inputs/manifest.json`의 labels·원본 영상·canonical PNG SHA를 모두 확인했다. 순차 해독한 원본 RGB 250개와 canonical PNG 픽셀 250개가 정확히 같다. 기존 `research/v6_stage2/public_pts/*.json`의 native PTS·time base도 재해독 결과와 정확히 같다.
3. 공개 5개의 **기존 V6 feature cache를 사용한 것이 아니다.** 실제 동결 V6 `_dual_motion_scan`을 CPU로 다시 실행해 특징을 기록했다. 재계산 base 점수는 frozen primitive `_motion_scan` 및 `research/v5_stage2/paired_layout_run/report.json`의 canonical V3 base 점수 250개와 float32 byte 단위로 일치했다.
4. 기록한 특징에서 median/MAD·floor·clip·가중치를 별도로 계산하여 V6 점수와 초기화 보정 점수가 모두 byte 단위로 일치함을 확인했다. index 1 이외 점수는 불변이다. 전체 SHA 바인딩은 실행 전후 동일했다.

따라서 표의 의미는 **검증한 원본 입력에서 frozen V6 수식과 단일 보정 수식을 재실행한 CPU 대조**다. 실제 V6 공개 5개 전체 VLM 예측을 재실행했다고 주장하지 않는다. 방향·진입·회피는 새로 추론하지 않았다. 공개 5개 canonical PNG는 제공 원본과 픽셀이 같지만 실제 서버 JPEG 렌더링과 비트 동일하다는 주장도 하지 않는다.

원시 특징·점수·native PTS·모든 SHA는 `motion_public_audit.json`, 재실행 절차는 `motion_public_audit.py`에 있다. 출력은 덮어쓰기 거부이며 실행 명령은 `.venv/Scripts/python.exe -I research/v7/motion_public_audit.py`다. 현재 실행 결과를 보존해야 하므로 재실행 목적의 기존 파일 삭제는 하지 않는다.

**최종 판정:** 제공 공식 접촉 라벨을 이용한 재사용 공개 데이터 대조는 가능하고 완료했다. 독립 성능 개선 근거는 없으며, 실패한 개발 gate 완화·후보 채택·제출 근거로 사용할 수 없다.
