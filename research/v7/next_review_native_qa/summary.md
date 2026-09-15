# 고정6개 원본 영상 QA

완료. 실제 파일 SHA가 선택/획득 기록과 일치하며,6개 서로 및 기존12개 파일과 동일 SHA 중복은 0개다.

| ID | 실제 프레임 | 해상도 | stream timebase | 관측 FPS | PTS 간격 종류 | 누락/비증가 PTS |
|---|---:|---|---|---:|---:|---|
| 00017 | 1210 | 1280×720 | 1/15360 | 30.000000000 | 1 | 0/0 |
| 00018 | 1219 | 1280×720 | 1/15360 | 30.000000000 | 1 | 0/0 |
| 00019 | 1226 | 1280×720 | 1/19584 | 30.600000000 | 1 | 0/0 |
| 00021 | 1199 | 1280×720 | 1/15360 | 30.000000000 | 1 | 0/0 |
| 00022 | 1213 | 1280×720 | 1/15360 | 30.000000000 | 1 | 0/0 |
| 00023 | 1217 | 1280×720 | 1/15360 | 30.000000000 | 1 | 0/0 |

각 영상의 `.frames.json`은 zero-based sequential decoded index와 원시PTS/timebase/정확 유리수시각·해상도를 보존한다. `report.json`에 첫·중간·마지막 RGB SHA를 기록했다. PNG 전체 덤프는 만들지 않았다.

SHA equality detects identical file bytes only. No perceptual, same-incident, route, training/pretraining independence certification. No human/AI annotations or model predictions read. Rational PTS variation is reported, not automatically treated as corrupt video.

실행 12.11초. GPU·다운로드·모델·정답/검수 주석 열람 없음.