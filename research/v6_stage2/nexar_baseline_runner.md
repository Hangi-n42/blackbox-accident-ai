# Nexar 두 개발 영상의 frozen V5 진단 실행기

`run_nexar_baseline.py`를 구현하고 CPU 입력 준비를 완료했다. GPU 실행은 루트 담당이다. 사용자 GT를 읽거나 연결하지 않았다. 코드가 허용하는 ID는 00000, 00003뿐이며 예약 검증 영상은 열거나 분석하지 않았다.

실제 V5 추출본 `artifacts/submissions/verify_v5`의 `stage2_motion_collision`과 저장된 NF4 모델을 사용한다. 두 영상에서 기존 네 호출을 그대로 실행하며 후보 모델/프롬프트/정책을 지원하지 않는다. 추출본에 파일이나 bytecode를 만들지 않는다. `-I` import 격리, HF offline 및 socket 차단을 적용한다.

입력 정책은 전체 native PTS의 시작부터 마지막까지 0.1초 간격 목표 시각에 가장 가까운 원본 프레임이다. 동률은 앞 프레임, 중복 선택은 제거하며 첫 프레임과 마지막 프레임을 포함한다. 원본 decoded index를 PNG 파일명에 보존한다. 양끝 포함 때문에 프레임 수는 정확히 영상길이×10일 필요가 없다. PNG는 RGB lossless, compress_level=1이다. 정답을 이용해 영상 범위를 자르지 않았다.

실행기와 입력 freeze:

- `nexar_baseline_run/freeze.json`
- SHA-256: `93f041d51dbc8d4dfc6db279a5860d9dc2b7e84a4e561077e62dfd5972ff8c4b`
- 각 영상 402장, 합계 804장.
- root `nexar_cohort_plan.json`은 내용을 읽지 않고 SHA로만 연결했다.

루트 실행 명령:

```text
.venv\Scripts\python.exe -I -B research/v6_stage2/run_nexar_baseline.py run --output research/v6_stage2/nexar_baseline_run
```

실행 전후 실제 V5 Stage2 코드/모델 바이트 SHA, 원본 영상/입력 PNG, 실행기 및 protocol 해시를 대조한다. 네 호출의 PNG·원문 질문·원시 답변·출력 토큰 한도·영상 크기·모델 전처리 크기·시간·VRAM을 기록한다. 호출 1~3의 후보는 원문에서 추출하며, 네 번째 회피공간 질문의 후보는 네 호출이 끝난 뒤 실제 내부 collision과 valid index에서 계산해 `call_4/offered_context.json` 및 `trace.json`에 남긴다. 네 번째 질문 원문에는 번호 목록이 없기 때문에 이 보조 기록의 출처를 구분한다.

최종 trace에는 전체 입력 번호, 정상 디코드 번호, native PTS, motion scores, 내부 VLM 충돌 번호, 최종 motion 충돌 번호, 실제 entry prefix와 후보·최종 선택이 포함된다. 생성은 총 8회 제한이며 파일별 recorder 상태를 새로 만든다. 실패 기록도 보존하고 같은 결과 경로 재실행은 거부한다.

CPU 계약 6개 통과: 불규칙 PTS와 원본 index, 동률/중복, 시간 원점 이동·한 프레임, 비정상 PTS 거부, ID/토큰 예산 고정, recorder 입력·출력 동일 전달 및 파일별 호출 예산. 이것은 동작 검사이며 모델 정확도 검증이 아니다. 공식/사용자 GT의 평가는 이 실행기에 없다.
