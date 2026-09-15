# 예약 검증 입력 준비 계약

`run_nexar_reserved_baseline.py`는 00005/00006/00007의 CPU 입력 준비와 동결 파일 검증 전용이다. GT와 예측을 읽지 않는다. `run()`은 입력에 접근하기 전에 무조건 예외를 내며 CLI는 `prepare`/`verify`만 허용한다. 이 도구로 모델 추론을 실행할 수 없다.

원 개발 split 00000/00003/00004 및 예약 split 00005/00006/00007은 그대로다. `nexar_reserved_protocol_addendum.json`은 원 cohort plan, 기존 개발 실행기와 입력 freeze, 00004 실행기와 freeze의 SHA를 보존한다. 기존 파일 내용은 수정하지 않았다.

입력은 전체 원본의 native PTS를 순차 디코딩한 후 0.1초 격자의 가장 가까운 원본 index를 선택한다. 동률은 앞 프레임, 양끝 포함, 중복 index 제거, 원본 decoded index의 PNG 이름을 보존한다. RGB lossless PNG compress_level=1이며 GT로 구간을 자르지 않는다. `sampled_indices`, `prepare`, `package_binding` 함수는 기존 개발 실행기와 AST가 동일하다.

baseline용 메타데이터 예산은 영상당 4호출, 합계 12호출, 생성 토큰 64/48/40/40이다. 이 값은 준비 도구의 추론 허용을 의미하지 않는다. 실제 예약 평가의 baseline 4호출 + 추가 검증 1호출은 개발 gate 통과와 후보 소스 SHA 동결 뒤 루트의 별도 paired 실행기에서 같은 인스턴스로 수행한다. 이 도구는 GT·개발 성능 gate·후보 SHA의 내용을 판단하지 않는다.

CPU 계약은 ID 화이트리스트, 예산 및 토큰 보존, 합성 PTS 선택의 기존 정책 동일성, 직접 run 차단, 기존 파일 해시 불변을 확인했다. 모델 성능 검증은 아니다.

준비된 입력 검증 명령:

```text
.venv\Scripts\python.exe -I -B research/v6_stage2/run_nexar_reserved_baseline.py verify --output research/v6_stage2/nexar_reserved_inputs
```

별도 root 실행기가 이 파일을 import하여 `verify_freeze(output)`를 호출할 수 있다. 함수는 실행기·실제 V5 추출 코드/모델·원본 영상·PNG·protocol 해시를 검사한다. 가져오기만으로는 모델 import나 추론이 발생하지 않는다. 완료 후 입력 수와 해시는 `nexar_reserved_ready.json`에 기록한다.
