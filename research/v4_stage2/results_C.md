# C 시작 이미지 참조 결과 — 미채택 근거

C는 사전 설계 `design_C_frozen_before_results.md` 및 `frozen_C.json` 후 실행했다. 기존 A/B source와 report, 원본 모델 파일은 전후 SHA 동일이다. 코드/질문을 결과 확인 후 수정하지 않았다.

CPU 계약 10개 PASS, 실제 공개 5개 총20질문 완료, 오프라인 연결 시도0. A의 첫 세 질문은 모든 영상에서 prompt/이미지 SHA/bounded 크기/토큰 예산/원응답이 동일했다. C의 첫 프레임 번호는 실제 파일명 N0이며 정수0 하드코딩을 하지 않았다.

마지막 질문은 다섯 영상 모두 `BEFORE_WINDOW`와 `entry_frame: null`을 반환했다. `ALREADY_IN_LANE_AT_START` 또는 `OBSERVED`는 한 건도 반환하지 않았다. 사전 guard에 따라 최종 예측은 모두 A/B와 동일했다. 따라서 C가 이미 차로 안인 사례를 복원했다는 근거도, 진입 성능을 개선했다는 근거도 없다. root/독립 reviewer의 시각 관찰은 공식 GT와 별개지만, 관찰이 일치한 S2_002/004에서도 기대한 복원이 없었다.

제공 충돌 라벨에 대한 ±0.3초 적중은 기존4/5 그대로다. predictor 평균10.190초, 질문 시간합 평균9.184초, scan 평균0.591초였다. 진단 PNG/로그 비용을 포함한 로컬 공개영상 측정이며 대회전체시간 보장이 아니다. 최대 CUDA allocated4,065,442,304 bytes.

입력 예산 confound도 실제 기록했다. 마지막 질문의 원본 첫 이미지1280×720은 로더에서1024×576으로 제한됐다. 8장 이하 local sheet1536×512는1312×416으로, 9~10장 local sheet1536×768은1088×544로 제한됐다. 총1.2M 예산을 두 이미지가600k씩 나누므로 B 대비 local 이미지도 달라진 비교다. 단순한 시작 시점 정보만의 효과로 해석하지 않는다.

근거: `C_run/report.json`, 각 `C_run/C/{ID}/call_4_entry_fine`의 PNG와 result.json. 모델 원응답을 잘못된 status라고 임의 수정하거나 N0로 바꾸지 않았다. 예측을 개선하려는 후속 문구 탐색은 하지 않는다.

- C source SHA256: `a9a7c2e542078e516d7b8c2175430a7b2f67c475051be06bbd5b26bbaf7be447`.
- 평가 source SHA256: `7e1fd40a2f0041fa78a0002d8688c7d1fe5922dadd321b73b2f4eda644160666`.
- C는 그 자체로 A의 S2_004 진입 변경을 유지한다. 이 변경까지 묶은 패키지 채택 근거가 없어, 마지막 D는 C 기반이 아니라 frozen V3 첫 질문의 한 문장만 바꾸는 별도 대조로 사전 확정했다.
