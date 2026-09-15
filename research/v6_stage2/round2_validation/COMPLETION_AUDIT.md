# V6 제출 목표 완료 감사

목표의 최종 동작인 제출 및 접수를 확인했다. 서버 점수 개선·서버 전체 실행 완료는 이번 완료 판정에 포함하지 않으며 아직 미확인이다.

| 요구 | 확인한 실제 근거 | 결론과 한계 |
|---|---|---|
| 실제 Stage2 정밀 사람 검수와 오류 분해 | 추가 4파일 무결성 감사, 전체 6개 V5 진단, 새 3개 intake_20260915_actual/integrity.json 및 trace | 실제 사람 JSON 원본 보관과 원본 프레임·시각 대응 검증. AI 생성 정답을 대체 사용하지 않음. 단일 사람의 관측 기준이라는 한계 유지 |
| 한 원인 수정 | 고정 C SHA 3e86e5117230fb4680c1d4af630cb02dd7e9aca99b7c4184d7baed13f479d64e, ZIP manifest 47파일 | 최종 접촉의 jerk 점수 상한만 제거. 모델 가중치와 다른 단계 유지 |
| 수정에 사용하지 않은 자료에서 개선·부작용 확인 | 사전 protocol/freeze, 실제 사람 binding, run/report.json 및 evaluation.json | 고정 새 3영상에서 접촉 0/3→1/3, MAE 비증가, 기존 성공 손실 0, 다른 세 출력 동일. 사건·기기·사전학습의 완전한 독립성이나 모집단 개선을 보장하지 않음 |
| Stage1 실제 재촬영 자료 준비 병행 | research/v6_stage1/촬영_기록_안내.md, record_template.json, check_capture_manifest.py, readiness_report.json | 촬영·등록·검사 절차 준비. 실제 대응 자료는 0쌍으로 명시하며 취득이나 Stage1 학습 성공을 주장하지 않음 |
| Stage3 정답 정합성 감사 | research/v6_stage3/summary.md 및 audit_report.json | 공식 대응 10행의 가감속 클래스 공백과 조향 불일치 1행 확인. 변경 근거 부족으로 기존 모델 유지 |
| 실제 ZIP·오프라인·자원 검사 | submit_v6.manifest.json, verify_v6_results/report.json PASS | CRC/SHA/47파일, 2.91GB ZIP·3.42GB 추출, 실제 entry point 3단계 실행. S1 10행/S3 2998행 V5 동일, S2 3행 후보 동일, network 0. 유한 로컬 실행 및 기존 V5 서버 이력으로 검토했으며 hidden 전체 시간은 미확인 |
| DACON 제출 및 접수 확인 | submit_v6.receipt.json, 브라우저 제출 완료 안내 및 90342 목록 행 | 2026-09-15 17:52:35 KST 접수, 대기 중. 중복 제출하지 않음 |

실제 새 정답·예측을 평가 후 바꾸지 않았고, 옛 10Hz 예약 검증 실패도 소급해서 통과 처리하지 않았다. 새 3개는 이제 결과가 노출됐으므로 후속 후보의 미사용 검증으로 재활용하지 않는다.
