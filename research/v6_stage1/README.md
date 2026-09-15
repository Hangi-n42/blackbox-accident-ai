# V6 Stage1 준비 결과

2026-09-14. 완료한 범위는 기존 가용 자료의 제한된 재확인, 촬영·주석 양식, 메타데이터 검사 도구와 계약 검증이다. **실제 도로 화면 재촬영 취득, 사람 검수, Stage1 새 모델 학습·추론·채택은 수행하지 않았다.** 기존 공유 모델·자료·코드는 변경하지 않았다. 이 폴더만 추가했다.

## 가용 자료 재확인

| 기록으로 확인한 자료 | 실제 이용 가능한 범위 | 목표 도로 재촬영 양성으로 인정 가능한가 |
|---|---|---|
| baseline 공개 5원천의 원본 5개·재촬영 5개 | `training_manifest.json`에 재촬영은 simulated로 기록 | 아니오: 합성 |
| comma 도로 원본 23구간 | 공개 중복 1개를 제외한 기존 동결 원본 오탐 진단. 두 차량 계열 | 아니오: 원본만 있음 |
| DLC 6문서×24 원영상 부분집합, 192 JPEG | 물리적 문서 원본/화면 재촬영, 영상당 8장 | 아니오: 실제 재촬영이나 문서 도메인이고 JPEG 부분집합 |
| TPO 학습 데이터 | 기존 코드/가중치 및 원천 설명; 신청 배포 원자료는 미취득 | 아니오: 실내 물체 도메인이며 현 체크포인트 학습 원천 |

검토한 manifest/진단 기록에서 **실제 도로 화면 재촬영 양성은 0개**다. 이는 위 기록 범위의 결론이며 디스크 전체를 새로 광범위 탐색한 결과가 아니다. 자료·장비·사람 검수 가능 여부는 아직 확인되지 않았다. 기존 진단의 TPO/forensic/결합 수치는 새 실험으로 다시 계산하지 않았다.

근거 파일: `research/stage1/training_manifest.json`, `research/stage1/comma_original_diagnostic/frozen_manifest.json` 및 `metrics.json`, `research/stage1/dlc_subset/frozen_diagnostic_manifest.json` 및 `frozen_metrics.json`, `research/stage1/real_recapture_validation_audit.md`. 이번에 읽은 파일의 SHA256은 `validation_result.json`에 보존했다.

## 산출물과 검증

- `촬영_기록_안내.md`: 실제 취득·원본 대응·권리·촬영 증거·기기·분할·사람 주석 안내.
- `record_template.json`: null 기반 필드 양식. 촬영했다고 꾸민 예시 데이터가 아니다.
- `capture_manifest.json`: 준비 상태의 빈 records. protocol 선택 및 동결도 미완료.
- `check_capture_manifest.py`: 파일 SHA·필수 provenance·물리 재촬영 증거 존재·부모 원본 일치·source/content/device split 중복·동일 SHA의 상충 라벨 검사. 입력과 자산에 쓰지 않는다. 명시된 report는 신규 파일로만 생성한다.
- `test_manifest_checks.py`: 실제 영상이 아닌 임시 fixture로 누락·합성·분할 누출·부모 불일치·권리/검수 미완료·SHA 오류 등을 검사.
- `readiness_report.json`: 현재 빈 manifest를 검사한 실제 결과 **NOT_READY**. accepted_counts는 null이다.
- `validation_result.json`: 직접 Python subprocess로 확인한 종료 코드 **2**, 계약 테스트 **9개 통과**, 코드 및 근거 SHA.

실행: `python research/v6_stage1/test_manifest_checks.py` → 9 tests OK. `python research/v6_stage1/check_capture_manifest.py research/v6_stage1/capture_manifest.json` → NOT_READY, exit 2. 준비 목록이 비어 있으므로 실패 상태가 의도한 정상 동작이다. 테스트 fixture의 bytes는 영상이 아니며 검사기 통과가 물리 촬영 진위나 디코딩을 보증하지 않는다는 한계도 테스트에 명시했다.

실제 사용 명령과 필드 설명은 촬영 안내에 있다. 추가 패키지·GPU·네트워크가 필요하지 않다. 사람 검수 명단이나 취득 영상을 만들지 않았으며 자동 생성한 라벨도 없다.

## V6 계획과 채택 기준 독립 검토

`research/v6/README.md`의 “GT 일부만 있으면 전체 S2를 산출하지 않는다”, “수치 문턱을 후보 예측 전에 정한다”, “정확한 대응이 확인되기 전 후보 탐색을 하지 않는다”는 범위 제한은 타당하다. 미응답을 사람 검수 완료로 간주하지 않는 것도 필요하다. Stage2 항목 하나만 바꾸는 제한된 개선이 유효하게 검증된다면, Stage1 새 모델이 없다는 이유만으로 V6 전체를 막을 필요는 없다. 반대로 준비 도구 완성만으로 정확도 검증이나 제출 목표를 완료로 처리할 수 없다.

다음 한계를 유지해야 한다.

1. source/device holdout은 누출 방지 절차이며 통계적 일반화 보장이 아니다. 원천·물리 기기·기기 모델·촬영 조건 각각의 독립성은 다르다. 23구간을 23카메라로 세거나 인접 프레임을 독립 표본으로 늘리지 않는다.
2. 모든 소수 공개 사례 무회귀 조건은 위험 회피 정책이다. 이를 통계적으로 유의한 개선 또는 목표분포 최적성의 증명으로 표현하면 안 된다. 반대로 결과를 본 뒤 후보를 살리려고 기준을 느슨하게 해도 안 된다. 개선폭·허용 손실·불확실성 판단은 정답 품질과 원천 수를 먼저 확인하고 고정한다.
3. 사람 주석이 존재해도 대회 정의의 물리적 접촉/첫 차선 진입, 실제 시간축과의 대응이 맞는지는 별개다. 넓은 AI 구간, 사고 시작이나 위험 인지 시점을 exact GT로 바꾸지 않는다. Stage1은 촬영 과정·출처가 클래스 근거이며 외관만으로 물리 재촬영을 확정하지 않는다.
4. 현재 Stage1의 목표 양성 부족은 해결되지 않았다. 다음 승인된 진단에서 실제 도로 대응 촬영을 확보하기 전에는 합성/DLC 재보정 반복을 목표 도메인 검증으로 대신하지 않는다. 정해진 N만으로 충분성을 주장하지 않는다.

실행 가능한 다음 입력은 실제 취득 원본/재촬영 파일, 사용권 근거, 기기/조건 기록과 사람 확인이다. 없는 입력은 null/미완료로 유지한다. 현재 산출물은 그 입력이 들어왔을 때 허위 완료나 원천·기기 누출을 조기에 검출하기 위한 준비물이다.
