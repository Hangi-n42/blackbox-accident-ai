# 새 세 영상 검증 준비 — 사람 검수 연결 전 실행 불가

## 최신 준비 상태 — 2026-09-15 후속

사람 검수 importer(`import_reviews.py`), 실제 새 3개 평가기(`evaluate_validation.py`), 통과 결과에만 연결되는 패키저(`package_v6.py`), 실제 추출 ZIP 실행 검사기(`verify_v6_offline.py`)를 준비했다. 아래 본문의 importer 미준비·freeze 미실행 문장은 초기 전달 당시 기록이며 현재 상태가 아니다. **기존 `run/freeze.json`은 이미 생성됐으므로 freeze 명령을 다시 실행하지 않는다.**

준비 검사: importer 합성 자료 8개 검사, evaluator 합성 native 대응 4개 검사 및 판정 자체 검사, offline 계약 4개 검사가 통과했다. 패키저 문법 및 실제 평가 파일 누락 시 산출물 없이 거절하는 검사도 통과했다. 이는 실제 새 영상의 정확도나 실제 ZIP 추론을 검증한 결과가 아니다.

현재 실제 `run`에는 `freeze.json`만 있다. 새 3개의 사람 검수 JSON, review_binding, 추론 report, evaluation, V6 ZIP은 아직 없다. 검수 화면에서 NEXAR_REVIEW_00008·00010·00013의 실제 상대와 최초 접촉을 기록해야 한다. 판정 불가는 그대로 보존하며 정답을 추정하지 않는다.

실제 파일 수령 후 순서는 다음과 같다. `<...>`는 실제 경로로 교체하며, 모든 출력 폴더는 새 경로를 사용한다.

```powershell
.\.venv\Scripts\python.exe -I -B research/v6_stage2/round2_validation/import_reviews.py --review <00008_JSON> <00010_JSON> <00013_JSON> --output <새_작업공간_검수보관폴더>
.\.venv\Scripts\python.exe -I -B research/v6_stage2/round2_validation/run_validation.py run --output research/v6_stage2/round2_validation/run
.\.venv\Scripts\python.exe -I -B research/v6_stage2/round2_validation/evaluate_validation.py --run-dir research/v6_stage2/round2_validation/run
```

앞 단계의 실제 성공을 확인한 후 다음 단계로 진행한다. 검수 불확실성·무결성 실패·개선 조건 실패가 있으면 자동으로 통과 처리하지 않는다. 새 검증 통과 후 출처 중복 감사와 한계를 확인한 뒤 다음 절차를 적용한다.

```powershell
.\.venv\Scripts\python.exe -I -B research/v6_stage2/round2_validation/package_v6.py --validation-run research/v6_stage2/round2_validation/run
.\.venv\Scripts\python.exe -I -B research/v6_stage2/round2_validation/verify_v6_offline.py --package artifacts/submissions/verify_v6 --manifest artifacts/submissions/submit_v6.manifest.json --validation-run research/v6_stage2/round2_validation/run --stage1-data artifacts/public_eval/stage1 --stage3-data artifacts/public_eval_10hz/stage3 --baseline-results artifacts/submissions/verify_v5_results --output artifacts/submissions/verify_v6_results
```

새 manifest는 실제 새 평가의 `validation_evaluation_sha256`을 연결한다. 기존 10Hz 예약 평가 실패를 요구하는 옛 패키저는 실행하지 않는다. Stage2는 새 영상의 전체 원본 PNG, Stage3는 기존 출력과 대응하는 10Hz 입력을 사용한다. 새 검사기는 PNG 원본 번호를 검사하고 실제 ZIP 진입점 출력을 비교한다. 유한한 로컬 실행 시간은 전체 hidden 평가 시간 보장이 아니며, 제출 전 실제 자원 기록과 최신 대회 제약을 확인한다.

## 초기 전달 기록

최신 실행 상태(2026-09-15): root가 source freeze를 실제 완료했다. `run/freeze.json`에 새 3개 원본·전체 프레임·코드·모델·비교 조건을 고정했다. 새로운 모델 추론과 사람 검수 연결은 아직 수행하지 않았다. 아래의 freeze 미실행 설명은 코드 전달 시점 기록이다. 검수 화면 http://127.0.0.1:8766/ 에 00008/00010/00013을 등록했고 응답을 확인했다. 각 영상의 실제 충돌 상대와 최초 접촉(또는 판정 불가 사유)을 검수해 JSON을 보존해야 한다. 기존 제출 지시는 유지되며, 여기에 적힌 조건은 재승인 요구가 아니라 정확도·오프라인 검증 조건이다.

대상은 00008/00010/00013이다. 정확한 C 코드 SHA를 고정하고 기존 V5 전체 PNG·4회 호출을 사용한다. 최대 모델 호출은 12회, NF4 로딩은 1회다. Stage1/3은 변경하지 않는다. 기존 여섯 영상의 통과는 새 자료 검증 진행만 허용하며 채택을 뜻하지 않는다.

현재 작성한 것은 protocol과 runner다. **실제 새 검수 파일·review_binding.json은 만들지 않았다.** 사람 검수 importer는 root가 실제 파일을 받은 뒤 준비해야 한다. `run`은 review_binding.json이 없으면 모델 import 전에 반드시 중단한다. 준비 과정에서 source freeze도 실행하지 않았다.

Root 실행 명령:

```powershell
.\.venv\Scripts\python.exe -I -B research/v6_stage2/round2_validation/run_validation.py freeze --output research/v6_stage2/round2_validation/run
```

위 단계는 전체 source/case/mapping/PNG·선택 기록·모델·코드·protocol과 이전 진단 통과 자료를 해시로 동결한다. 검수 내용을 읽지 않는다. 원본 PNG는 복제하지 않는다.

이후 root importer가 실제 세 초안을 검증하고 `run/review_binding.json`을 새 파일로 생성해야 한다. 필수 구조는 다음과 같다. 이 설명은 schema이며 실제 binding이 아니다.

- status=`VALIDATED_CONTACT_BINDING`, ground_truth_promotion=false, predictions_seen_before_binding=false
- freeze_sha256, protocol_sha256
- validator={path,sha256}: 실제 importer 코드
- integrity_report={path,sha256}: 원본 review/영상 SHA, case.frames JS JSON.stringify SHA, 원본 native PTS와 선택 PNG를 대조한 실제 감사 결과
- reviews: 00008/00010/00013 순서. 각 항목의 정확한 키는 ID, review_path, review_sha256, source_sha256, frame_mapping_sha256, contact_status, native_pts_validated, selected_png_validated. 마지막 두 값은 검증 후 true, contact_status는 실제 observed여야 한다. contact frame/시간·대상 설명 등 답변 내용은 binding에 넣지 않는다.

Importer는 실제 사람 JSON의 ID/원본SHA/관측 contact/프레임-PTS 대응/PNG를 검증하고 원본 바이트를 보존해야 한다. 검수 파일 누락, uncertain 또는 실제 접촉 식별 불가를 observed로 바꾸지 않는다. 단일 사람 초안을 확정 GT로 승격하지 않는다. 실행기는 importer의 감사 파일과 원본 review를 해시로 확인하며, 원본 검수 답변 내용은 파싱하지 않는다.

조건이 충족된 뒤에만 root가 실행한다.

```powershell
.\.venv\Scripts\python.exe -I -B research/v6_stage2/round2_validation/run_validation.py run --output research/v6_stage2/round2_validation/run
```

프롬프트·원문 응답은 trace 파일에 보존하되 콘솔에는 ID/상태만 표시한다. 사람 검수자에게 trace나 report를 미리 보여주지 않는다. 같은 소스에 재시도·덮어쓰기는 허용하지 않는다.

한 번의 dual scan에서 V5 식을 별도로 다시 계산해 base float32 바이트가 정확히 같은지 확인한다. 별도 원본 optical-flow scan을 한 번 더 수행한 것은 아니므로 결과 키도 `base_scores_exact_original_expression=true`, `base_scores_equal_separate_original_scan=null`로 구분한다. 기존 CPU 이미지 계약에서 원본 scan과의 동등성을 확인한 C 구현을 그대로 사용한다.

나중의 별도 사람 기준 평가는 ±0.3초(+1e-12 여유) 정답 수 최소 +1, 기존 정답 손실 0, MAE 비증가, 다른 세 필드 동일을 모두 요구한다. 통과도 전체 S2나 leaderboard 성능 보장이 아니다. 같은 사건·기기 원천 독립성, 실제 ZIP 등가성·자원 조건은 별도 확인 대상이다.
