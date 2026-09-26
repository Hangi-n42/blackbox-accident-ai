# Stage2 다음 단일 변경 검토

2026-09-20. 새 모델 호출·훈련·운영 코드 수정 없이 기존 응답 20건과 실제 V6/Mac 경로를 읽고 CPU 파싱을 대조했다. **다음 한 번의 진단은 기존 진입 시트에 검수된 동일 상대의 픽셀 표식만 주는 대조가 타당하다.** 시간창·질문·파서를 동시에 바꾸지 않는다. 현재 자료는 이 표식을 자동 생성하는 제출 정책의 성능을 뒷받침하지 않는다. 따라서 아래는 구현 가능한 보조정보 진단 규약이며, 자동 제출 후보 채택 결정은 아니다.

## 코드에서 확인한 원인과 확인하지 못한 원인

1. **상대 동일성이 질의 사이에서 유지되지 않는다.** Q1은 전체 10장으로 접촉 번호와 방향만 받고 `entry_side`를 즉시 확정한다. Q2는 별도 접촉 후보에서 번호만 받는다. Q3에는 Q1의 방향·차량 외형·bbox·Q2의 원본 참조 이미지가 전달되지 않는다. 단지 `the other collision vehicle`라는 문구와 내부 접촉까지의 12장만 준다. Q1의 side가 Q3을 잘못 유도한다는 직접 데이터 흐름은 없다. 서로 독립적으로 다른 상대를 정할 수 있는 구조다. [V6 Q1–Q3](/Users/hyeongi/projects/blackbox-accident-ai/releases/v7/source/model/stage2/code/solution/stage2_v2.py:72)

2. **Mac에서도 앞선 대화 기억은 없다.** `MLXVLM.ask`는 매번 하나의 user 메시지를 새로 만들고 prompt/image에서 processor 입력을 구성한다. 같은 모델 객체를 사용한다는 사실은 Q1의 상대가 다음 질의에 전달된다는 뜻이 아니다. [Mac 메시지 구성](/Users/hyeongi/projects/blackbox-accident-ai/scripts/mac/mlx_stage2.py:45)

3. **이미 시작 전 진입 규칙이 구현돼 있다.** Q3의 기존 문구는 첫 이미지부터 차로 안에 있으면 첫 이미지를 고르라고 지시한다. 후보는 `uniform_indices(0, internal_collision, 12)`이므로 첫 원본도 포함된다. 같은 문구를 더 강하게 반복하거나 0을 다시 추가하는 것은 새 원인 수정이 아니다. 일반 제출에서는 첫 원본 번호를 사용해야 하며 무조건 숫자 0을 새로 만드는 정책도 부적절하다. [기존 규칙·후보·선택](/Users/hyeongi/projects/blackbox-accident-ai/releases/v7/source/model/stage2/code/solution/stage2_v2.py:102)

4. **파서의 위험과 실제 발생을 분리해야 한다.** `_json_object`는 처음 파싱 가능한 객체를 취하고, `_choice`는 허용되지 않은 숫자를 가장 가까운 후보로 snap하며 무효 응답은 첫 후보로 보낸다. 따라서 first-frame fallback을 이미 진입한 상태의 인식 성공으로 세면 안 된다. 그러나 이번 기존 20건에서는 Q3 원응답 모두 실제 후보 번호이고 최종 진입과 같았다. side fallback도 0건이다. 현재 실패를 JSON 파싱으로 설명할 근거는 없다. [파서](/Users/hyeongi/projects/blackbox-accident-ai/releases/v7/source/model/stage2/code/solution/stage2.py:123)

5. **시간 경계와 최종 출력은 별도 문제다.** Q3 상한은 내부 VLM 접촉이며 후처리는 최종 접촉만 두 차례 덮어쓴다. Q3 선택 또는 이미지 표식만 바꾸면 잘못된 상한·희소 후보·공간 시점은 그대로다. [base motion 교체](/Users/hyeongi/projects/blackbox-accident-ai/releases/v7/source/model/stage2/code/solution/stage2_motion_collision.py:19), [uncapped 교체](/Users/hyeongi/projects/blackbox-accident-ai/releases/v7/source/model/stage2/code/solution/stage2_uncapped_jerk_v6c.py:97)

저장소 `solution/stage2_v2.py`, release V7의 동명 파일, Mac이 실제 import하는 `artifacts/submissions/verify_v6/model/stage2/code/solution/stage2_v2.py`의 SHA-256은 모두 `89ae0bc640670d7ed76103ce7c829d405ad27cc25368c74d153d3b96962b8442`다. Mac 기준 경로는 [run_stage2.py](/Users/hyeongi/projects/blackbox-accident-ai/scripts/mac/run_stage2.py:20)에서 확인했다. 새 구현은 운영 파일을 고치지 않고 기존 `_predict_file`의 세 번째 `ask` 입력을 연구 wrapper에서 치환하면 된다.

## 기존 증거의 범위

| 자료 | 현재 근거 | 해석 |
|---|---|---|
| 사람 초안 9건 | 접촉 4/9, 진입 0/8; 진입 A0=8 → 내부 접촉 상한 A1=7 → 실제 후보 A2=4 → 선택 P=0 | 단일 사람 초안·개발 노출. 누락 원인: 상한 1건, 후보 3건, 선택 4건 |
| 공개 5건 | 현재 JPEG 입력 접촉 4/5, MAE .54초; 진입·방향·공간 정답 없음 | 진입 성능 분모에 넣지 않음 |
| 새 00024–00029 | 네 엄격 타깃 각각 0건; 실제 실행 네 출력이 보조합 제거 전후 6/6 동일 | 실행 진단이며 0% 정확도나 개선 없음의 성능 추정이 아님 |

위 수치는 [현재 core 분해](/Users/hyeongi/projects/blackbox-accident-ai/artifacts/stage2_goal_20260919/core_v2/decomposition.json:1)와 [새 6건 조정 검수](/Users/hyeongi/projects/blackbox-accident-ai/artifacts/stage2_goal_20260919/expert_mac_review/adjudication/review.md:3)에 근거한다. 과거 CUDA 기준 진입 1/8을 현재 Mac 기준 0/8과 혼합하지 않는다. 공개 PNG .46초와 JPEG .54초도 별도 입력 비교다.

현재 파싱 대조는 사람 초안 9건·공개 5건·새 6건의 저장 `result.json`을 사용했다. **20/20에서 12개 후보에 첫 프레임이 있고, already-inside 문구가 있으며, Q3 원응답이 유효 후보이고 최종 진입과 일치한다.** 세부 값과 출처 해시는 `next_policy_review.json`에 보존했다. 영상 직접 재검수 또는 모델 재추론 결과는 아니다.

진입 선택 실패의 구체 사례는 다음과 같다.

- 00003: 초안 규칙 진입 f0, 실제 후보에 f0 있음, 모델 f194. 00006: 초안 f0, 후보 f0 있음, 모델 f436. 두 사례 모두 문법적 fallback이 아니다.
- 00000·00007도 초안 허용 범위 후보가 있으나 다른 번호를 고른다. 00007의 f172가 다른 상대인지, 같은 상대의 다른 진입인지 숫자만으로 확정하지 않는다.
- 00005·00008·00013은 후보 누락, 00010은 내부 접촉 f337 상한이 초안 진입 f572를 제외한다. 표식만 바꿔도 이 네 사건의 정확도 상한은 복구되지 않는다.
- 00028은 f0를 포함한 후보 `[0,56,111,167,223,279,334,390,446,502,557,613]`에서 f446을 출력했다. 앞 SUV가 실제 사고 상대일 때만 f0가 규칙값이다. 상대가 확정되지 않아 엄격 오답으로 세지 않는다. [원응답·후보](/Users/hyeongi/projects/blackbox-accident-ai/artifacts/stage2_goal_20260919/expert_mac_review/mac_run/00028/result.json:24), [조건부 근거](/Users/hyeongi/projects/blackbox-accident-ai/artifacts/stage2_goal_20260919/expert_mac_review/adjudication/review.md:20)
- 00026은 진입 f324가 최종 접촉 f301 뒤다. 내부 VLM 접촉 f509를 상한으로 사용한 결과와 양립한다. 번호 순서 이상은 확인됐지만 실제 진입 정답을 모르므로 `min(entry, final_contact)`로 강제 수정하지 않는다.

## 이미 실행했으므로 반복하지 않을 방식

| 방식 | 기록된 결과와 기각 이유 |
|---|---|
| contact 참조 이미지 + SAME vehicle + 전체 12장 + 두 번 refinement | Mac 진입 0/8→1/8이나 MAE 12.6237→13.5838초. 00006의 f0도 후속 창에서 탈락. 추가 질의 최대 3회. [결과](/Users/hyeongi/projects/blackbox-accident-ai/docs/mac-priority01-results-20260916.md:29) |
| 위 temporal의 anchor만 최종 접촉으로 변경 | 당시 AI 참조 진입 1/4→1/4. 새 엄격 GT 수치가 아니며 현재 정제 자료와 섞지 않음. [결과](/Users/hyeongi/projects/blackbox-accident-ai/artifacts/data_pilot_20260916/nexar/final_contact_ablation.md:3) |
| 최종 접촉 시점으로 space 재질문 | 9/9 응답 0, Macro-F1 .1818 유지. [결과](/Users/hyeongi/projects/blackbox-accident-ai/research/v7/research_decision_20260915.md:27) |
| contact 질문 문구만 변경 | 저장 공개 후보 63개를 어느 것으로 바꿔도 최종 motion 출력 63/63 불변. [구조 검사](/Users/hyeongi/projects/blackbox-accident-ai/artifacts/next_experiments_20260917/REPORT.md:26) |
| V6A 재확인, V6B 개별 타일, 사람 상대 설명 | V6A 1/3→1/3·MAE 악화, V6B 1/3→1/3, 설명 추가 3/3 null. 새 contact 질의 반복의 근거가 아님. [V6A](/Users/hyeongi/projects/blackbox-accident-ai/research/v6_stage2/contact_verify_v6a_result.md:3), [V6B](/Users/hyeongi/projects/blackbox-accident-ai/research/v6_stage2/contact_tiles_v6b_development/evaluation.json:4), [설명 진단](/Users/hyeongi/projects/blackbox-accident-ai/research/v6_stage2/human_target_diagnostic/result.md:5) |
| 초기 jerk artifact 보정 | 접촉 4/9 유지. MAE 개선은 있었지만 정확도 증가 gate 실패. [결과](/Users/hyeongi/projects/blackbox-accident-ai/research/v7/research_decision_20260915.md:26) |
| 8B 확대·복합 target/bbox/time 계획·native video | 기존 정책 8B 개선 없음, 복합 계획 null 다수·접촉 회귀, native video 4/9→1/9. [연구 결과](/Users/hyeongi/projects/blackbox-accident-ai/research/v7/research_decision_20260915.md:15), [video 결과](/Users/hyeongi/projects/blackbox-accident-ai/research/v7/progress.md:63) |

jerk-only 보조합 제거는 기존 개발 00007 한 건의 개선 신호를 냈지만 새 6건 출력은 동일했다. 이 결과를 근거로 가중치·초기화·진입 규칙을 이번 대조에 추가하지 않는다.

## 단일 진단 규약: Q3 시트의 동일 상대 픽셀 표식

**가설:** 시작부터 차로에 있는 동일 상대를 현재 모델이 명확히 시각적으로 지정받으면, 이미 존재하는 first-image 규칙을 적용하여 올바른 후보를 선택할 수 있다. 성능은 아직 확인 불가다. 이 실험은 동일성 정보와 시각적 주목을 함께 주므로 성공해도 순수한 identity 원인만의 인과효과로 표현하지 않는다.

**변경 하나:** 기존 Q3의 12개 프레임·번호·순서·전체 차로 영상·시트 크기는 그대로 두고, 각 시점에서 독립 검수로 확인한 동일 차량의 상단 차체에 고정된 얇은 표식만 그린다. contact 정답 시점의 새 reference 프레임·crop은 추가하지 않는다. 기존 후보에 보이는 상대가 픽셀 수준 참조가 된다. Q3 문구와 `entry_frame` 한 키, 40토큰, `_choice`도 그대로다. 이를 이미 실패한 시간창·질문 변경과 구분한다.

**기존 코드 재사용:** `_sheet(paths, entry_candidates, columns=4)`로 원래 시트를 만들고 해당 시트에 표식을 덧그린다. 기존 V6 `_predict_file`에 전달하는 `ask` wrapper가 세 번째 호출에서만 이미지를 바꾼다. `_json_object`, `_integer`, `_choice`, 원본 번호 lookup을 그대로 사용한다. Q1/Q2/Q4 응답은 캐시에서 돌려주므로 다른 세 출력과 내부 접촉 상한은 고정된다. `temporal_entry.refine_entry`나 `stage2_v7_grounded` 전체를 도입하지 않는다.

**표식 준비와 누설 제한:**

- 주석에는 source SHA, 동일 상대 식별 근거, 원본 프레임별 `visible/occluded/absent/uncertain`, 원본 좌표만 둔다. AI 검수와 사람 검수 구분을 유지한다. 다른 시점으로 box를 복제하거나 가림 상태의 위치를 추정하지 않는다.
- 추론 입력 manifest에 접촉·진입 번호, 정답 구간, `already_in_lane`, 정답 후보 여부를 넣지 않는다. 정답 전후 프레임만 골라 표시하거나 색·크기를 바꾸면 시간 라벨 누설이다. 모델 예측을 보고 box를 고치지 않는다.
- 모든 실제 제시 후보를 점검하며 같은 상대가 확실히 보이는 곳만 같은 스타일로 표시한다. 상단 차체 안의 얇은 노란 선으로 제한하고 바퀴·차선 경계·접촉면·기존 번호를 덮지 않는다. 정확 색·두께·좌표 변환은 후보 출력 전에 한 번 동결한다. 해상도 손실·왜곡·경계 가림이 발견되면 추론 전에 수정하고 재검수한다.
- 상대를 특정할 수 없거나 필요한 후보의 표식을 검수할 수 없으면 적격성 미달로 기록한다. 조건부 00028은 조건부 관측 진단만 가능하며 확정 진입 정확도에 포함하지 않는다.

**실행·판정:**

1. 새 자료의 사건 단위 분리, 동일 상대·진입 참조, 표식과 해시를 예측 전에 동결한다. 시작부터 진입한 적격 사례와 영상 중 진입하는 적격 대조가 각각 최소 1건은 있어야 두 상태의 구별에 관한 제한적 주장을 할 수 있다. 이는 통계적 표본 충분성 기준이 아니다.
2. 표식 없는 시트가 원래 저장된 RGB 해시·질문·후보를 재현하는지 먼저 확인한다. 대조군과 표식군의 크기·후보·프롬프트는 같고 선언한 표식 픽셀만 달라야 한다. 추가 reference 이미지는 이미지당 예산을 절반으로 바꿀 수 있으므로 금지한다.
3. 실제 효과를 실행할 때는 적격 clip마다 원래 Q3와 표식 Q3 각 한 번만 평가한다. 나머지 세 답변은 캐시를 쓴다. 이는 두 번의 완전한 four-call 추론이 아니며 실측 호출수를 그대로 기록한다. 이번 검토의 실제 모델 호출은 0회다.
4. 변경 가능한 제출 필드는 `entry_frame` 하나다. 유효 후보 정수 여부, 원응답, snap/fallback, source/시트 해시, PTS 오차, 다른 3개 출력 동일성을 모두 기록한다. 무효 응답이 첫 프레임으로 fallback한 사례는 already-inside 성공으로 세지 않는다.
5. 개발의 제한적 신호 조건은 `entry gained ≥ 1`, `lost = 0`, `MAE 비증가`, 알려진 후발 진입 사례의 새 첫 프레임 오선택 0건이다. unknown과 조건부 사고 상대는 정확도 분모에서 제외한다. 후보 누락 4건은 고정된 구조적 실패로 함께 남긴다.
6. 근거·입력 계약 오류는 실행 무효, 유효 실행의 gate 실패는 이 후보 종료다. 색·두께·문구·창·fallback을 결과에 맞춰 재탐색하지 않는다. 통과해도 표식 보조 진단의 개발 신호만 인정한다. 자동 상대 검출 성능, 새 독립 자료 성능, 공식 S2 상승은 별도로 입증해야 한다.

현재 표식 manifest와 적격 새 진입 참조가 확보됐다는 증거는 이 검토에서 확인하지 못했다. 자동 상대 지정을 첫 질의의 추가 bbox 출력으로 대체하면 Q1 과제·후속 참조·파서까지 함께 바뀌며 과거 복합 계획 실패도 다시 고려해야 한다. 따라서 이를 같은 단일 변경의 자동 배포판이라고 부르지 않는다.

구조화 규약·20건 실제 응답·근거 SHA: [next_policy_review.json](/Users/hyeongi/projects/blackbox-accident-ai/artifacts/stage2_goal_20260920/next_policy_review.json).
