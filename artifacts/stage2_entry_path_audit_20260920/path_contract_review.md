# 제출 경로 감사 및 단일 후보 검토

판정: **경로 감사 PASS. 후보는 조건부 타당성 검토이며 아직 선택·실행하지 않았다.** 신규 모델 호출 0회. 기존 제출·검수·동결 파일은 수정하지 않았다.

- V6/V7 source와 verify_v6/verify_v7의 Stage2 실행 모듈 6개는 실제 파일 SHA256이 모두 같다. VLM 자산 20개는 두 제출 manifest의 크기·SHA256이 같다. 큰 모델 파일 자체는 이번에 다시 해시하지 않았다.
- V7 실제 진입점은 보존된 V6c다. HANDOFF의 마지막 V6는 이전 시점 기록이며, 이후 V7 접수 92730 문서가 있다. 현재 서버 상태는 조회하지 않았다.
- 원본 파일 번호로 정렬한 유효 입력 → Q1 최대 10장 → Q2 접촉 선택 → Q3 첫 프레임부터 내부 Q2 접촉까지 **포함**하여 최대 12장 → 원본 번호 파서 선택이다. 샘플링은 경로 인덱스 기준이며 PTS가 아니다.
- 최종 motion 접촉 교체는 Q3/Q4 이후 collision 열만 바꾼다. 진입 후보와 공간 문맥은 내부 Q2 접촉에 남는다. Q3는 무표식 시트이며 각 질문은 새 단일 user 메시지라 Q1/Q2의 응답이나 상대 식별 이력을 받지 않는다.
- 파서는 후보 안 숫자 그대로, 후보 밖 숫자 최근접 보정, 해석 실패 첫 후보 선택이다. 최근 별도 공간 진단의 엄격 JSON 파서와 다르다.
- Mac worker는 verify_v6 정책을 사용하지만 MLX/Metal 모델이다. 반드시 baseline_prediction을 사용한다. candidate_prediction은 별도 접촉 변경 연구 결과다. 영상마다 모델을 새로 로드하는 Mac의 시간은 제출의 일괄 모델 재사용 시간과 다르다.
- 부모가 생성한 path_replay.json과 구현을 읽었다: 24개·96개 캐시 응답 재생 PASS, 선택 8개 raw 진입 번호 모두 제시 후보, 파서 보정/실패 0. 이번 검수자가 해당 재생을 다시 실행하지는 않았다. 정책·질문·이미지·출력 재생은 CUDA 모델의 수치 동일성이나 새 점수를 증명하지 않는다.
- 기존 **5→5→5→0**은 완료된 진단의 재사용이다. 넓은 참조 구간의 모든 시점에 각각 맞는 후보가 존재하는지와, 후보 하나가 구간 전체에 맞는지는 별도로 기록해야 한다.

## 제안의 타당성과 필수 조건

자동 Q2 접촉 문맥과 무표식 확대 첫 프레임을 **한 질의**에 제공하고, 같은 상대의 시작 상태가 엄격 INSIDE일 때만 첫 원본 프레임으로 교체하는 정책은 첫 프레임 단독 질문보다 상대 연결 계약이 분명하다. 그러나 Q2 접촉도 추정치이며, 실제 같은 차량을 연결했는지는 별도 증거가 필요하다. 상대·바퀴·경계가 불명확하면 UNCERTAIN, 잘못된 형식도 기준 유지로 처리해야 한다. 전문가 표식·라벨 적격성·수동 ID 선택을 호출 조건에 사용하면 자동 제출 후보가 아니다.

**입력 크기 주의:** 현 ask는 이미지 수로 1.2M 픽셀을 나눈다. 별도 4장 입력은 첫 프레임을 300k 예산으로 축소한다. 가능한 배치 제안은 단일 1536×768 시트에 왼쪽 1152×768 첫 프레임, 오른쪽 384×256 접촉 문맥 3장이다. 총 1,179,648픽셀·32배수로 확대 크기를 보존할 수 있다. 아직 렌더링·질의하지 않았다. 문맥 anchor는 raw Q2 숫자가 아니라 기존 _choice로 확정한 유효 경로 인덱스를 사용한다.

이 후보는 과거 temporal_v1의 접촉 문맥 가설을 재사용하지만, 전체 영상 시간 재탐색과 3회 세분화를 하지 않는 시작 상태 보정이다. 별도 정책으로 검증할 수 있으나, 문맥·크기·배치·질문의 기제별 효과를 분리했다고 말할 수 없다. 영상 중 진입 오류는 고치지 못하며 거짓 첫 프레임 교체 위험이 있다. 고정 자료의 known OUTSIDE 회귀와 미상 응답을 포함하고, 노출 자료 개선을 독립 일반화나 공식 점수로 승격하지 않는다.

## 근거

- entrypoint: [releases/v7/source/inference.py](/Users/hyeongi/projects/blackbox-accident-ai/releases/v7/source/inference.py:18)
- input_and_loader: [releases/v7/source/model/stage2/code/solution/stage2_uncapped_jerk_v6c.py](/Users/hyeongi/projects/blackbox-accident-ai/releases/v7/source/model/stage2/code/solution/stage2_uncapped_jerk_v6c.py:102)
- four_query_policy: [releases/v7/source/model/stage2/code/solution/stage2_v2.py](/Users/hyeongi/projects/blackbox-accident-ai/releases/v7/source/model/stage2/code/solution/stage2_v2.py:72)
- entry_candidates: [releases/v7/source/model/stage2/code/solution/stage2_v2.py](/Users/hyeongi/projects/blackbox-accident-ai/releases/v7/source/model/stage2/code/solution/stage2_v2.py:104)
- collision_only_override: [releases/v7/source/model/stage2/code/solution/stage2_uncapped_jerk_v6c.py](/Users/hyeongi/projects/blackbox-accident-ai/releases/v7/source/model/stage2/code/solution/stage2_uncapped_jerk_v6c.py:83)
- candidate_parser: [releases/v7/source/model/stage2/code/solution/stage2.py](/Users/hyeongi/projects/blackbox-accident-ai/releases/v7/source/model/stage2/code/solution/stage2.py:150)
- cuda_preprocess_stateless_chat: [releases/v7/source/model/stage2/code/solution/vlm.py](/Users/hyeongi/projects/blackbox-accident-ai/releases/v7/source/model/stage2/code/solution/vlm.py:26)
- cuda_nf4: [releases/v7/source/model/stage2/code/solution/vlm_candidate.py](/Users/hyeongi/projects/blackbox-accident-ai/releases/v7/source/model/stage2/code/solution/vlm_candidate.py:20)
- mac_worker: [artifacts/stage2_goal_20260919/expert_mac_review/run_paired_mac.py](/Users/hyeongi/projects/blackbox-accident-ai/artifacts/stage2_goal_20260919/expert_mac_review/run_paired_mac.py:47)
- mac_preprocess: [scripts/mac/mlx_stage2.py](/Users/hyeongi/projects/blackbox-accident-ai/scripts/mac/mlx_stage2.py:45)
- pts_and_backend_limit: [docs/mac-stage2-validation-contract.md](/Users/hyeongi/projects/blackbox-accident-ai/docs/mac-stage2-validation-contract.md:5)
- old_funnel: [docs/stage2-goal-results-20260920.md](/Users/hyeongi/projects/blackbox-accident-ai/docs/stage2-goal-results-20260920.md:12)
- v7_submission_record: [docs/v7-submission-validation-20260918.md](/Users/hyeongi/projects/blackbox-accident-ai/docs/v7-submission-validation-20260918.md:7)
- prior_temporal_policy: [scripts/mac/temporal_entry.py](/Users/hyeongi/projects/blackbox-accident-ai/scripts/mac/temporal_entry.py:5)
- prior_temporal_failure: [docs/mac-priority01-results-20260916.md](/Users/hyeongi/projects/blackbox-accident-ai/docs/mac-priority01-results-20260916.md:29)
- current_replay_implementation: [artifacts/stage2_entry_path_audit_20260920/replay_paths.py](/Users/hyeongi/projects/blackbox-accident-ai/artifacts/stage2_entry_path_audit_20260920/replay_paths.py:23)
