# Stage2 단일 후보의 실제 실행 경로 검증

> 2026-09-20 후속 검증 완료: 새 CCD 12건·전문가 검수·Mac 48회 실행 결과, 기존 단일 후보를 미채택으로 종료했다. 합의한 준비·진단·검증 작업은 완료했고 점수 상승은 미입증이다. [최신 결과](/Users/hyeongi/projects/blackbox-accident-ai/docs/stage2-goal-results-20260920.md). 아래는 당시 기록이다.

2026-09-19 후속 실행. **보조합 제거 후보를 실행 가능한 모듈로 분리하고,14건 재생·진입 함수 계약·1건 실제 Mac 추론을 통과했다.** 새 가중치/시간창 후보는 만들지 않았으며 기존 V6/V7 제출 자산도 변경하지 않았다. 이 기록 이후 사용자가 전문가 에이전트 검수와 Mac 실행을 지정했으므로 사람 검수·Linux/CUDA 확보를 후속 작업의 필수 조건으로 요구하지 않는다.

최신 후속 실행은 새6건·24질의로 확대했다. 모두 exit0, 네 출력은 기준선과 동일, worker 시간 합계243.313초·MLX peak 최대6.092GB다. 시각 검수와 별도 실행 검증도 마쳤다. [후속 검수·Mac 결과](/Users/hyeongi/projects/blackbox-accident-ai/docs/stage2-expert-mac-results-20260919.md). 아래 표는 그 이전 실행 기록이다.

| 검사 | 실제 수행 범위 | 결과 |
|---|---|---|
| 모델 자산 확인 | 기존 MLX16파일 SHA 재검사 | 모두 일치 |
| 후보의 네 질문 입력 보존 | 공개5+사람초안9, 저장 응답56개 재생 | 질문·토큰예산·RGB 해시/크기 일치, 후보14건 출력이 기존 대조와 동일 |
| 제출 함수 형태 | 실제 `predict_stage2` 폴더 순회·DataFrame·trace 저장, 광류/VLM은 고정 캐시로 대체 |14건 ID/열/범주/정수/원본 번호 통과 |
| 실제 모델 실행 | 기존 개발00007 한 건, 원본 전부 광류 재계산,4회 MLX 새 추론 | exit0; 원응답·processor 입력·시트·token trace 동일 |
| 후보의 CUDA 실행 진입 | 현재 Mac에서 수정된 서버 검사기 호출 | exit2, CUDA 없음, 모델 추론0 |

실제 모델 실행 결과는 `collision_frame=630, entry_frame=172, entry_side=LEFT, evasion_space=0`이다. 같은 실행의 보존 기준선은 접촉330이며 다른 세 값은 같다. 검사 wall time43.021초, Python socket 연결 시도0이다. 입력 해시 확인 등 검사 시간을 포함하므로 과거 worker 시간과 직접 속도 비교하지 않는다. 한 사례 실행이 새로운14건 정확도 평가나 전체 서버 자원 시험을 대신하지 않는다.

후보는 기존 특징 추출·네 질문·기준선 후처리를 그대로 실행하고 마지막 접촉 선택에서만 기존 보조합을 제거한다. 기존 기준선 예측을 trace에 함께 남긴다. 보조합 제거 규칙, 초기 jerk 처리, 정규화, 동률 처리는 전 턴의 사전 규약과 같다. 기존 대조의 사람4/9→5/9·공개4/5 유지 수치를 이번에 새로운 표본의 성과로 다시 집계하지 않는다.

코드와 증거:

- [실행 가능한 후보](/Users/hyeongi/projects/blackbox-accident-ai/artifacts/stage2_goal_20260919/candidate_runtime/candidate.py)
- [실행 전104파일 동결](/Users/hyeongi/projects/blackbox-accident-ai/artifacts/stage2_goal_20260919/candidate_runtime/freeze.json)
- [14건 질의 재생 검사](/Users/hyeongi/projects/blackbox-accident-ai/artifacts/stage2_goal_20260919/candidate_runtime/replay/report.json)
- [DataFrame 진입 함수 검사](/Users/hyeongi/projects/blackbox-accident-ai/artifacts/stage2_goal_20260919/candidate_runtime/entrypoint_replay/report.json)
- [실제 Mac 모델 실행](/Users/hyeongi/projects/blackbox-accident-ai/artifacts/stage2_goal_20260919/candidate_runtime/live/report.json)
- [실제 네 질의·예측·진단](/Users/hyeongi/projects/blackbox-accident-ai/artifacts/stage2_goal_20260919/candidate_runtime/live/result.json)

Linux에서는 [기존 검사 명령](/Users/hyeongi/projects/blackbox-accident-ai/artifacts/stage2_goal_20260919/server_audit/PROBE.md)에 다음 두 인수를 더하면 동일 후보를 실행한다. 기준선과 후보는 서로 다른 새 출력 폴더를 지정한다. `<repo>`는 옮긴 저장소의 실제 절대 경로다.

```sh
--candidate <repo>/artifacts/stage2_goal_20260919/candidate_runtime/candidate.py \
--candidate-freeze <repo>/artifacts/stage2_goal_20260919/candidate_runtime/freeze.json
```

검사기는 보존V7 Stage2 모델·코드 SHA와 후보 동결 SHA를 검사하고 후보를 별도 모듈로 로드한다. 기존 제출물은 수정하지 않는다. 후보 trace에 기준선의 네 출력도 남기지만, 실제 CUDA에서 기준선·후보를 각각 실행해 비교했다는 기록은 아직 없다. 이전 기준선 전용 검사기는 `stage2_probe_baseline_v1.py`로 보존했다. 과거 `probe_checks.json`은 이전 버전의 검사이며 새 후보의 CUDA 성공 증거가 아니다.

**최초 기록 당시의 미완료 조건:** 검수 양식6개는 `unfilled_human_review_template` 상태였고, 독립 정답·Linux/CUDA 실행을 확보하지 못해 goal을 blocked로 기록했다. 이 상태는 역사 기록이며, 이후 사용자 지시로 전문가 에이전트가 별도 AI 검수 기록을 작성하고 Mac에서 동일 후보를 실행하는 작업을 재개했다. 빈 사람 양식을 AI 검수 완료로 바꾸지 않으며 두 출처는 별도로 보존한다.
