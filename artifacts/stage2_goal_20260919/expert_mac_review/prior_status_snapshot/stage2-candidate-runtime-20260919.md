# Stage2 단일 후보의 실제 실행 경로 검증

2026-09-19 후속 실행. **보조합 제거 후보를 실행 가능한 모듈로 분리하고,14건 재생·진입 함수 계약·1건 실제 Mac 추론을 통과했다.** 새 가중치/시간창 후보는 만들지 않았으며 기존 V6/V7 제출 자산도 변경하지 않았다. 독립 정답과 Linux/CUDA 실행은 여전히 미완료다.

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

**완료되지 않은 조건:** 새로운 검수 양식6개는 여전히 `unfilled_human_review_template`/unknown이고 평가 부적격이다. 관측 가능한 같은 상대·접촉·진입 정답, 사고 단위 독립 확인 분리, 접근 가능한 Linux/CUDA 자원이 필요하다. 해당 자료/서버 정보 요청은 기존에 전달했으며, 이번 후속 실행에서 새 응답이나 검수 정답을 확보하지 못했다. 이 조건의 미충족이 세 goal 턴 연속 확인돼 goal은 blocked 상태다.
