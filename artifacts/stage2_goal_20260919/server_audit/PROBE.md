# Linux/CUDA Stage2 실행 준비

`stage2_probe.py`는 보존된 V7의 `inference.predict_stage2`만 실행한다. 신규 모델·의존성 다운로드, 다른 Stage 실행, 원격 접속 기능은 없다. 실제 Linux/CUDA 실행은 아직 하지 않았다.

Linux에 동결 증거·입력 이미지·V7 추출 패키지를 같은 저장소 상대 구조로 옮기고, 대회와 호환되는 CUDA/Transformers/accelerate/bitsandbytes 런타임에서 실행한다. `<repo>`와 `<new-output>`을 실제 절대 경로로 바꾼다. 출력 폴더가 이미 있으면 거부한다.

```sh
python3 -B <repo>/artifacts/stage2_goal_20260919/server_audit/stage2_probe.py \
  --repo-root <repo> \
  --package <repo>/artifacts/submissions/verify_v7 \
  --freeze-dir <repo>/artifacts/stage2_goal_20260919/core_v2 \
  --readiness <repo>/artifacts/stage2_goal_20260919/server_audit/readiness.json \
  --output <new-output>
```

CUDA guard를 먼저 통과한 뒤 다음을 검사한다.

- core freeze 전체 파일 SHA와 manifest·readiness의 바인딩.
- V7 Stage2 소스·config·두 NF4 가중치까지 실제 SHA를 readiness의 기대값과 대조. 패키지 SHA 검사 시간은 `package_asset_hash_seconds`, 전체 사전 확인 시간은 `preflight_seconds`로 추론과 분리한다.
- manifest에 있는 이미지 바이트·원본 번호·중복·순서를 확인하고 임시 `data/images/<ID>`에 원본을 가리키는 symlink만 만든다. PTS/정답은 추론 함수에 전달하지 않는다.
- 오프라인 환경변수 및 Python socket 연결 차단. 모델 로더는 보존된 로컬 NF4 로더를 사용한다.
- 네 출력 필드, ID 전체 일치·중복·결측, 방향 범주, 정수 공간 0/1, 두 시점의 실존 원본 번호를 검사한다. 기존 `scripts/verify_submission.py:49-57`의 계약을 작은 독립 함수로 재사용했다.

산출물은 `input_mapping.json`, 유효한 전체 출력 `stage2.csv`, `report.json`이다. Python 예외도 report에 남기고 nonzero로 종료한다. 실제 추론의 시간, CUDA peak allocated/reserved, 네트워크 연결 시도, 패키지 metadata 변경 여부를 기록한다. 임시 symlink 폴더는 종료 시 제거하며 원본 이미지는 변경하지 않는다.

**Mac 확인:** 최종 스크립트 실행은 `mac_cuda_guard_v2/report.json`에 보존했다. 종료 코드 2, `NOT_EXECUTED_CUDA_UNAVAILABLE`, 모델 추론 시작 false, 연결 시도 0이다. CUDA guard 뒤의 해시/모델 경로는 실행하지 않았으므로 이를 Linux 검증 통과라고 부르면 안 된다. 최초 버전의 guard 결과도 별도 `mac_cuda_guard/`에 보존했다.

`probe_checks.json`은 가짜 DataFrame을 이용한 정상 계약 및 잘못된 번호/ID/범주/실수 프레임/중복 ID 거부 검사 결과다. 모델 정확도 시험이 아니다. Linux에서 성공해도 상태 이름은 `PASS_OUTPUT_CONTRACT_ONLY`이며 공식 S2·일반화 성능·숨은 전체 평가의 60분 통과를 뜻하지 않는다.

## 동결 후보 실행 선택

보조합 제거 연구 후보를 실행할 때만 위 명령에 다음 인수를 추가한다. 기준선과 다른 새 output을 사용한다.

```sh
--candidate <repo>/artifacts/stage2_goal_20260919/candidate_runtime/candidate.py \
--candidate-freeze <repo>/artifacts/stage2_goal_20260919/candidate_runtime/freeze.json
```

후보는 별도 모듈로 로드하며 제출 폴더를 수정하지 않는다. 후보 실행의 `candidate_traces/`에는 같은 네 질의의 기준선 예측과 후보 예측을 남긴다. 후보 없는 기본 실행은 보존V7이다. 두 인수를 하나만 제공하면 거부한다. 후보 모듈과 관련 증거의 해시가 runtime freeze와 일치해야 한다.

이전 검사기 소스는 `stage2_probe_baseline_v1.py`로 보존했다. 기존 `probe_checks.json`은 그 버전에 대한 기록이다. 새 후보 옵션의 Mac 실행은 `../candidate_runtime/mac_cuda_guard/report.json`처럼 CUDA 미지원으로 모델 실행 없이 exit2 종료했다. 후보 구현의14건 입력 재생·DataFrame 계약·1건 실제 MLX 실행은 [별도 보고서](/Users/hyeongi/projects/blackbox-accident-ai/docs/stage2-candidate-runtime-20260919.md)에 있다. 이들은 실제 CUDA 분기의 통과 기록이 아니다.
