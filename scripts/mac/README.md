# Mac Stage2 실행 경로

`run_stage2.py`가 단일/여러 영상/고정 manifest 실행의 공통 진입점이다. 모든 영상은 동일한 환경 변수·스레드·seed·import/모델 로드 순서를 사용하며 새 작업 프로세스에서 처리한다. 모델 재사용에 의한 파일 간 상태 잔류를 피하는 대신 영상마다 모델 로드 비용이 발생한다. 속도 최적화 및 Linux 제출용 구현은 별도다.

프로젝트 루트에서:

```sh
artifacts/mac_experiments/stage2_mlx/.venv/bin/python scripts/mac/run_stage2.py \
  --case S2_001 --output-dir artifacts/mac_experiments/runs/single_new

artifacts/mac_experiments/stage2_mlx/.venv/bin/python scripts/mac/run_stage2.py \
  --case S2_001 S2_002 --output-dir artifacts/mac_experiments/runs/batch_new

artifacts/mac_experiments/stage2_mlx/.venv/bin/python scripts/mac/run_stage2.py \
  --manifest artifacts/mac_experiments/baseline_20260916/inputs.json \
  --output-dir artifacts/mac_experiments/runs/baseline_new
```

기존 `artifacts/mac_experiments/stage2_mlx/run.sh`도 이 진입점으로 연결된다. 새 출력 경로가 필수다. 결과는 `<output>/stage2/<group>/<ID>/result.json`, `predictions.csv`, `calls.json`, `motion.npz` 및 루트 `stage2_report.json`에 저장한다. 기존 baseline_20260916 및 snapshot은 과거 기록으로 보존하고 다시 실행하거나 수정하지 않는다. `--manifest`는 예측 입력 경로/해시/원본 프레임 번호만 사용한다.

모델 전처리 후 텐서를 평가·동기화한 뒤 선택한 생성 경로에 전달하며 이미지, 채팅 프롬프트, 처리된 텐서 해시를 기록한다. 동일 경로가 GPU 출력의 비트 동일성을 보장하지는 않는다. `check_run_equivalence.py`는 두 실행의 공통 사례에서 실제 입력·원시 응답·최종 예측을 비교하고 차이가 있으면 exit 1을 반환한다.

```sh
artifacts/mac_experiments/scipy_compat/.venv/bin/python scripts/mac/check_run_equivalence.py \
  artifacts/mac_experiments/runs/single_new artifacts/mac_experiments/runs/batch_new \
  --output artifacts/mac_experiments/runs/comparison_new.json
```

필수 로컬 자산: Mac MLX 환경/모델과 `artifacts/submissions/verify_v6/model/stage2/code`. 모델 설치나 다운로드는 이 실행기가 수행하지 않는다. 모델의 반환값·질문·정책은 기존 V6 기반 Mac 실험 정책이며, CUDA NF4와 동일 모델 수치라고 간주하지 않는다.

## 2026-09-16 재현성 수정

기본 실행은 `--decode-mode sync --compute-dtype native --deepstack-fix --policy baseline`이다. `deepstack_fix.py`는 설치된 mlx-vlm 0.3.4의 두 오류를 프로세스 안에서만 교정한다.

- `[batch, sequence]` 이미지 마스크에서 배치 인덱스를 토큰 위치로 사용하는 오류: 실제 `(batch, sequence)` 위치에 특징을 더한다.
- 다중 이미지 특징을 개별 길이로 잘못 분할한 뒤 합치는 오류: 이미 순서대로 연결된 특징을 그대로 사용한다.

패키지 파일과 모델 가중치는 변경하지 않는다. mlx-vlm 버전이 바뀌면 수정본 적용을 중단하고 재감사를 요구한다. 동기 디코더는 생성 토큰별 전체 logits 해시와 상위 두 토큰 차이를 기록한다. 원래 경로 대조는 `--no-deepstack-fix --decode-mode legacy`로 명시한다. `--compute-dtype float32`는 실패한 원인 분리용 옵션이며 기본값이 아니다.

`--policy temporal_v1`은 실험 전용이다. 기존 네 번의 질의 결과를 `baseline_prediction`으로 저장한 후, 접촉 상대 문맥과 전체 영상 후보에서 진입 시점을 찾고 두 번 세분화한다. 최대 일곱 번 질의하며 다른 세 필드는 바꾸지 않는다. 개발 점수가 좋아도 독립 검증 전 제출 정책으로 승격하지 않는다.

새 실험 및 반복성 근거는 `artifacts/mac_experiments/priority01_20260916/`에 있다. 과거의 반복성 미통과 기록은 당시 상태의 증거로 보존한다.
