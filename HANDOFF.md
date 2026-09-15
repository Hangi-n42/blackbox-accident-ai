# Project Handoff

작성 기준: 2026-09-16, Windows 작업 디렉터리 실사. 대상 MacBook은 **M4 Max / 36GB**. 이전 대화 없이 이 문서부터 읽고 `research/v7/research_decision_20260915.md`와 `research/v7/native_video_result_review.md`를 읽으십시오.

## 1. 프로젝트 목적

DACON 236753 블랙박스 영상 기반 지능형 고의사고 분석 대회 연구. Stage1은 재촬영 판별, Stage2는 사고 상대의 최초 접촉·진입 시점·진입 방향·회피 공간, Stage3는 가감속·조향 분류입니다. 대회 조건은 `대회_통합_정보.md`, 실행 계약은 `SUBMISSION_GUIDE.md`에 있습니다.

기록된 평가식: 총점 0.2 S1 + 0.4 S2 + 0.4 S3. S1 Macro F1. S2는 접촉 ±0.3초 정확도 0.35 + 진입 ±0.3초 정확도 0.35 + 방향 Macro F1 0.15 + 공간 Macro F1 0.15. S3는 가감속 Macro F1 0.7 + 조향 Macro F1 0.3이며 GT STOPPED 구간의 조향 평가 마스킹과 전체 행 출력 계약을 구분해야 합니다. 제출 전 공식 최신 규정도 재확인하십시오.

## 2. 현재 구현 상태

마지막 실제 제출은 **V6**, 제출 번호 90342. 사용자 제공 2026-09-15 결과 화면: Stage1 0.5078837545, Stage2 0.2817803332, Stage3 0.5266169416, 42분 41초. V5 Stage2 0.2281306982에서 개선됐으나 절대 성능은 낮습니다. 저장된 초기 제출 receipt는 대기 상태일 수 있습니다. 위 결과는 화면 근거이며 이번 이전에서 새 제출은 하지 않았습니다.

V6는 V5의 최종 접촉 jerk 점수 상한만 제거했습니다. 4B Qwen NF4 네 번 질문, Stage1/3 가중치와 나머지 판단 구조를 유지했습니다. 정확한 제출 코드는 `releases/v6/source`, 모델 포함 원본 ZIP은 별도 이전 대상입니다. **루트 inference.py는 V6 진입점이 아닙니다.**

## 3. 현재 작업 중인 부분

V7의 여러 후보가 실패한 이유를 검증하던 상태입니다. 마지막 실험은 Qwen native-video 입력. 동일한 노출된 사람 검수 9개에서 접촉 정확도가 V6 4/9에서 후보 1/9로 악화되어 채택하지 않았습니다. V7 제출물은 생성·제출되지 않았고 유효한 개선안은 아직 없습니다. 이전 작업은 연구 구조·라벨·가중치를 변경하지 않습니다.

## 4. 아직 완료되지 않은 작업

- 새로운 독립 영상의 신뢰할 수 있는 최초 접촉·진입 정답 확보.
- Stage1 실제 도로 영상과 물리적 화면 재촬영 대응 자료 확보: 현재 확보 0쌍.
- Stage3 외부 proxy 라벨과 공식 의미의 일치 검증.
- Mac에서 모델 포함 Stage1/3 전체 추론 및 Stage2 별도 포팅 검증.
- 다음 제출은 독립 검증에서 목표 오류 감소와 다른 항목의 회귀를 확인한 후 결정. 개선 점수 보장은 없습니다.

## 5. 프로젝트 구조

| 경로 | 역할 |
|---|---|
| solution/ | 기존 추론 구현과 공통 특징 계산 |
| inference*.py, train_stage1.py | 역사적 실행·학습 진입점. V6와 혼동 금지 |
| scripts/ | 공개 예제 준비, 평가·제출 유틸리티 |
| scripts/migration/ | 안전한 V6 복원, 외부 자산 검증, 경로 대응, smoke·기록 재계산 |
| releases/v6/source/ | 제출 ZIP에서 추출한 25개 코드·라이선스 파일, 바이트 보존 |
| releases/v6/*.json | 원본 제출 47개 파일 및 소스 해시 |
| research/v6_stage2/, research/v7/ | 검수 정답, 프로토콜, 실패 실험과 근거 |
| research/v6_review_tool/dist/ | 실제 수작업 HTML/JS/CSS 소스. 생성물처럼 일괄 제외하면 안 됨 |
| Baseline/data/ | 공식 공개 자료, Git 제외 |
| model/, external_data/, artifacts/ | 모델·외부 자료·실험 산출물, Git 제외 |
| docs/migration/ | 환경 실사, macOS wheel 목록, 외부 파일 목록 |
| WORK_LOG.md | 누적 작업 기록 |

## 6. 실행 방법

처음 설치는 문서 마지막 Fresh macOS Setup 순서대로 진행합니다. 소스만으로 `smoke.py`, `replay_metrics.py`, 두 Node 검수 테스트를 실행할 수 있습니다. 모델 추론은 원본 ZIP과 공개 데이터를 별도로 복원해야 합니다. 검수 웹 서버는 Python 표준 HTTP 서버이며 Node 서버나 npm 설치가 필요하지 않습니다. 출력은 새 경로를 사용하십시오.

## 7. 개발환경

실사 결과 전체 버전은 `docs/migration/windows-environment.json`. Windows11 AMD64, Python 3.12.14, PyTorch 2.8.0+cu128 / torchvision 0.23.0+cu128, CUDA build 12.8, cuDNN numeric 91002, RTX2080 SUPER 8GiB / driver 595.95. transformers 4.57.6, accelerate 1.9.0, bitsandbytes 0.48.1. NumPy 1.26.4, sklearn 1.5.2, OpenCV headless 4.10.0.84, PyAV 16.0.1. Node 24.14.1은 검수 UI의 내장 모듈 테스트용. Git 2.53.0.windows.1.

macOS ARM64 Python 3.12 의존성은 `requirements/macos-arm64.txt`, 전이 의존성까지 42개 wheel SHA를 고정한 파일은 `requirements/macos-arm64.lock.txt`. Windows 원본 requirements는 보존했습니다. lock은 macOS14 ARM64 대상 PyPI 해석을 통과했습니다. 이는 native 실행 검증과 다릅니다. macOS15 ARM64 GitHub Actions smoke도 구성했습니다. 실제 실행 결과는 GitHub Actions에서 확인하십시오.

## 8. 환경변수

`.env.example`에는 값 없는 비밀키 대신 공개 실행 설정만 있습니다. 원본 코드에 dotenv 자동 로더는 없습니다. shell에서 명시적으로 source합니다.

- BLACKBOX_V6_DIR: migration/run_stage.py가 사용하는 복원 V6 디렉터리.
- HF_HUB_OFFLINE, TRANSFORMERS_OFFLINE: 다운로드 단계 0, 오프라인 추론 단계 1.
- HF_HUB_DISABLE_IMPLICIT_TOKEN: 공개 다운로드에 저장된 계정 토큰을 자동 사용하지 않음.
- OMP_NUM_THREADS, MKL_NUM_THREADS, OPENBLAS_NUM_THREADS: CPU 스레드 제한.
- 기존 Windows CUDA_PATH/CUDA_PATH_V12_8은 Mac에 옮기지 않습니다.

인증은 새 컴퓨터에서 별도 로그인하십시오. 기존 GitHub/Hugging Face 인증 파일은 이전 목록·Git에 포함하지 않습니다. 실제 .env 및 개인키 파일은 검사 범위에서 발견되지 않았습니다.

## 9. 데이터

현재 원본 루트는 `C:\Users\dsl\Desktop\Dacon\블랙박스`. 새 Mac의 clone 루트 아래 **동일 상대경로**에 배치합니다.

`docs/migration/external-assets.jsonl`은 Git에 제외된 모든 보존 대상 파일을 기록합니다. resume-evidence 22,357개 / 19,543,447,111바이트는 SHA256을 기록했습니다. archive-or-redownload 4,766개 / 82,244,747,758바이트는 크기만 기록하며 완전 무결성 증거는 아닙니다. 핵심 약 19.54GB와 선택 기록 약 82.24GB에 압축 해제 여유 공간을 추가하십시오. Windows env·캐시·다운로드 조각·로그·기기 로컬 hosting 상태·인증은 제외했습니다. 기존 파일을 삭제하지 않았습니다.

필수 보존 범주:

- Baseline/data/: Stage1 원본5+재촬영5 영상과 labels.csv, Stage2 5영상과 labels.csv, Stage3 OPEN001~005와 labels.csv.
- external_data/: 기존 학습용 외부 데이터. 하위 모든 경로는 자산 목록 기준 복사.
- model/: 커스텀 가중치와 provenance. 공개 모델 이름만으로 재현되지 않는 학습 결과 포함.
- research/ 아래 Git 제외 미디어·배열·학습 자산: 검수 프레임, 원본 Nexar 영상, 평가용 시각 대응과 외부 데이터.
- research/v6_review_tool/dist/cases.js 및 cases/: 검수 UI의 사례 목록·프레임. HTML/JS/CSS만 clone하면 영상은 없습니다.
- artifacts/submissions/submit_v6.zip: 정확한 마지막 제출 복원용.

개인 외장 SSD 등으로 자산 목록의 상대경로를 유지해 옮기십시오. 기존 목적지와 다른 파일을 무조건 덮어쓰지 말고 비교해야 합니다. `verify_assets.py`로 핵심 목록 전체를 검증합니다. 선택 보관 자료까지 옮길 경우 --all. 원본 Downloads 검수 JSON의 프로젝트 내부 사본은 research에 보존되어 있습니다. 브라우저 localStorage 자체는 이전하지 않았으므로 브라우저에만 남은 초안의 존재 여부는 별도 확인 필요입니다.

공개 입력은 `scripts/prepare_public_eval.py`, Stage3 10Hz 개발 입력은 `scripts/prepare_stage3_10hz.py`로 생성 가능합니다. 후자는 원본 CSV 시각 근거의 20Hz 프레임을 2개마다 취하며 원본은 유지합니다. **기존 출력 경로에는 재실행하지 마십시오.** 생성된 frame_mapping.json과 원본 프레임 식별자를 함께 보존합니다.

## 10. AI 모델

- V6 Stage1: FoundPAD/TPO + OpenAI CLIP ViT-B/16 기반 병합 가중치 및 forensic 결합. `model/stage1/tpo/asset_manifest.json` 및 별도 이전 V6 ZIP의 notices/provenance가 기준. 재학습 대체 금지.
- V6 Stage2: Qwen/Qwen3-VL-4B-Instruct, 원본 revision `ebb281ec70b05090aa6165b016eac8ec08e71b17`. raw 원본 위치 artifacts/candidates/qwen3_vl_4b. 실제 제출은 자체 저장 NF4로 **원본 HF 다운로드와 동일 파일이 아닙니다**. ZIP을 복원해야 합니다.
- V6 Stage3: 자체 학습 sklearn/joblib 모델. 공식 정답과 완전히 검증되지 않은 외부 proxy 라벨 사용. ZIP 가중치 그대로 복사.
- 연구 전용 Qwen/Qwen3-VL-8B-Instruct revision `0c351dd01ed87e9c1b53cbc748cba10e6187ff3b`, artifacts/candidates/qwen3_vl_8b_v7, 약 17.55GB. 실패 후보이며 V6 실행에는 불필요.
- 역사적 2B 모델은 artifacts/model/stage2/vlm_provenance.json을 포함한 archive 대상 기록이 기준. 현재 채택 모델이 아닙니다.

가중치는 Git 미포함. V6 ZIP은 2,914,590,320바이트, SHA256 `860f9d08d6013f090ae4bc710bdbbfeee06e01a0cd71f8a784b5689cdb4b7ee0`. 해제 47파일 3,424,385,695바이트. `bootstrap_v6.py --zip ...`가 ZIP과 각 파일 해시를 확인하며 다른 기존 파일은 덮어쓰지 않습니다.

raw 4B를 새 연구용으로 다시 받을 때만 다음 명령을 사용합니다(오프라인 설정 해제 필요). V6 복원 대체가 아닙니다.

```bash
python -c 'from huggingface_hub import snapshot_download; snapshot_download(repo_id="Qwen/Qwen3-VL-4B-Instruct", revision="ebb281ec70b05090aa6165b016eac8ec08e71b17", local_dir="artifacts/candidates/qwen3_vl_4b", token=False)'
```

## 11. 데이터베이스

DB 파일·서버·migration 설정은 발견되지 않았습니다. DB 초기화나 dump/restore 단계는 없습니다. 검수 기록은 JSON이며 브라우저 초안 저장은 localStorage입니다.

## 12. 현재 Git 상태

이전 시작 시 로컬 Git 저장소가 없었습니다. 원격 공개 main에 Initial commit `f8fd7dcaff20105e768bb8616d3b05b78f837769`와 .gitignore/LICENSE만 있었습니다. 원격 이력을 로컬 main의 부모로 보존하여 그 위에 이전 준비 커밋을 추가합니다. 기존 개발 파일은 삭제·되돌리지 않았습니다.

GitHub: https://github.com/Hangi-n42/blackbox-accident-ai . branch main. 현재 최종 커밋은 `git rev-parse HEAD`, upstream은 `git rev-parse --abbrev-ref '@{upstream}'`으로 확인합니다. 문서 자신이 포함된 커밋 해시를 문서 내부에 고정하지 않습니다. force push/history rewrite 금지.

## 13. 중요한 설계 결정

- 역사적 JSON·프로토콜·소스는 바이트 해시로 연결됩니다. Windows 경로가 보이더라도 원본 기록 일괄 치환 금지. 새 migration/paths.py는 읽을 때만 현재 clone 루트로 대응하고 프로젝트 밖 경로는 거부합니다.
- 실제 프레임/PTS와 VLM 제시 FPS를 구분. Stage2 최초 접촉과 진입은 같은 상대 차량을 추적하며 진입 방향은 화면 좌우 기준입니다.
- 사람 검수 9개는 이미 여러 차례 튜닝에 노출된 개발 자료입니다. 독립 평가라고 부르지 않습니다.
- 새 AI 검수 6개는 사람·공식 정답이 아닙니다. Nexar positive/time_of_event도 최초 접촉 정답과 같지 않습니다.
- 효율 개선·JSON 유효성·메모리 감소는 정확도 개선 증거가 아닙니다. 실패한 후보를 제출 개선안으로 바꾸지 않습니다.
- 원본 Windows CUDA 환경과 제출 서버 Linux L40S 환경, Mac 연구 환경은 구분합니다.

## 14. 알려진 문제

Stage1 실재촬영 도로 대응 데이터 부족. Stage2 후보에 정답이 빠지거나 후보 안에서 선택 실패. Stage3 proxy 라벨의 의미 불확실. public 예제가 적어 일반화 보장 불가.

V7 native video: 최종 접촉 4/9→1/9; 내부 VLM 기준만 보면 0/9→1/9이지만 최종 기준에서 실패입니다. 후보 커버리지 5/9, 그중 선택 실패 4/5. 이전 맞힌 사례 중 00003/6/8은 native 후보에 정답 자체가 없었습니다. 다른 세 항목은 캐시를 복사했으므로 종단 간 신규 평가가 아닙니다.

8B/통합 grounding/space anchor/temporal CNN도 채택 기준 실패. Stage3 proxy 경로 6개 가중 점수 .604002→.602949. root WORK_LOG와 research/v7/factorial_summary.json 참조.

Mac CUDA NF4 추론은 현재 동작 불가. MPS 모델 일치 및 전체 영상 실행은 확인 필요. 원래 일부 연구 스크립트의 Windows 절대경로·hash gate는 자동 이식하지 않았습니다. 오래된 다운로드 링크의 유효성도 전체 재확인하지 않았습니다.

## 15. 다음 작업

1. 새 Mac 설치와 외부 파일 SHA 검증; source snapshot과 기록 재계산 통과 확인.
2. 정확한 V6 Stage1/3를 Mac CPU에서 공개 예제로 평가하여 기존 결과와 비교.
3. Stage2는 검증된 CUDA 호스트를 유지하거나 별도 Mac 포팅 후보를 만들고 V6와 동일 입력 비교. MPS로 단순 문자열 치환 금지.
4. 독립 사람/공식 정답 확보 후 오류 유형을 분리하고 단일 변경씩 검증.
5. 전체 단계 회귀·시간·오프라인 패키지 조건 통과 후 제출 결정.

## 16. 마지막 작업 맥락

최근 연구 주요 파일은 research/v7/solution/native_video_v7.py, native_video_protocol.json, run_native_video.py, assess_native_video.py 및 native_video_run/*.json입니다. 실제 temporal-patch 영상 입력을 시험했고 최종 정확도가 악화되었습니다. 관련 연구 판단은 research_decision_20260915.md, native_video_result_review.md, goal_completion_and_blocked_audit_20260915.md에 기록되어 있습니다.

이번 이전은 .gitignore/.gitattributes, macOS requirements, migration 도구, V6 소스 사본, README/HANDOFF와 CI를 추가했습니다. 기존 모델·추론 알고리즘은 수정하지 않았습니다. 로컬 기본 import/144특징/PyAV 3프레임 encode-decode/경로 대응, 기록 해시 및 4/9→1/9 재계산, 두 검수 UI 테스트를 통과했습니다. 전체 대회 추론·새 Mac GPU 실행은 이 검사에 포함되지 않습니다. 작업 중 메타데이터 생성기의 Windows 기본 cp949 디코딩 오류는 UTF-8 명시로 수정했습니다. 기존 연구 오류가 아닙니다.

## 17. 새 컴퓨터 이전 체크리스트

- [ ] main clone 후 문서와 실패 연구 기록 읽기
- [ ] uname -m이 arm64인지, macOS14 이상인지 확인
- [ ] 새 Python3.12 가상환경 및 hash lock 설치
- [ ] 소스 bootstrap / smoke / 기록 재계산 / UI 테스트
- [ ] 외부 핵심 19.54GB 상대경로 복사 및 verify_assets
- [ ] V6 ZIP 복원, 공개 입력 준비
- [ ] Stage1/3 CPU 결과 비교, Stage2 CUDA 실행 위치 결정
- [ ] 브라우저에만 있는 미수출 초안 여부 확인
- [ ] 새 실험을 새 출력 경로에서 시작, 원본 증거 보존

# Windows → macOS Migration

## 경로·파일시스템·줄바꿈

실제 Windows 절대경로 코드에는 research/v6_stage2/audit_user_review_001.py의 Downloads 경로와 research/v7/test_nf4_8b_io_guards.py의 테스트 fixture가 있습니다. 다수 기록 JSON/MD에도 원래 경로가 남아 있습니다. 새 경로 reader는 프로젝트 내부 기록만 안전하게 재매핑합니다. 과거 전체 runner를 자동으로 이식했다는 뜻은 아닙니다. 새 실행에는 pathlib 기반 migration 진입점을 사용하십시오.

실사에서 대소문자/Unicode 정규화 충돌·symlink·env 밖 Windows native binary는 발견되지 않았습니다. 역사적 파일 SHA를 보존하기 위해 .gitattributes는 기본 바이트 보존, 새 migration 텍스트와 .sh에 LF를 지정합니다. 기존 .ps1/.bat/.cmd 실행 스크립트는 발견되지 않았습니다. 새 shell script 설치나 chmod 단계는 필요 없습니다.

## ARM64·CUDA·MPS

Mac M4 Max36GB에서 NVIDIA CUDA는 사용할 수 없습니다. 기존 solution/vlm_candidate.py는 CUDA 유무 검사, CUDA device와 NF4 bitsandbytes0.48.1/FP16 compute를 사용합니다. 이 환경을 복사하지 않습니다. 현재 bitsandbytes 최신 문서의 Apple Silicon 지원 여부와 **이 프로젝트의 고정된 CUDA 구현**은 별개입니다.

Stage1은 CUDA 없으면 CPU, Stage3는 sklearn/OpenCV CPU 경로. torch MPS backend를 확인할 수는 있지만 기존 추론에 자동 적용하지 않았고 전체 연산 호환·수치 일치는 확인 필요입니다. 36GB 용량만으로 4B/8B 실행 성공이나 동일 점수를 보장하지 않습니다. 별도 MPS/FP16 포팅은 새 연구 변경이며 이전 범위를 벗어납니다.

공식 참고: https://huggingface.co/docs/bitsandbytes/main/en/installation , https://docs.pytorch.org/docs/stable/notes/mps.html . Mac 의존성 잠금은 ARM64 wheel 전용이며 Intel Mac에는 그대로 사용하지 않습니다. ONNX Runtime/TensorFlow/custom executable 사용은 발견되지 않았습니다.

## 외부 프로그램·환경·DB·Docker

Python OpenCV/PyAV wheel의 codec을 사용합니다. 현재 ffmpeg CLI는 PATH에 없으며 Java/Ollama/DB/Redis/Tesseract 의존성은 발견되지 않았습니다. Dockerfile/Compose도 없어 이미지 architecture나 Windows volume migration 대상이 없습니다. Node24는 두 JS 테스트에만 필요합니다. CUDA Toolkit/Windows .venv/.exe/.dll/.pyd를 복사하지 않습니다.

PowerShell `$env:NAME="value"`는 macOS에서 `export NAME="value"`. HF 캐시 전체를 Git에 올리지 말고 모델 ID/revision으로 새로 받거나 커스텀 가중치를 별도 복사하십시오.

| 항목 | 기존 Windows | 새 macOS | 조치 |
|---|---|---|---|
| Python | 3.12.14 AMD64 | Python3.12 ARM64, patch 설치 시 확인 | 재설치 |
| 가상환경 | Windows .venv 약8.46GB | 새 .venv | 재생성 |
| GPU backend | RTX2080 SUPER CUDA12.8 | M4 Max36GB CPU, MPS 미검증 | CUDA Stage2 별도 호스트/포팅 |
| 외부 프로그램 | Git2.53, Node24.14.1; FFmpeg CLI 없음 | Git, Node24, wheel codecs | 설치 |
| 데이터 경로 | C:/Users/dsl/Desktop/Dacon/블랙박스 | clone 아래 상대경로 | 외부 자산 복사 |
| 모델 | V6 NF4 및 custom 가중치 | 동일 파일 복원, 실행 호환 별도 | ZIP 복사 |
| DB | 발견 없음 | 없음 | 해당 없음 |
| Docker | 발견 없음 | 없음 | 해당 없음 |

# Fresh macOS Setup

Apple Silicon macOS14 이상 기준. Homebrew가 없다면 공식 https://brew.sh 설치 안내를 먼저 따르십시오. 다음은 Bash/zsh에서 실행하며 clone 루트에 머무릅니다.

```bash
uname -m
sw_vers
xcode-select --install  # Command Line Tools가 없을 때만
brew install git python@3.12 node@24
export PATH="$(brew --prefix node@24)/bin:$PATH"
git clone https://github.com/Hangi-n42/blackbox-accident-ai.git
cd blackbox-accident-ai
"$(brew --prefix python@3.12)/bin/python3.12" -m venv .venv
source .venv/bin/activate
python -m pip install --require-hashes -r requirements/macos-arm64.lock.txt
cp -n .env.example .env
set -a
source .env
set +a
python scripts/migration/bootstrap_v6.py
python scripts/migration/smoke.py
python scripts/migration/replay_metrics.py
node research/v6_review_tool/test_review.cjs
node research/v6_review_tool/test_frame_loading.cjs
```

다음 단계 전에 외부 SSD에서 자산 목록의 resume-evidence 파일들을 clone 아래 동일 상대경로로 옮기십시오. secrets, Windows env, native binary는 복사하지 않습니다.

```bash
python scripts/migration/verify_assets.py
python scripts/migration/bootstrap_v6.py --zip artifacts/submissions/submit_v6.zip
# 출력이 없을 때만 공개 예제 준비. 이미 있으면 기존 자료를 먼저 확인.
if [ ! -e artifacts/public_eval ]; then python scripts/prepare_public_eval.py; fi
if [ ! -e artifacts/public_eval_10hz ]; then python scripts/prepare_stage3_10hz.py; fi
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
python scripts/migration/run_stage.py --stage 1 --data artifacts/public_eval/stage1 --output artifacts/mac_first/stage1.csv
python scripts/migration/run_stage.py --stage 3 --data artifacts/public_eval_10hz/stage3 --output artifacts/mac_first/stage3.csv
# 검수 도구: 별도 터미널, 종료는 Ctrl-C
python -m http.server 8766 --bind 127.0.0.1 --directory research/v6_review_tool/dist
```

브라우저에서 http://127.0.0.1:8766 을 엽니다. 출력 파일이 이미 있으면 다른 새 경로를 선택하십시오. Stage2는 이 Mac 설치에서 실행하지 않습니다. Linux/CUDA 호스트에서 원래 제출 requirements와 V6 패키지로 검증해야 합니다. GitHub Actions smoke 통과는 모델·데이터 전체 이전이나 MPS 추론 통과를 의미하지 않습니다.
