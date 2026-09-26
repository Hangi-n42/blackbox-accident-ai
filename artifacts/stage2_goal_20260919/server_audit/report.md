# Stage2 제출 자산·서버 실행 준비 감사

2026-09-19. **제출 자산 무결성 확인 완료, Linux/CUDA 실행 검증 미완료.** 모델 로드·추론·다운로드·ZIP 재해제·원격 접속은 수행하지 않았다. 원본 소스와 제출물을 변경하지 않았다. 원시 파일별 결과는 `readiness.json`, 감사 코드는 `audit.py`다.

## 확인한 범위

| 항목 | 확인 결과 |
|---|---|
| V6 ZIP | 47파일, 2,914,590,320바이트. ZIP SHA 및 47개 내부 파일 SHA·크기·CRC가 release manifest와 일치 |
| V7 ZIP | 48파일, 2,914,591,230바이트. ZIP SHA 및 48개 내부 파일 SHA·크기·CRC가 release manifest와 일치 |
| Stage2 공통 파일 | `model/stage2/` 아래 공통 34파일이 V6/V7 바이트 동일. 이 폴더에는 Stage1/3 공유 코드도 포함됨 |
| Stage2 진입점 | `_runtime`과 `predict_stage2` 함수 AST 동일. V7 추가 파일은 Stage1 반전 모듈 하나 |
| NF4 체크포인트 | 양 ZIP의 2개 shard, index의 1,781개 tensor 연결, header offset 범위 확인. 모델 로드 검증은 아님 |
| 추출·release 사본 | 존재하는 Stage2/진입점/requirements 파일 전부 ZIP과 일치. V7 두 사본은 모델 자산을 보유 |
| Mac 모델 | 기록된 16개 파일 SHA 모두 기존 model_integrity와 일치. 실제 단일 safetensors 3,093,767,283바이트, header 1,219개 tensor의 offset 범위 확인 |

V6 ZIP SHA: `860f9d08d6013f090ae4bc710bdbbfeee06e01a0cd71f8a784b5689cdb4b7ee0`.

V7 ZIP SHA: `2cbdc77cab56dea781025c693a038fc3b0b6d23b108b9d904454b973b9a97fae`.

**V6 추출본의 구분:** `artifacts/submissions/verify_v6/model/stage2/`는 code만 있고 vlm 폴더가 없다. `releases/v6/source/model/stage2/vlm/`도 문서 4개만 있다. 이 두 경로를 가중치까지 준비된 V6 실행 경로라고 부르면 안 된다. 가중치는 V6 ZIP 안에 보존돼 있고 V7 추출본에는 동일 바이트가 존재한다. 최초 감사 스크립트의 추출 index 가정을 수정한 이력은 `attempt1.json`에 남겼다.

**Mac index의 구분:** Mac 모델의 `model.safetensors.index.json`은 없는 원래 두 shard를 가리킨다. 이 불일치를 원시 결과에 보존했다. 현재 설치된 mlx-vlm 0.3.4의 `utils.py:142,167`은 index를 읽지 않고 실제 `*.safetensors`를 glob하여 로드한다. 따라서 오래된 index만 보고 현 MLX 실행 불가라고 단정하지 않는다. 이번 실제 모델 로드는 수행하지 않았다.

## 실행 환경과 연결 경로

- 현 호스트: macOS 27.0 arm64, Python 3.12.14. 두 검사 환경 모두 torch 2.8.0, CUDA build 없음, CUDA 사용 불가, CUDA device 0개. MPS는 사용 가능하다.
- scipy 환경: Transformers 4.57.6, accelerate 1.9.0, bitsandbytes 없음. MLX 환경: mlx 0.29.3, mlx-vlm 0.3.4, bitsandbytes·accelerate 없음.
- 제출 requirements는 `bitsandbytes==0.48.1`이며, 보존된 CandidateVLM은 CUDA가 없으면 중단한다. Mac MLX 모델은 이 NF4 체크포인트와 다른 자산이다.
- 사용자 `~/.ssh/config` 없음. 시스템 SSH config 및 Include 2개까지 읽어 literal host alias 0개를 확인했다. 키·인증정보는 읽지 않았고 원격 연결 시도는 0회다. 검사한 설정에서 실제 Linux/CUDA 연결 경로는 확인되지 않았다. 사용자가 보유한 모든 외부 서버가 없다는 의미는 아니다.

## 다음 실행에 필요한 검사

1. 구성된 Linux/CUDA 실행 자원을 확보한 뒤 보존된 V7 Stage2 경로와 동결한 전체 프레임 입력으로 오프라인 로드·추론을 실행한다. 서버 구매·계정 생성·접속을 수행한 상태가 아니다.
2. `scripts/verify_v7_entry.py:36-64`에는 Stage2 전용 출력 검사가 없다. CUDA에서 이 도구가 PASS하더라도 다섯 열·ID·범주·프레임 계약 검증은 완료되지 않는다. `scripts/verify_submission.py:49-57`의 ID·중복·LEFT/RIGHT·0/1 정수·원본 번호 검사를 재사용하고 시간·CUDA peak memory 기록을 추가한다.
3. 같은 영상의 같은 이미지 바이트·원본 번호·CPU 특징/점수 배열·라이브러리 버전을 먼저 대조한다. 이번 감사는 파일·환경 감사이며 optical flow 재계산이나 저장 배열 delta 검사를 수행하지 않았다. `.46초` canonical PNG CPU 진단과 `.54초` Mac 캐시 결과 차이를 양자화 영향으로 단정하지 않는다.
4. 새로운 전체 Stage2 출력과 서버 실행 결과가 있어야 후보의 제출 준비를 판정할 수 있다. 작은 공개 표본 실행만으로 숨은 전체 입력의 60분 제한 충족을 확정하지 않는다.

이번 완료 범위는 자산 무결성·실행 환경의 사실 확인이다. 정확도 개선, 새 CUDA 로드 성공, L40S 자원·시간 통과, 새 제출 완료를 주장하지 않는다.
