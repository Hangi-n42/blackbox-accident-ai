# Stage2 4B NF4 후보 검증 — 2026-09-11

공개 5개에서 충돌 ±3프레임 적중이 최종 2B의 3/5에서 4B NF4의 4/5로 바뀌었다. 이미 반복 검토한 5개이므로 독립 검증 성능 또는 숨겨진 평가 성능으로 해석할 수 없다. 차선 진입 시점·방향·회피공간의 공식 정답은 없으며 정확도를 주장하지 않는다.

## 고정 조건과 자산

- 원본: 공식 `Qwen/Qwen3-VL-4B-Instruct`, revision `ebb281ec70b05090aa6165b016eac8ec08e71b17`, Apache-2.0.
- 원본 다운로드: `artifacts/candidates/qwen3_vl_4b`, 공식 파일 8,887,291,213 bytes. 두 가중치 shard는 공식 LFS SHA와 일치한 다운로드 manifest를 사용했다.
- 실행 환경: Windows, RTX 2080 SUPER 8GB, torch 2.8.0+cu128, transformers 4.57.6, accelerate 1.9.0, bitsandbytes 0.48.1.
- NF4, compute float16, double quantization 없음, SDPA, greedy generation.
- `solution/stage2_v2.py`의 질문·후보 탐색·영상당 4회 호출을 그대로 사용했다. 원본 `LocalVLM.ask`를 상속해 총 pixel budget 1,200,000을 유지했다. 이후 질문의 실제 후보는 앞선 모델 응답에 따라 달라진다.
- 4B BF16과 2B FP16의 순수 크기 비교가 아니다. 4B에는 NF4 양자화도 적용됐으므로 크기와 수치 정밀도가 함께 바뀌었다.
- 기존 제출 predictor SHA: `89ae0bc640670d7ed76103ce7c829d405ad27cc25368c74d153d3b96962b8442`.
- 기존 ask 소스 SHA: `01680e84220c64a6ff8f3415cf59bb7cd225f425814c92f6d2fe8213c5fe1cda`.

## 실제 결과

| 공개 ID | 충돌 정답 | 기존 2B | 4B NF4 | 4B 진입 | 4B 방향 | 4B 회피공간 |
|---|---:|---:|---:|---:|---|---:|
| S2_001 | 32 | 33 | 33 | 24 | LEFT | 0 |
| S2_002 | 30 | 44 | 44 | 36 | LEFT | 0 |
| S2_003 | 31 | 30 | 31 | 23 | LEFT | 0 |
| S2_004 | 41 | 46 | 40 | 4 | RIGHT | 0 |
| S2_005 | 30 | 32 | 32 | 23 | RIGHT | 0 |

유효 충돌 후보를 잃은 영상은 없었다. S2_002는 정답 후보가 제공됐지만 44를 선택해 14프레임 오차가 남았다. 회피공간은 기존 2B의 모두 1에서 4B의 모두 0으로 바뀌었다. 이 변화를 개선으로 판단할 정답이 없다.

| 측정 | 기존 2B FP16 | 4B 실시간 NF4 변환 | 저장 NF4 오프라인 재로딩 |
|---|---:|---:|---:|
| 충돌 ±3프레임 | 3/5 | 4/5 | 4/5 |
| 모델 준비 시간 | 11.815초 | 18.126초 | 12.969초 |
| 영상 평균 시간 | 7.405초 | 9.134초 | 9.070초 |
| 영상당 4회 모델 호출 평균 | 6.403초 | 8.134초 | 8.114초 |

4B 최초 전체 실행의 최대 CUDA allocated는 3,967,821,312 bytes(약 3.70 GiB), reserved는 4,722,786,304 bytes(약 4.40 GiB)였다. 수치는 로컬 GPU 측정이며 L40S 처리량이나 60분 통과를 보증하지 않는다. CPU offload 옵션은 구현됐으나 이번 실제 모델 검증에서는 실행하지 않았다.

## 저장 및 새 프로세스 재로딩

- 전체 5개가 성공한 후 이미 평가한 NF4 모델과 processor를 `artifacts/candidates/qwen3_vl_4b_nf4`에 저장했다. 원본 및 V1은 보존했다.
- `EXPORT_MANIFEST.json` 자체를 제외한 저장 payload는 3,078,241,946 bytes, 저장·해시 계산은 11.523초였다.
- 저장 manifest SHA: `ca6a1131fd311ece208b953a3c4806b1448dd1eb19afafd2e11a2139ea7dcf0d`.
- 라이선스 전문, 원본 모델 카드, 원본 revision과 다운로드 파일 해시, 양자화 변경 고지, 저장 파일별 SHA를 보존했다.
- 별도 새 프로세스에서 HF_HUB_OFFLINE/TRANSFORMERS_OFFLINE을 활성화하고 socket.connect, connect_ex, create_connection을 차단했다. 네트워크 연결 시도는 0회였다.
- 저장 파일 19개의 SHA를 검증한 뒤 전체 5개를 다시 추론했다. 최종 예측 5개와 20회 raw 출력·프롬프트·이미지 크기·출력 토큰 예산이 모두 원 실행과 일치했다.
- 저장 시의 manifest에는 당시 사실대로 `saved_unverified`가 남아 있다. 그 후의 검증 증거는 별도 `4b_nf4_offline_reload/reload_verification.json`에 기록했다. 기존 manifest는 변경하지 않았다.

재로딩을 위해 후보 로더에 저장된 quantization_config 검증 및 사용 경로만 추가했다. 원본 BF16→NF4 동작과 predictor/ask는 유지했다. 원 실험의 로더 스냅샷은 `4b_nf4_public5/vlm_candidate_source.py`에 보존돼 있다.

- 원 실험 candidate 로더 SHA: `1c1eacdfd42cef6dd35e221ace66933b61765ddad1f1a67579f73d48404e1987`.
- 재로딩 candidate 로더 SHA: `99b1cccb3de3172d08b4db8a40cbca3f6a1e595f6f6beb7250ab0c501d47ba7a`.
- 평가 스크립트 SHA: `21cb8d73a8e7534b49a866595e1306001266b871b29dc2658dd667271c0442d8`.
- 오프라인 검증 스크립트 SHA: `50ab88bc1980a3ec6fb413bbdb2206cfc63ffd97e2417bd11ddc62c00542a896`.

## 관찰한 경고와 한계

저장본 재로딩에서 Mistral tokenizer regex 경고가 발생했다. 설치된 Transformers의 `tokenization_utils_base.py` 2460–2496은 저장 config의 transformers_version이 4.57.2 이하일 때만 비Mistral 모델을 제외하고, 4.57.6 config에도 경고 조건을 적용한다. 자동 regex 수정은 하지 않았다. 실제 20회 출력은 모두 동일했다. 이 경고 때문에 검증된 tokenizer 동작을 임의 변경하지 않는다.

초기 실행은 Windows 경로 구분자가 소스 해시 키와 달라 GPU 로딩 전에 KeyError가 발생했다. 평가 스크립트의 키를 as_posix()로 통일한 뒤 재실행했으며, 이후 전체 검증 및 저장·재로딩은 성공했다.

## 근거 파일

- `artifacts/eval_stage2/4b_bnb_smoke.json`: 실제 NF4 CUDA 기본 연산.
- `artifacts/eval_stage2/4b_nf4_smoke/report.json`: 첫 영상 로딩·추론.
- `artifacts/eval_stage2/4b_nf4_public5/report.json`: 원본 모델 NF4 변환 전체 실행 및 export.
- `artifacts/eval_stage2/4b_nf4_offline_reload/report.json`: 저장본 전체 실행.
- `artifacts/eval_stage2/4b_nf4_offline_reload/reload_verification.json`: 오프라인·동등성 검증.
- [공식 원본 모델](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct).
