# Qwen3-VL-8B 공식 체크포인트 취득·실행 계획

확인 시각: 2026-09-15 21:10 KST 이후. **공식 NF4 파생본을 만들어 시험하는 경로는 있다. 로컬8GB·1.2MP 실행과60분 제출 적합성은 아직 미확인이다.** 메타데이터111,540바이트만 취득했으며 가중치 다운로드, 양자화, 모델/GPU 실행은 하지 않았다.

### 진행 상태 추가 — 원본 다운로드 시작, 완료 아님

위 문장은 최초 조사 시점의 상태다. 이후 root가 `research/v7/download_capacity_model.py`로 고정 공식 원천을 `artifacts/candidates/qwen3_vl_8b_v7`에 받기 시작했다고 보고했다. root 사전 확인값은 C드라이브 여유69GB(요구45GB 이상), 선택 파일18GB 미만,4worker·16MiB 범위 요청,30분 상한·총 전송36GB 상한이다. 원본 최종SHA 확인 뒤 중복 조각만 제거하는 정책이다. **다운로드 완료·양자화 성공·로컬 fit은 아직 확인되지 않았다.**

후속 실험 순서도 root 지시로 구체화됐다. 먼저 같은1.2MP의 전체 프레임 영상1개·4ask로 로딩/실행 smoke를 하고, 통과하면 **원래 V6와 동일한 정책**으로9개를 대조한다. 아래 §6의 V7 통합 모듈 교체는 최초 제안 기록이며 아직 실행/확정하지 않았다. OOM에서 임의로 정밀도/device/픽셀/fallback을 바꾸고 동일 시험이라고 부르지 않는다.

원래 V6 정책을 그대로 사용하면 최종 collision은 새 VLM 응답과 무관한 동일 uncapped motion argmax이므로 **최종 접촉 개선은 구조상 기대하지 않는다.** 이 대조는 기존4ask 내부 접촉·진입·방향·공간의 모델 교체 효과를 측정한다. V7의 복합 계획 실패가8B에서 해소되는지는 그 계획의 동일 입력/질문을 별도로 시험하기 전에는 확인할 수 없다.

## 1. 확인된 원천과 파일

[공식 Qwen 저장소](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct)의 메타데이터를 받아 revision을 고정했다. 원 응답과 소형 설정은 `qwen3_vl_8b_metadata/`, 파일별 고정 URL·크기·LFS SHA256은 `model_capacity_source_manifest.json`에 있다.

- repository: `Qwen/Qwen3-VL-8B-Instruct`
- revision: `0c351dd01ed87e9c1b53cbc748cba10e6187ff3b`
- 공개 접근: gated=false, private=false. 라이선스 표기: Apache-2.0.
- BF16 파라미터:8,767,123,696. tensor 데이터17,534,247,392바이트. safetensors 파일 합17,534,339,512바이트(16.3301GiB). 전체 저장소 파일 합17,545,915,883바이트.

| 파일 | 바이트 | 공식 LFS SHA256 |
|---|---:|---|
|model-00001-of-00004.safetensors|4,902,275,944|d5d0aef0eb170fc7453a296c43c0849a56f510555d3588e4fd662bb35490aefa|
|model-00002-of-00004.safetensors|4,915,962,496|8be88fb5501e4d5719a6d4cc212e6a13480330e74f3e8c77daa1a68f199106b5|
|model-00003-of-00004.safetensors|4,999,831,048|83de00eafe6e0d57ccd009dbcf71c9974d74df2f016c27afb7e95aafd16b2192|
|model-00004-of-00004.safetensors|2,716,270,024|0a88b98e9f96270973f567e6a2c103ede6ccdf915ca3075e21c755604d0377a5|

위 SHA는 **공식 API가 제공한 기대값**이다. 아직 실제 tensor 파일을 받아 계산한 검증값은 아니다. 메타데이터 응답 SHA는 `716db6668dac746a8172f310505f7789e09b64581bbd2a00795b67784cc9d55a`다.

저장소에 별도 LICENSE/NOTICE 파일은 없고 model card가 Apache-2.0을 선언한다. 파생본에는 공식 card·출처·revision·변경 고지 및 [Apache 원문](https://www.apache.org/licenses/LICENSE-2.0)을 포함한다. 원 저작권/귀속 고지를 보존하고, NF4 변환 및 processor 재직렬화 사실을 명시한다. `Qwen 공식 NF4 배포본`이라고 부르지 않는다. 이는 Apache §4의 재배포 요건에 대응하는 계획이다.

## 2. 아키텍처와 비교 조건

[고정 config](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct/blob/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b/config.json)의 architecture는 기존과 같은 `Qwen3VLForConditionalGeneration`, model_type은 `qwen3_vl`이다. remote code가 필요하다는 항목은 없다. 기존 로더의 `trust_remote_code=False`를 유지한다.

| 항목 | 현재 공식4B | 공식8B |
|---|---:|---:|
|언어 hidden size / layer|2560 /36|4096 /36|
|언어 intermediate size|9728|12288|
|KV heads / head dim|8 /128|8 /128|
|입출력 embedding 공유|true|false|
|시각 depth / hidden size|24 /1024|27 /1152|
|시각 patch / merge / temporal patch|16 /2 /2|16 /2 /2|
|DeepStack 시각 layer|5,11,17|8,16,24|

따라서 단일 변경요인은 **공식 모델 체크포인트 교체**다. 언어 파라미터 수만 바꾸는 순수 인과 실험이 아니다. 시각 인코더/merger/임베딩도 함께 바뀐다. [Qwen3-VL 기술 보고서 §2,§5.5,§5.9](https://arxiv.org/html/2511.21631v2)는 작은2B/4B와 다른 모델의 시각 인코더 규모 차이, grounding 및 영상 벤치마크를 설명한다. 그러나 해당 평가는 우리 NF4·20타일·짧은 공동 JSON 계획이나 최초 접촉±0.3초 검증이 아니다. 논문 성능을 대회 점수로 환산하지 않는다.

processor는 `Qwen3VLProcessor`/`Qwen2VLImageProcessorFast`다. 공식4B 로컬 원본과8B 메타데이터의 preprocessor/video_preprocessor/tokenizer_config/generation_config는 바이트 동일했다. chat_template는 파일 바이트가 다르지만 JSON 파싱 내용은 동일했다. tokenizer.json/vocab.json/merges.txt는 로컬4B 파일에서 재계산한 Git blob ID가8B 공식 API와 모두 같았다. 이를 근거로 같은 전처리 경로를 사용할 수 있지만, 실험 때 실제 input_ids/image_grid_thw/전처리 tensor 해시도 확인한다. 배포에는8B 원천 파일과 출처를 사용한다.

## 3. 8GB 메모리 위험과 서버 적합성

현재 설치 메타데이터: torch2.8.0+cu128, transformers4.57.6, accelerate1.9.0, bitsandbytes0.48.1. [bitsandbytes0.48.1 공식 설치표](https://huggingface.co/docs/bitsandbytes/v0.48.1/installation)는 Windows CUDA12.8–12.9의 sm75와 Linux의 sm89를 지원 대상으로 명시한다. 따라서 RTX2080 SUPER/Turing와 L40S/Ada는 지원 계열이다. 지원 계열이라는 사실이 특정 입력의 메모리 적합성을 보장하지 않는다.

기존 `solution/vlm_candidate.py:28–55`는 FP16 compute, sdpa, NF4, double quant=false, 모든 가중치 GPU0 배치다. `gpu_memory` 옵션은 **nf4 모드에서 사용되지 않는다.** 기본5GiB 옵션을 바꿔도 NF4 메모리 상한/CPU offload가 생기지 않는다. BF16 원본을 Turing에서 `dtype=auto`로 읽는 변경은 하지 않고 기존 FP16을 유지한다.

**계산 추정이며 실측 아님:** 입력 embedding과 별도 lm_head는 각각151936×4096이다. 설치된 Transformers의 `integrations/bitsandbytes.py:291–320`은 untied output embedding을 양자화 제외하고, 일반 embedding도 Linear4bit 교체 대상이 아니다. 두 행렬을FP16, 나머지를NF4 0.5byte+64개당FP32 scale 0.0625byte로 근사하면:

`1,244,659,712×2 + (8,767,123,696−1,244,659,712)×0.5625 = 6,720,705,415 bytes = 6.2591GiB`

이는 weight 중심 예산이며 정확한 직렬화 크기나 GPU peak가 아니다. 비선형 가중치, padding/양자화 metadata, CUDA context, KV cache, 시각 attention 활성화, temporary buffer, allocator reserved memory를 완전히 계산하지 않았다. 현재4B의 메모리를 단순2배 하거나8B×0.5byte만으로 판단하면 안 된다. **6.26GiB에 런타임을 더하면 로컬8GB는 여유가 작다.** 데스크톱/다른 프로세스 점유가 있으면 로딩 또는1.2MP prefill에서 OOM이 날 수 있다.

- 로컬 첫 시도: 현재 NF4/FP16/sdpa, 같은1.2MP·같은최대 입력·단일 모델만 로딩. load/prefill/decode/export 각각 peak와 wall time을 기록한다. OOM이면 이미지 예산을 조용히 줄여4B와 다른 조건으로 비교하지 않는다.
- 대안: 기존 `fp16_offload`는 원본 FP16을 `device_map=auto`, CPU/GPU max_memory로 나누는 기능이다. [Accelerate1.9 공식 문서](https://huggingface.co/docs/accelerate/v1.9.0/concept_guides/big_model_inference)는 CPU/disk weights의 필요시 GPU 이동을 설명한다. 실행 경로는 있지만 PCIe 이동으로 매우 느려질 수 있다. **NF4와 다른 정밀도 조건**이므로 그대로 단일요인 본실험에 합치지 않는다. NF4 offload가 필요하면 별도 로더 검증이 필요하며 현재는 구현/검증되지 않았다.
- 서버44.7GiB: NF4 또는 원본FP16 weight 예산은 장치 총량보다 작다. 하지만 실제 서버 전체 peak/처리시간은 미측정이다. 로컬에서만 다른 정밀도·device map을 썼다면 최종 NF4 저장본을 서버 조건에서 다시 검증해야 한다.

NF4 설정과 compute dtype은 [Transformers 공식 양자화 문서](https://huggingface.co/docs/transformers/v4.57.0/quantization/bitsandbytes)에 있다. 여기서는 추가 절약을 위해 double quant를 켜지 않는다. 켜면 모델 크기 외 양자화 조건도 달라진다.

## 4. 기존 코드 재사용 범위

- `solution/vlm_candidate.py`: 클래스 이름의4B 주석과 달리 실제 로딩은 path/config 기반이다. 같은 아키텍처의8B에 구조상 재사용 가능하다. `ask()`의192토큰/4call 여부는 호출 모듈이 관리한다. 메모리 성공은 별도다.
- `scripts/download_candidate_vlm.py:17–29`: repository와폴더가4B로 고정돼 **그대로 실행하면 잘못된 모델**을 받는다. 승인 뒤 새8B 다운로드 파일에서 지금 고정 revision/4shard/각SHA를 명시해야 한다. 기존16MiB Range 검증·최종SHA 방식은 재사용할 수 있다. 원본17.55GB, 조립 parts와원본 동시 보관 시약35.1GB에 export·ZIP 공간이 추가된다. 현재 디스크 여유 확인은 root 담당이다.
- `scripts/evaluate_stage2_4b.py:28–100`: `export_nf4`는4B repo 검증·모델명/README를 하드코딩하고 공개5개 실행 완료 조건에 결합돼 있다. 새8B용 export helper로 경계만 분리해야 한다. 그대로8B에 호출하면 검증에서 실패한다. 기존 파일은 변경하지 않는다.
- 저장 기술은 `model.save_pretrained(safe_serialization=True,max_shard_size='2GB')`와 `processor.save_pretrained`를 재사용한다. 설치된 `quantizer_bnb_4bit.py:316–324`는bitsandbytes≥0.41.3에서 저장 가능하다고 검사하며 현재0.48.1이다. **저장 성공과 재로딩 동등성은 별개다.**
- 모델/processor를 새 프로세스에서 HF offline+소켓 차단으로 재로딩하고, 동일 입력·질문의 원문응답/최종4출력·전처리 해시·모든 파일 SHA를 비교한다. 저장된 quantization_config를 사용하고 새 설정을 중복 적용하지 않는다.

## 5. ZIP과 오프라인 제약

대회 통합 정보 §9(265–275행)의 명세는 ZIP10GB/해제32GB, Stage1→2→3 전체60분, 설치 외 네트워크 금지다. 기존 submit_v6 manifest의 Stage2 VLM 외 payload는346,139,010바이트다. 위NF4 weight 근사에 이를 더하면약7.067GB이며, processor·변환metadata 등은 추가된다. **따라서10GB 안에 들어갈 가능성을 뒷받침하는 계산은 있지만 확정은 실제 export/ZIP 크기로 한다.** 원본BF16 17.53GB를 그대로 넣는 방법은 압축률에 의존하므로 현재 제출 계획으로 채택하지 않는다.

제출에는 선택된8B NF4 한 벌과 로컬 processor/설정/라이선스/출처만 넣고,4B·원본BF16·parts·실험 영상·캐시는 함께 넣지 않는다. source map과 파일 해시 허용목록을 새로 동결해야 한다. `local_files_only=True`, `trust_remote_code=False`, 사전 설치한 호환 라이브러리와 네트워크0을 실제 압축 해제본에서 검증한다. 서버에서 원본 다운로드/양자화로 시간을 소비하는 경로는 사용하지 않는다.

## 6. 승인 후 최소 실험 및 판정

1. **취득/런타임 시험:** 고정 원천만 다운로드·SHA 확인, NF4 load→실제 최대 pixel 입력 한 번→export→별도 프로세스 재로딩.4B/8B 동시 GPU 로딩 금지. 이 단계에서 시간·메모리 미달이면 의미 성능 실험을 시작하지 않는다.
2. **모델 체크포인트만 교체:** 동결 V7 모듈·원본9개·4call/192token·1.2MP·precision·generation을 유지한다. 첫 계획은 같은 실제 image/prompt를 비교할 수 있다. 후속 질문/이미지는 앞 응답에 의존하므로8B가 다르게 계획하면 달라지는 것이 구조상 예상된다. 이를 숨기고 전체4call 입력이 동일했다고 부르지 않는다.
3. **실행 성공과 정확도 분리:** null 감소만으로 통과시키지 않는다. 동일 차량 참조의 실제 이미지 타당성, 전체→창→fine 후보 포함률, 알려진 접촉/진입 정확도, 방향/공간 Macro-F1, fallback과기존 성공 손실을 기존 사전 gate로 평가한다. 이번8B 결과를 보고 gate를 완화하지 않는다.
4. **노출 경계:** 기존9개는 개발 자료이며8B가 좋아져도 독립 일반화 증거가 아니다. fresh3 AI 검수는 접촉 미확정0/3 evaluable이고 human/공식 GT가 아니다. 독립 성능 판정은 실제 사용 가능한 정답이 있는 별도 자료에서만 한다. 최종 패키지/전체60분 통과 전 채택·제출 적합성을 확정하지 않는다.

이번 제안은4B의 공동 사건 계획 실패를 모델 체크포인트 교체로 반증 가능한지 보는 실험이다. 단일/시트 caption과blank 구분 성공은 시각 경로가 동작한다는 근거지만,8B가 작은 타일에서 정확한 접촉/바퀴 경계를 찾는다는 보장은 아니다. 해상도 손실·장면 모호성·정답 부족은 모델 크기를 늘려도 남는다.
