# 8B 입력 embedding CPU 상주안 실행 검토

**결론: 조건부로 실행 가능한 메모리 배치 대안이다. 모델 정확도 개선안은 아니다.** 같은 FP16 embedding weight에서 행을 선택한 뒤 원래 input ID의 장치로 돌려주는 연산은 가중치 재양자화·dtype 전환 없이 구현할 수 있다. 그러나 코드 의미상의 동등성과 전체 생성 결과의 실제 동등성은 별도다. 이번 검토는 설치 소스/API/8B 설정만 읽었으며 모델·가중치를 로드하거나 GPU·시험 구현을 실행하지 않았다.

## 확인한 환경과 원본 경로

설치 버전은 torch2.8.0+cu128, transformers4.57.6, accelerate1.9.0, bitsandbytes0.48.1이다. [`qwen3_vl_8b_metadata/config.json`](C:/Users/dsl/Desktop/Dacon/블랙박스/research/v7/qwen3_vl_8b_metadata/config.json)의 vocab_size151936, hidden_size4096에 따라 embedding 하나의 FP16 payload는 **1,244,659,712bytes=1.1591796875GiB**다. 최상위 `tie_word_embeddings=false`다. 실제 로드 후 dtype와 storage alias를 아직 확인하지 않았으므로 절약량은 설정 기반 payload 추정이다. 부모가 제공한 모델 전체6.26GiB 추정도 이번 검토에서 실측한 값이 아니다. 원본 text config의 dtype은 bfloat16이므로 실제 NF4/FP16 로드가 FP16 embedding을 만들었는지 반드시 확인해야 한다.

로컬 [`modeling_qwen3_vl.py`](C:/Users/dsl/Desktop/Dacon/블랙박스/.venv/Lib/site-packages/transformers/models/qwen3_vl/modeling_qwen3_vl.py:771)는 입력을 `nn.Embedding`으로 정의한다. 상위 get/set_input_embeddings(1285–1289행)가 같은 내부 embedding에 연결된다. 모델 forward의1132행에서 embedding을 호출하고,1140행 부근에서 image embedding을 그 출력의 device/dtype에 맞춘다. 따라서 잘못 CPU 출력을 반환하면 영상 특징까지 CPU로 옮기거나 GPU decoder와 충돌할 수 있다.

또한1440–1454행에는 special token의 **0차원 scalar ID**를 embedding에 넣는 경로가 있다. 구현이 input을 항상2차원 batch라고 가정하면 안 된다. 일반 generate의 첫 prefill 및 뒤따르는 단일 token decode도 모두 지원해야 한다.

PyTorch [`Embedding.forward`](C:/Users/dsl/Desktop/Dacon/블랙박스/.venv/Lib/site-packages/torch/nn/modules/sparse.py:191)는 padding_idx/max_norm/norm_type/scale_grad_by_freq/sparse를 F.embedding에 전달한다. max_norm이 지정되면 weight를 실제로 갱신할 수 있으므로 단순 gather라는 주장은 **max_norm=None·추론 모드·같은 weight** 조건에서만 사용한다. [PyTorch 공식 API](https://docs.pytorch.org/docs/main/generated/torch.nn.functional.embedding.html)

## 허용할 구현 경계

모델을 기존 방법으로 로드한 뒤 입력 embedding만 `nn.Embedding`의 하위 클래스로 교체한다. 내부의 `weight` parameter 이름을 유지하고 값/dtype/shape/embedding 속성을 보존해야 한다. CPU tensor를 `_weight`로 전달하는 등 불필요한 무작위 전체 embedding 생성·초기화·padding행 재설정을 피한다. 파라미터는 학습하지 않는다.

forward는 입력 ID의 원래 device를 기록하고, CPU ID와 CPU weight로 F.embedding을 실행한 뒤 결과를 그 device로 반환한다. `.to(input_ids.device)`와 `.to(input_ids)`를 혼동하면 안 된다. 후자는 정수 ID의 dtype으로 변환할 수 있다. 결과 dtype은 원래 FP16이어야 한다. weight를 FP32로 올렸다가 다시 내리거나 NF4로 바꾸는 것은 이 동등 후보의 범위를 벗어난다.

별도 child module에 embedding을 넣으면 `...embed_tokens.weight`가 `...embed_tokens.inner.weight` 등으로 바뀐다. 기존 state_dict key를 유지하려면 추가 wrapper child 없이 같은 이름의 weight를 등록해야 한다. `set_input_embeddings`로 교체하고 lm_head/vision/decoder는 그대로 유지한다. config의 untied 설정뿐 아니라 실제 embedding/lm_head parameter 객체와 storage가 다름을 검사해야 한다.

## 주요 실행 위험

1. **초기 all-GPU 로드 OOM은 해결하지 못한다.** 교체는 로드 성공 후에만 가능하다. 이미 load peak에서 실패하면 post-load 교체안의 선행조건이 충족되지 않는다. CPU에서부터 선택 로딩하는 별도 loader를 동등성 검증 없이 같은 안으로 취급하면 안 된다.
2. **GPU 공간 해제는 payload 추정과 다르다.** 기존 GPU embedding을 참조하는 지역 변수·hook·weight map이 남으면 실제 공간이 해제되지 않는다. allocator reserved와 allocated도 구분해야 한다. `empty_cache`만으로 살아 있는 tensor가 해제되지 않는다. 실측으로 확인한다.
3. **Accelerate hook은 module 교체와 충돌할 수 있다.** 설치 `big_modeling.py:354–369`는 여러 device이거나 force_hooks인 경우 hook을 건다. 현재 singleGPU+b nb0.48.1의4bit는 이 조건만으로 force_hooks가 되지 않지만 실제 `_hf_hook` 존재를 확인해야 한다. 부모 hook이 재귀적으로 weight를 CUDA로 옮기면 CPU정책을 깨뜨린다. hook이 있으면 무조건 제거하는 대신 해당 모델의 hook 실행 경로를 확인한 뒤 진행한다. [공식 dispatch_model 설명](https://huggingface.co/docs/accelerate/main/en/package_reference/big_modeling)
4. **모델 전체 `.to('cuda')`·재dispatch·자동 tie를 다시 호출하지 않는다.** subclass의 일반 `.to`는 CPU weight도 이동시키므로 절약이 사라진다. `.half()`·`.float()`도 고정 dtype 계약을 깨뜨린다. CPU/GPU weight 배치를 검사하고 이후 일괄 이동을 금지한다.
5. **`model.device`가 전체 장치 배치를 의미하지 않는다.** transformers `get_parameter_device`(modeling_utils.py:300)는 첫 parameter device를 반환한다. 현재 최상위 Qwen3VL은 visual을 먼저 등록하지만 text submodel의 첫 parameter는 CPU embedding이 된다. 입력은 기존 `LocalVLM.ask`의 명시적 `self.device='cuda'` 경로를 유지하고 반환 장치를 ID에서 결정한다. 잘못된 `.device` 기반 자동 입력 이동을 검사한다.
6. **메타데이터와 실제 배치를 구분한다.** `hf_device_map`을 바꾸는 것만으로 동작은 바뀌지 않는다. GPU root만 기록하면 CPU embedding을 숨기고, 임의 CPU map으로 재dispatch하면 새 hook 정책이 생긴다. 실제 parameter별 residence와 `cpu_embedding_gather_return_to_input_device` 정책을 명시하고 가능한 모델 prefix별 device map을 정확히 기록한다. metadata를 다시 loader 지시로 쓰지 않는다.
7. **저장은 동작을 직렬화하지 않는다.** state_dict key/value가 같아도 `save_pretrained`는 이 custom forward를 자동 복원하지 않는다. 저장 NF4 가중치를 fresh process에서 로드한 뒤 같은 subclass 교체를 다시 적용해야 한다. 별도 압축·양자화·embedding dtype 변경 없이 state_dict와 양자화 설정을 보존한다. 대회 local/server 모두 같은 정책을 사용해야 한다.
8. **시간 비용이 있다.** 매 decode token에서 ID CUDA→CPU 이동은 동기화를 유발하고 embedding CPU→CUDA 전송을 추가한다. prefill 전송량은 token수×4096×2bytes이며 visual placeholder도 embedding 단계에 포함된다. 1.159GiB 상주량 감소가 전체60분 실행을 보장하지 않는다. CPU2 연구에서 처리량과 host RAM도 측정해야 한다.

## 의미 있는 검증 순서

| 검증 | 필요한 증거 | 중단 조건 |
|---|---|---|
| 로드 전제 | 기존 allGPU NF4 로드·full 입력 smoke의 allocated/reserved peak 기록 | 로드 자체 OOM이면 이 post-load안으로 해결했다고 주장 금지 |
| 실제 구조 | embed FP16/shape, max_norm=None, storage untied, hooks, 입력device 확인 | dtype/alias/hook 경로 미확인 |
| 국소 exact 계약 | 작은 합성 weight의 scalar/1D/2D/비연속/중복/padding/special ID 및 FP16 signed-zero 등; CPU wrapper와 기존 GPU gather 출력 bytes 비교 | 값·shape·dtype·device 차이 |
| 메모리 계약 | 교체 전후 살아 있는 GPU embedding 참조·allocated/reserved·CPU RAM, vision/decoder/lm_head 장치 유지 | 절약이 없거나 원치 않는 CPU decoder·weight 이동 |
| 실제 weight 계약 | 기존 embedding CPU사본과 교체 후 weight bytes/SHA, 선택된 ID의 embedding 출력 exact, 전체 state_dict key/dtype/shape 불변 | 값·키·dtype 변경/새난수초기화 |
| 생성 계약 | 동일8B/동일입력의 allGPU가능한 작은 호출에서 token IDs·raw문자열·4개필드 비교; prefill+cacheddecode 포함 | 차이가 나면 원리상동등만 근거로 승인 금지 |
| 예산 확장 | 기존full1.2MP 입력 정책 유지한 단일/반복4호출 및 여러파일 CPU정책 메모리·시간 | OOM/누적메모리/전체시간예산 부족 |
| 저장·재로딩 | fresh offline load→동일교체→같은 raw token/문자열, strict state키 및 CPU정책 확인 | subclass미복원·다른dtype·모델전체재이동 |

입력 축소를 사용한 국소 동등성 검사는 정확도 후보를 축소 입력으로 평가하는 것과 다르다. full 입력 allGPU가 OOM이면 그 입력에 대한 직접 생성 동등성을 로컬에서 검증할 수 없다는 한계를 남기고, 충분한 메모리 환경에서 같은 두 정책 비교가 필요하다. 작은 입력에서의 일치만으로 full 입력 동등성까지 검증 완료라고 쓰지 않는다.

채택 검토는 full 입력 정책 유지·실제 실행·오프라인 재로딩·자원 기록을 갖춘 뒤 할 수 있다. 이번 문서는 구현 승인이나 메모리 절약 실측 결과가 아니다.
