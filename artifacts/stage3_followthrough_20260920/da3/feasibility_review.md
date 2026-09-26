# DA3-Small Mac 소규모 실행 준비 검토 — 2026-09-20

전문가 독립 준비 검토. 이 문서 작성자는 가중치 전체 다운로드·패키지 설치·모델 추론을 하지 않았다. 부모 작업자는 별도로 실행 예정이므로 실제 완료여부는 후속 로그를 봐야 한다.

## 고정할 원본

- 공식 코드: https://github.com/ByteDance-Seed/Depth-Anything-3
- 코드 commit: `3d835ec1a5802d64a8b8b15f817a1ab54809bfe4`
- 실제 실행 소스 경로: `artifacts/stage3_followthrough_20260920/da3/vendor/Depth-Anything-3/src` (부모가 clone, commit 동일 확인).
- 검토용 `official_source`도 같은 commit이며 수정하지 않은 src와 LICENSE 등만 보존했다. 최초 공식 source tarball은 23,650,902bytes였고 연구 가중치 다운로드가 아니다. 실행경로는 위 vendor 하나만 사용한다.
- 모델: https://huggingface.co/depth-anything/DA3-SMALL
- HF revision: `e08cab65ca0ec38e7826075418411ab90cab4da3`
- `model.safetensors`: **137,248,940bytes** (~130.89MiB), SHA256 `364492e38a3a06d221ac75da7f6621ada3f2361cd24fde11ba79091e9f40efcf`.
- `config.json`:1202bytes, `small_config.json`으로 확보. source/HF metadata는 별도 JSON 보존.
- 모델 cardData license=`apache-2.0`, 소스 LICENSE Apache2.0. 개별 DINO 출처 고지도 보존.
- HF 기본cache `/Users/hyeongi/.cache/huggingface/hub`에 DA3모델 없음. repo 파일검색에서도 기존 DA3실행본을 찾지 못했다.

### 문서 모델 크기와 실제 파일 차이

공식 README의 Small0.08B 표기 대신 실제 체크포인트를 기준으로 기록해야 한다. safetensors HTTP Range로262,144bytes만 읽어51,080bytes header를 파싱했다. 437개 텐서, 전부F32, 총 **34,299,463 elements**이다. 가중치 전체를 받거나 로드한 검증은 아니다. 모든 key는 `model.` prefix를 가진다. 부모는 실제 strict load 후 parameter count를 최종 확인해야 한다.

## 현재 환경과 최소 의존성

기존환경 `artifacts/mac_experiments/scipy_compat/.venv/bin/python`: Python3.12.14, torch2.8.0, torchvision0.23.0, numpy1.26.4, safetensors0.6.2, huggingface_hub0.34.4. MPS available=True. cv2/PIL/tqdm/PyYAML6.0.3 존재.

없는 것: `omegaconf`, `addict`, `antlr4`, `imageio`, `einops`(기본sys.path), `xformers`, `evo`, `trimesh`, `pycolmap`, `gsplat`.

최소 신규 target 패키지 후보(기존env upgrade없이):
- omegaconf2.3.0
- antlr4-python3-runtime4.9.3 (omegaconf 의존)
- addict2.4.0
- imageio2.37.0 (공식InputProcessor→parallel_utils import)
- einops0.8.1은 `artifacts/stage3_state_learning_20260919/vendor`의 기존본 재사용 가능.
- PyYAML/numpy/Pillow/torch/torchvision/safetensors는 기존 설치 재사용.

이 버전들은 최소 실행 제안이며 실제 설치/호환 성공은 부모의 smoke로 검증해야 한다. 새 패키지는 격리 vendor에 `--target --no-deps`로만 넣는 방식이 기존환경을 보존한다. 설치후 목록/버전 저장 필요.

## 실행 경로 권고

high-level `depth_anything_3.api`는 사용하지 않는다. import 시 export 모듈과 evo 등 이 실험에 불필요한 IO/GS 의존성을 불러온다. 공식 `DepthAnything3Net`과 `InputProcessor`를 직접 호출하면 원래 네트워크/공식 전처리를 보존하면서 해당 의존성을 피할 수 있다. 모델 소스 수정·fake module 주입 필요 없음.

```python
config = json.load(open('small_config.json'))['config']
model = create_object(OmegaConf.create(config))
wrapper = torch.nn.Module()
wrapper.model = model
missing, unexpected = safetensors.torch.load_model(wrapper, 'model.safetensors', strict=True)
assert not missing and not unexpected
model.eval().to(device).float()
images,_,_ = InputProcessor()(rgb_frames, process_res=504,
                             process_res_method='upper_bound_resize', sequential=True)
# 실제 InputProcessor._stack_batch는 (N,3,H,W): doc과 다름.
images = images.unsqueeze(0)
with torch.inference_mode():
    output = model(images.to(device).float(), infer_gs=False,
                   use_ray_pose=False, ref_view_strategy='saddle_balanced')
```

주의: 공식 API wrapper는 CUDA bf16여부를 보고 다른장치에서도 autocastdtype를 결정하므로 본 실험은 direct net FP32를 명시한다. 이는 수학구조 수정이 아니라 inference precision 고정이다. core Attention은 PyTorch scaled_dot_product_attention을 사용한다. SwiGLU의 xformers import는 try/except fallback이 있으므로 xformers 부재만으로 core실행 차단은 아니다. 현재Small구성에서GS/ray-SVD 경로는 켜지 않는다.

## Mac 메모리·연산 위험

MPS available은 실제 whole-model 성공의 증거가 아니다. 3frame504 CPU/MPS smoke, 유한출력/pose차이 확인 후21frame을 시도한다. global attention이 `(B,S,N,C)`를`(B,S*N,C)`로 펼치는 코드이므로21frame504에서는 dense attention 중간값이 수GB가 될 수 있다. 실제kernel이 전체행렬을물리화하는지/메모리피크는 측정전확인불가다. 동시실험과메모리경쟁 피할것.

21frame504가 자원상 불가능할 때만 센서/성능결과를 보기전에 한단계낮은 고정해상도를 선택하고 freeze에 사유를 기록한다. 영상마다 유리한해상도/posehead를 선택하지 않는다. unsupported operation은 정확로그를 남기고CPU대조를 하며, silent fallback으로 실행했다고숨기지 않는다.

## 28window pilot 설계 검토

- 기존28창/21frames/2초를그대로사용, 각창을1회 공동추정. 독립프레임깊이/독립window의scale을 이어 미분하면안됨.
- camera head output `extrinsics`는w2c. `C=-R^T t`로camera center변환. extrinsic t그대로미분금지.
- 중앙시각원점t에C(t)=c0+v*t+0.5*a*t² 적합. `q=(v·a)/(v·v)` 단위1/s. 일정positive scale와전역회전에불변, 시각별scale왜곡에는불변아님. 최소speed/궤적총이동 및fit잔차·reverse일관성검사로미정값구분.
- 기존proxy target이창평균a/v인지중앙a/v인지명시하고q중앙시각해석과일치시킬것. 두 target비교를보조로기록할수있지만더유리한것을선택하지않음.
- reverse: 입력순서를반대로하고증가하는시간에적합하면`q_reverse≈-q_forward`가예상됨. 혹은 예측camera center순서를원복한뒤동일원래t에적합하면`q_reverse_restored≈q_forward`. 두표기혼동금지.
- reference middle: 원래와 같은 이미지에서 ref_view_strategy만middle로변경. q는공통회전/translation/scale에불변이므로q자체비교가능. 궤적비교는Sim3정렬하되센서pose를쓰지않음.
- 코드vision_transformer는참조frame앞재배열후출력feature를원래순서로복구함을확인했다. 중복복구하지않도록원본frameindex저장.
- default/reverse/middle은3개성능후보가아니라하나의고정방법의안정성대조. 가장맞는것을라벨기준으로선택하면안됨.
- 통과창만의MAE보다전체28창coverage, A/D/C별오류, 동일속도pair방향과기존반전사건의연결을우선보고. 학습/검증2A표본의부족은기존대로남음.
- 이는학습전타당성검사다. 통과여부를점수향상으로부르지않는다. 기하한후보pilot에서약하면더큰DA3/다른head검색으로확대하지않는것이합리적이다.


### 초기 strict-load 차단 후 정정

부모 smoke의직접 state_dict strictload에서6LayerNorm alias누락보고를받아소스/설치safetensors로감사했다. DualDPT는`ln_seq`를한번생성해4개auxhead에공유하므로state_dict에는동일weight/bias가여러이름으로존재하지만safetensors는한개만저장한다. `wrapper.model=net`후`safetensors.torch.load_model(wrapper,...,strict=True)`는공유tensor를인식하는공식loader이며modelprefix도그대로보존한다. 위실행예를이방식으로정정했다. `strict=False`로missing을무시하거나임의zero를넣는방식이아니다. 실제missing/unexpected0 및전체parameterload는부모runtime결과로확인필요.

InputProcessor 문서와달리실제`_stack_batch`는`torch.stack(processed_images)`만호출하므로(N,3,H,W)이다. batch축을한번추가해야한다.
