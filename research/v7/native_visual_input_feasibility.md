# Qwen native video 입력의 제한된 실행 가능성 감사

작성: 2026-09-15. 기존 코드·모델·라벨 수정 및 GPU 실행 없음.

**결론: 명시적으로 합성 재생 시간을 사용하는 native video 접촉 선택 진단은 실행 가능하다.** 물리 FPS가 없다는 이유만으로 대회에서 금지된다고 판단할 근거는 없다. 다만 이는 원래 촬영 시간의 복원도, 완전히 동일한 입력의 단순 포장 대조도 아니다. 합성 재생축·두 프레임의 temporal patch·영상 모달리티 토큰을 함께 도입하는, 아직 시험하지 않은 한정된 구조 가설이다. 기존 9개 개발 영상의 고정된 V6 두 번째 질문만 대상으로 한 번 시험할 것을 권한다. 결과로 바로 제출 채택하지 않는다.

## 이미 시험한 것과 새 메커니즘

- `research/v6_stage2/contact_tiles_v6b_protocol.json` 및 `contact_tiles_v6b_development/evaluation.json`: V6A의 동일 후보·타일·질문을 모자이크 한 장에서 최대 18개 개별 이미지로 바꾼 V6B를 실제 실행했다. 접촉 성공 1/3→1/3, MAE 2.834602076124567초로 동일하여 gate 실패했다. 따라서 “개별 이미지가 미시험”이라는 전제는 틀리다. 이 실패는 보존하며 반복하지 않는다.
- 실제 V6 `artifacts/submissions/verify_v6/model/stage2/code/solution/vlm.py:26`의 `ask`는 모든 항목을 `type=image`로 넣고 `processor(images=...)`를 호출한다(37~42행). V6 `stage2_v2.py:77,92,106,119` 및 V7 `research/v7/solution/stage2_v7_grounded.py`는 이 경로에 contact sheet를 전달한다. 별도 이미지도 video 모달리티가 되지 않는다.
- 설치된 Transformers 4.57.6의 `video_processing_qwen3_vl.py:99,238`은 서로 다른 인접 입력 두 프레임을 한 temporal patch로 묶는다. `processing_qwen3_vl.py:197`은 `video_grid_thw`를 사용하여 각 temporal patch 앞에 timestamp 문자열을 넣는다. 이는 이미지 경계만 분리했던 V6B와 실제 계산 경로가 다르다.

Qwen3-VL 공식 기술 보고서 §2.3은 temporal patch 앞의 텍스트 timestamp를 video 시간 표현으로 설명한다. 이는 이 모델의 구조에 관한 근거이지, 우리 사고 시점 선택이 개선된다는 근거는 아니다. [Qwen3-VL Technical Report](https://arxiv.org/html/2511.21631v1)

## 실제 API 제약과 합성 시간의 의미

설치된 `processing_qwen3_vl.py:201~225,314~325`는 `frames_indices / fps`를 계산하고 두 프레임 시각의 평균을 `<x.x seconds>`로 만든다. FPS 누락 시 24를 사용한다. `return_metadata=False`와 `do_sample_frames=False`는 이 timestamp 생성을 끄는 옵션이 아니다. `video_utils.py:81~107`의 `VideoMetadata`에도 독립적인 native PTS 배열 입력 필드는 없다. `modeling_qwen3_vl.py:920~997`은 video의 시간 정보를 timestamp로 인코딩한다는 전제에서 위치를 만든다. 따라서 **설치 버전에 공식 ordinal-only 또는 timestamp 생략 옵션은 확인되지 않았다.**

그러나 다음 metadata는 유효한 **합성 재생축**이다. `fps=2.0`, `frames_indices=[0,...,K-1]`, `total_num_frames=K`. 원본 영상의 FPS·프레임 간 실제 간격을 주장하지 않는다. 원본 파일 번호를 이 `frames_indices`로 넣으면 안 된다. 입력 타일에 이미 있는 `frame ORIGINAL_ID` 픽셀 라벨과 별도 기록만 원본 번호의 근거로 사용한다. 출력 시각 평가에는 기존 native PTS만 사용하며 synthetic seconds를 절대 사용하지 않는다.

공식 문서는 이미지와 video 모달리티를 구분하며 video tensor와 metadata를 processor에 전달하고, 이미 조정한 입력에 `do_resize=False`를 사용하는 경로를 제공한다. 공식 이미지 목록 video 예제의 sample FPS 역시 재생 시각 구성에 관여한다. 따라서 합성 재생축을 명시하는 것은 실행 가능한 입력 설정이며, 촬영 FPS를 알아냈다는 뜻이 아니다. [공식 Qwen3-VL utils](https://github.com/QwenLM/Qwen3-VL/blob/main/qwen-vl-utils/README.md), [공식 사용 예제](https://github.com/QwenLM/Qwen3-VL)

## 제안하는 단 한 번의 고정 진단

1. **대상/범위:** 기존에 노출된 개발 9개 전부. V6 call 2의 저장된 후보 번호·순서·원본 PNG·질문·48 출력 토큰·4B NF4 FP16 compute 모델을 고정한다. 새 후보 창, GT에 따른 입력 선택, 후속 질문 재실행은 없다. 기존 call 2 원응답이 control이다. 신규 native video는 영상당 한 번, 총 9회다. 최종 제출 policy의 4-call 재실행이나 개선이라고 부르지 않는다.
2. **입력:** 기존 `_sheet`의 각 384×256 타일을 원래 순서대로 분리한다. 원본 번호가 쓰인 머리글과 전체 장면을 보존한다. 동일 크기로 사전 축소한 타일들을 uint8 `T,C,H,W`로 쌓아 `videos=[tensor]`로 전달한다. 빈 모자이크 칸은 프레임으로 넣지 않는다.
3. **모달리티:** chat content에는 `type=video` 한 개와 질문을 넣는다. `processor(..., videos=[tensor], video_metadata=[metadata], do_sample_frames=False, do_resize=False, return_tensors='pt')`를 사용한다. `images=` 경로를 사용하지 않는다. 임의 `<frame>` special token이나 내부 positional embedding 패치는 만들지 않는다.
4. **문구:** 원래 접촉 정의·JSON 키·후보 번호는 그대로 둔다. 공간 배열을 설명하던 첫 문장만 다음 고정 선언으로 대체한다: `These selected dashcam frames are in chronological order on a synthetic playback timeline at 2 frames per second. Video timestamps are presentation positions, not physical event times. Return an original frame number printed on an image.` 이 문장 교체까지 포함한 입력 표현 대조이며, “prompt byte-identical”이라고 기록하지 않는다. 결과를 보고 재생률·문구를 바꾸지 않는다.
5. **번호/파서:** 원본 번호만 허용한다. JSON integer이 실제 제공 후보에 없거나 null/불량이면 원래 V6 최종 motion frame을 유지하며 fallback으로 별도 집계한다. seconds를 원본 번호로 역변환하지 않는다. 타일 K가 홀수면 processor가 마지막 프레임을 한 번 복제하는 사실과 원본 번호를 기록하고, 복제는 새 후보가 아니다. 영상마다 완전히 독립적인 단일 질문이다.
6. **비교:** 제공 후보의 ±0.3초 포함 여부와 후보 중 VLM 선택의 성공을 분리한다. 선택 교정≥1, 기존 성공 손실 0, MAE 비증가라는 기존 방향을 유지한다. invalid/default 수와 원응답도 보고한다. 최종 contact를 교체한 가상 결과는 다른 세 필드를 복사한 selector 진단이라고 명시한다. 전체 S2 개선·독립 holdout·원인 확정·즉시 채택으로 승격하지 않는다.

## 먼저 수행할 작은 CPU 계약 검사

이는 **제안이며 이번 감사에서 구현/실행하지 않았다.** 모델 가중치를 로드하지 않고 frozen processor만 로드한다. 서로 구분되는 네 개의 작은 RGB 타일과 원본 번호 `[7,19,31,55]`를 사용한다. 합성 metadata의 `frames_indices`는 `[0,1,2,3]`, FPS는 2이다.

- `do_sample_frames=False` 상태에서 입력 순서와 개수가 보존되고 `pixel_values_videos`/`video_grid_thw`가 생성되어야 한다. `pixel_values` image 경로이면 실패다.
- K=4에서는 temporal grid T=2, 생성 텍스트는 `<0.2 seconds>`, `<1.2 seconds>`여야 한다. 이는 각각 0.25, 1.25의 현재 `.1f` 표현이며 실제 촬영 시각이 아니다. K=3은 마지막 프레임 복제와 두 temporal patch를 확인한다.
- 실제 9개 입력에 대해서도 전처리까지만 수행하여 타일 SHA/순서/번호, 자동 추가·제거 프레임, 실제 grid/visual token 수, 전체 input token 수를 검사한다. 기본 2fps 재표본화가 다시 작동하거나 metadata의 24fps fallback 경고가 발생하면 GPU 실행 전 중단한다.
- 총 유효 픽셀 예산은 홀수 마지막 복제까지 포함하여 `ceil(K/2)*2*H*W <= 1,200,000`으로 고정한다. H/W는 32 배수로 내리고 모든 타일에 같은 크기를 쓴다. video 기본 최소 해상도를 그대로 적용하면 예산이 달라지므로 사전 축소 후 `do_resize=False`를 검증한다.
- video visual token은 `video_grid_thw.prod()/4`로 직접 센다. 기존 call 2와 동일한 상한을 넘으면 중단한다. 두 프레임 압축 때문에 실제 token 수가 같지 않을 수 있으므로 “상한 동일”과 “실제 수 동일”를 구분한다. 생성 상한은 48 그대로다.

## 자원·해석상 한계와 중단 규칙

4B 가중치는 기존과 같고 입력 상한도 통제할 수 있으므로 CPU 준비 후 첫 고정 영상의 단일 GPU 호출을 시험할 실행 근거는 있다. **2080 SUPER 8GB의 native video 실측 메모리/시간은 아직 확인하지 않았다.** 기존 7.5GiB 메모리 한도 및 1.25 시간비 기준을 그대로 점검하고 OOM/초과이면 종료한다. precision·offload·픽셀 상한을 결과 뒤 바꾸어 같은 실험으로 재시도하지 않는다. 단일 call의 시간 통과는 전체 제출 60분 통과를 증명하지 않는다.

선택 간격이 큰 두 프레임도 하나의 temporal patch가 되어 실제로 존재하지 않은 빠른 이동처럼 보일 수 있다. 원본 번호 머리글 두 개가 동시에 합쳐져 판독이 어려워질 수도 있다. 합성 2fps 선택은 최적값이 아니라 사전 고정된 재생 규칙이며 실제 속도/접촉 시각 증거가 아니다. 또한 모자이크 대비 resize·빈 칸·timestamp 텍스트가 함께 달라져 결과가 좋아져도 temporal convolution만의 인과효과로 분해할 수 없다.

그럼에도 이 대조는 실패한 multi-image 반복과 달리 **정식 video tensor와 temporal patch 경로가 사용되는지**를 처음 시험한다. 실패하면 같은 9개에서 FPS·문구·타일 크기를 재탐색하지 않는다. 통과하더라도 알려진 후보 누락은 해결되지 않으며, 별도 동결된 end-to-end/자원 검증과 새로운 검증 자료 없이 제출 채택하지 않는다. AI 검수는 계속 AI_secondary_evidence이며 human/official GT로 승격하지 않는다.
