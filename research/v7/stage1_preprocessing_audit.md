# Stage1 TPO 전처리 계약 감사

검사일: 2026-09-15. 범위: 공식 TPO 공개 소스, 보존된 원저자 코드, 현재 Stage1 실행 코드와 병합 가중치의 전처리 메타데이터. GPU·모델 추론·학습·라이브러리 설치·기존 파일 변경은 하지 않았다.

**결론: 조사한 범위에서 명백한 전처리 계약 불일치는 없다. 전체 프레임을 224×224로 강제 변환하는 현재 동작은 TPO 저자의 평가 코드와 일치한다. 이를 표준 CLIP의 중앙 자르기로 바꾸는 것은 버그 수정으로 정당화되지 않는다.**

## 1. 원저자 실제 처리와 현재 처리

TPO 공식 `src/dataset.py`의 학습·평가 경로는 모두 RGB 전체 프레임을 224×224로 바꾼다. 평가 경로는 Resize → CLIP 채널 정규화 → ToTensorV2이고 얼굴 검출이나 중앙 자르기는 없다. 학습 때만 뒤집기·노이즈·광도/색 변환을 추가한다. 보존된 파일과 현재 공개 페이지에서 이 내용을 각각 확인했다. [TPO 공식 dataset.py](https://github.com/gurayozgur/TPO/blob/main/src/dataset.py)

| 항목 | 원저자 로컬 소스 | 현재 실행 코드 | 판정 |
|---|---|---|---|
| 이미지 크기 | `source/src/dataset.py:39,51`: 높이·너비 각각 224 | `solution/stage1_tpo_merged.py:44`: `(size,size)` | 일치 |
| 보간 | Albumentations Resize 기본값 | OpenCV INTER_LINEAR 명시 | 일치 |
| 색 순서 | `source/src/infer.py:46`, `dataset.py:84`: BGR→RGB | `stage1.py:24,31` 및 `stage1_v4.py:29,86`에서 BGR→RGB 후 전달 | 한 번 변환, 일치 |
| 채널 평균 | `(0.48145466,0.4578275,0.40821073)` | 실제 병합 artifact에서 같은 값 확인 | 일치 |
| 채널 표준편차 | `(0.26862954,0.26130258,0.27577711)` | 실제 병합 artifact에서 같은 값 확인 | 일치 |
| 스케일/배치 | Normalize 후 HWC→CHW | `/255`, `(image-mean)/std`, HWC→CHW | 수학적 계약 일치 |
| 특징 벡터 | `source/src/models.py:33`: visual 후 L2 정규화 | `stage1_tpo_merged.py:48`: visual 후 L2 정규화 | 일치; 픽셀 정규화와 다른 단계 |
| 확률 의미 | 클래스 0=attack, 1=bonafide | `softmax[:,0]`를 재촬영 확률로 사용 | 인덱스 반전 오류 없음; PAD attack과 대회 재촬영의 의미 차이는 별도 문제 |

원저자 `models.py:62`는 `clip.load`가 반환하는 표준 CLIP 전처리를 버리고 시각 인코더만 사용한다. 따라서 CLIP 기본 Resize+CenterCrop을 기준으로 현재 코드를 오류라 판단하면 비교 기준 자체가 틀린다. `infer.py:37`은 `build_transforms(train=False)`를 호출하므로 평가 시 학습 증강 누락도 오류가 아니다.

원저자 requirements는 `albumentations>=2.0,<3.0`이다. 버전 2.0.8의 공식 소스에서 Resize 기본 보간은 INTER_LINEAR, 축소 시 INTER_AREA 자동 전환 옵션은 기본 None임을 확인했다. [Albumentations 2.0.8 Resize](https://github.com/albumentations-team/albumentations/blob/2.0.8/albumentations/augmentations/geometric/resize.py)

같은 버전의 Normalize 기본값은 `max_pixel_value=255`, standard 방식이며 `(img-mean*255)/(std*255)`를 적용한다. ToTensorV2는 채널 축과 텐서 표현만 바꾸며 추가 `/255`를 하지 않는다. 따라서 현재 수동 전처리에서 정규화의 중복·누락은 확인되지 않았다. [Normalize 구현](https://github.com/albumentations-team/albumentations/blob/2.0.8/albumentations/augmentations/pixel/transforms.py), [ToTensorV2 구현](https://github.com/albumentations-team/albumentations/blob/2.0.8/albumentations/pytorch/transforms.py)

## 2. forensic 경로는 TPO 입력과 독립적이다

`solution/stage1.py:70`은 입력 RGB 프레임에서 다섯 위치의 `min(192,h,w)` 정사각 패치를 원래 픽셀 간격으로 잘라낸다. 이 패치에 224×224 변환은 적용되지 않는다. `:38`에서 패치 복사본을 float32 `/255`로 만들고 RGB2GRAY를 사용해 주파수·잔차 특징을 계산한다. CLIP 평균·표준편차는 이 경로에 적용되지 않는다.

별도의 테두리 특징 경로 `:79`만 320×180 INTER_AREA를 사용한다. `:95`의 학습 artifact 평균/scale은 추출된 특징 벡터를 표준화하며, CLIP 픽셀 정규화와 중복되는 연산이 아니다. `predict_stage1`은 forensic 추출 뒤 같은 원본 RGB 배열을 TPO에 넘기지만 특징 추출은 원본 배열을 수정하지 않는다.

CPU 2스레드 검사에서 고정 seed 753의 360×640 uint8 RGB 배열로 50개 특징을 추출한 뒤 입력 배열이 바이트 단위로 동일함을 확인했다. 실제 병합 가중치는 CPU mmap으로 메타데이터만 읽었으며 `size=224`, `color=RGB`, `interpolation=opencv_INTER_LINEAR`와 위 정규화 값, 클래스 순서 `['RERECORDED','ORIGINAL']`을 확인했다. 모델 생성/추론은 하지 않았다.

## 3. 수치 동일성과 일반화는 별도 한계다

- uint8 전 범위 0…255와 세 채널에 대해 현재 float32 수식과 공식 Normalize 소스의 곱셈 순서를 순수 NumPy로 비교했다. 최대 절대차는 `2.384185791015625e-7`이었다. 이는 수식의 반올림 차이 검사이며 설치된 Albumentations 전체 경로 실행 또는 저자 학습 환경과의 비트 동일성 증명이 아니다. 현재 환경에 Albumentations가 없어 새로 설치하지 않았다.
- 원저자는 OpenCV/Albumentations 세부 버전을 범위로 지정한다. 실제 학습 당시 모든 패치 버전·플랫폼의 비트 동일성은 확인 불가다. 이 한계에서 실질적 성능 오류가 발생했다는 증거도 없다.
- `research/stage1/merged_equivalence.json`의 480프레임 최대 확률차 `1.4901161193847656e-6`은 이전 자체 TPO wrapper와 병합 wrapper의 비교다. 양쪽이 같은 수동 전처리를 사용했으므로 이를 원저자 Albumentations 전처리와의 실측 동등성 증거로 확대하지 않는다.
- 224×224 축소가 미세 재촬영 흔적을 없애는지, PAD attack 확률이 실제 도로 화면 재촬영에 적합한지는 이 계약 감사가 해결하지 않는다. 원저자 계약 준수는 대회 도메인 적합성의 증명이 아니다. 실제 도로 원본/물리 재촬영 대응 자료 없이 중앙 자르기·native crop·정규화 변경을 채택할 성능 근거는 없다.

**권고:** 이번 감사에 따른 모델 또는 전처리 수정은 제안하지 않는다. 후속 독립 비교가 필요하면 동일 RGB 입력의 저자 eval transform과 현재 tensor를 직접 비교하는 CPU 계약 검사부터 수행하고, 공간 전처리 변경은 실제 물리 재촬영 자료에서 별도 후보로 검증해야 한다.

## 4. 검사한 로컬 파일 식별

경로는 작업공간 기준. 원저자 세 파일과 requirements의 실제 SHA는 기존 `model/stage1/tpo/asset_manifest.json` 기록과 일치한다.

| 파일 | SHA-256 |
|---|---|
| solution/stage1.py | f9def79197fdfc949cc01ec373b7a019f11d7c276e81fedb2015c2bc6f24eeb9 |
| solution/stage1_v4.py | 624fc836bc83b09cd273204206ae5b5f0de57ab19bc2b0a428e2957ef08d5c54 |
| solution/stage1_tpo_merged.py | e1ddcbd9956b2c3e2d9d0edee55d2162d3d164b16c6423f8968c176526ff5011 |
| model/stage1/tpo/source/src/dataset.py | c44dcd8eb25b529335add18a49bf6a69dcf7a17cb568e3188ef2a423bc3a2c95 |
| model/stage1/tpo/source/src/infer.py | 6e68badb8cf341344d9d885e090267cfbdff5a93c8cbeb0c1156a810598b401d |
| model/stage1/tpo/source/src/models.py | d4bde20f35915379de842f83af3162de02de7949547c79560f71dbc0332802f2 |
| model/stage1/tpo/source/requirements.txt | 7708c4e111cdedc4bc006385d207d2b15da7144af43ec3d097752d11b428a844 |
