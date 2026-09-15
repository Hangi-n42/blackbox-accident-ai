# Stage 1: 재녹화 판별 전략 검토

확인일: 2026-09-11. 범위: 로컬 대회 통합 문서·baseline 코드 및 아래 1차 공개 출처. 1~6절은 초기 연구 가설이며, 이후 실제로 수행한 다운로드·학습·검증은 7절에 구분 기록했다. 외부 데이터 신청은 수행하지 않았다.

## 1. 점수 및 규칙에서 출발한 결론

로컬 `대회_통합_정보.md`에 따르면 Stage 1은 별도 기기로 화면을 재촬영했는지 판별하는 문제이며, 두 클래스의 Macro-F1이 종합점수의 20%다. Public 대표점수만 최적화하면 종합점수의 나머지 80%를 놓친다. Stage 1이 0.01 개선되면 나머지 조건이 같을 때 종합점수는 0.002 개선된다. 단순 화면 녹화·재압축을 전부 재녹화로 라벨링하면 과제 정의를 위반한다.

baseline은 MViTv2-S를 `weights=None`으로 만들고 영상 10개로 학습한다. 검증셋 없이 마지막 가중치를 `best.pt`에 저장한다. 공식 예제의 재녹화 5개는 실제 재촬영이 아닌 모사본이다. 이 5쌍만으로 모델 선택·임계값 조정을 반복하면 합성 파이프라인 식별기를 선택할 위험이 있다.

**권장 가설:** 공개 실제 재촬영으로 사전학습한 판별기 + 원해상도 주파수·잡음 특징 + 화면 주변의 시각 단서를 각각 시험한다. 검증되지 않은 거대 랜덤 초기화 영상모델보다 작은 전이학습 모델의 실험 우선순위를 높인다. 최고점 가능성을 보장하는 해법은 확인되지 않았다.

## 2. 외부 후보: 접근성과 이용조건

| 후보 | 확인한 사실 | 현재 판단 |
|---|---|---|
| TPO / FoundPAD | 비인물 물체의 원본·인쇄·화면 재촬영을 이용한 CLIP ViT-B/16+LoRA 모델. 공개 코드와 1.2MB 어댑터; backbone 약350MB 필요. 프로젝트 CC BY-NC-SA 4.0. 데이터는 별도 신청. | 가장 빠르게 실험 가능한 사전학습 후보. 도로영상 전이 검증 필수. 모델 라이선스 고지 보존 및 변경내역 기록. |
| DLC-2021 | 논문은 1,424개 문서 영상 및 화면 재촬영 범주, CC BY-SA 2.5 공개 배포를 명시. | 실제 화면 재촬영 질감의 보조 학습·검증 후보. 문서 도메인과 도로 도메인 차이가 크다. Zenodo 개별 자산 접근은 이번 확인에서 실패했으며 다운로드 성공 미검증. |
| ROSE Recaptured Images | 공식 사이트: 자연 이미지 2,000개, 재촬영 2,700개, 약17GB. 교육·연구기관 연구자의 비상업 학술연구로 제한; 별도 승인, 파생셋 생성·재배포 제한. | 일반 참가자의 대회 활용권을 자동으로 인정할 수 없다. 승인 없이 도입하지 않는다. |
| Imperial DS-05 | 공식 사이트에 전체 약8GB와 100장 예제 다운로드 링크 및 논문·코드 링크 제공. | 공개 접근 링크는 확인했으나 사이트에 명확한 사용 라이선스가 보이지 않아 대회 사용 허용 여부 확인 불가. 바로 학습에 쓰지 않는다. |
| Chimera의 recapture detector | 연구자 GitHub MIT 코드, Zenodo에 데이터 3.8GB·모델 묶음6.9GB 공개 링크. | detector 부분만 검토 가능. 묶음의 개별 학습데이터·가중치 권리가 코드 MIT와 같은지 별도 확인 필요. 전체 다운로드 비용도 큼. |
| RISA HF 모델 | 모델 카드상 DS-05의 86개 이미지·12개 콘텐츠 그룹 학습, raw-image 특징 추출 코드 미공개, `other` 라이선스. | 완전한 이미지 추론 파이프라인이 없어 즉시 사용 불가. 작은 데이터 성능을 도로영상 성능으로 인용하지 않는다. |

출처: [TPO 연구자 저장소](https://github.com/gurayozgur/TPO), [TPO 라이선스 원문](https://github.com/gurayozgur/TPO/blob/main/LICENSE), [DLC-2021 원논문](https://pmc.ncbi.nlm.nih.gov/articles/PMC9323793/), [ROSE 공식 데이터 페이지](https://rose1.ntu.edu.sg/dataset/recapturedImages/), [Imperial 공식 페이지](https://www.commsp.ee.ic.ac.uk/~pld/research/Rewind/Recapture/), [Chimera 연구자 저장소](https://github.com/ssysarch/Chimera), [Chimera 배포 기록](https://zenodo.org/records/14736478), [RISA 작성자 모델 카드](https://huggingface.co/HanyueShen/risa-recaptured-image-detection).

TPO 연구자는 얼굴 PAD 데이터셋 사이의 전이를 보고했지만, 그 AUC 수치는 DACON의 영상 Macro-F1이 아니다. 실제 주행영상 성능과 임계값은 확인 불가다. TPO는 인쇄 공격도 학습하므로 출력 attack을 화면 재촬영 확률로 그대로 해석하는 것도 검증 가설이다.

## 3. 구현 가능한 가설과 반증 조건

### H1. 원해상도 잡음·주파수 + 저용량 분류기

- 영상에서 시간상 고르게 프레임을 고르고, 확대·축소 없이 여러 위치의 고정 크기 패치를 사용한다.
- 고역통과 잔차의 2D FFT peak/방향성, 행·열 주기성, 색채 잔차 상관, edge spread/날카로움, 밝기 clipping 등을 측정한다.
- 프레임/패치 특징을 영상 내부 평균·중앙값·상위 분위수로 요약하고, 정규화된 logistic regression 또는 작은 gradient-boosting 모델을 비교한다.
- 반증: 외부 실제 재촬영, 재압축된 원본, 야간·비·울타리 등 반복 패턴에서 오류가 증가하면 해당 특징 가중치를 축소하거나 제외한다.

주파수 단서는 물리적 근거가 있지만 필수조건은 아니다. Imperial 연구는 화면 픽셀과 센서의 샘플링 관계에 따라 aliasing이 없어질 수 있음을 보여주므로, moiré 부재를 ORIGINAL의 증거로 확정해서는 안 된다. [원논문](https://www.commsp.ee.ic.ac.uk/~pld/publications/2013_ICASSP_Muammar.pdf)

### H2. 공개 TPO 어댑터의 영상 단위 전이

- 필요한 CLIP backbone과 어댑터를 로컬에 고정하고 평가 시 자동 다운로드를 제거한다.
- 영상당 균등 샘플 프레임에 대해 전체 프레임과 내부 crop을 별도로 평가한다. 전체화면은 테두리·주변 환경 단서, 내부 crop은 화면 질감 단서를 제공한다.
- 영상 내부에서만 확률을 집계한다. 처음에는 고정 threshold를 사용하며 공개 검증만으로 보정한다.
- 반증: 원본 도로영상 대부분을 attack으로 분류하거나, 공개 모사본만 분리하면 단독 제출모델로 채택하지 않는다.

### H3. 합성 재촬영 증강은 실제 재촬영의 보조 역할

- 허용된 원본 도로영상을 화면 픽셀 격자, 원근변환, blur, 카메라 흔들림, 반사·주사무늬, 색 전달, 재인코딩의 다양한 조합으로 변환한다.
- 단일한 고정 합성 파이프라인을 쓰지 않는다. 최소한 source-content와 augmentation family를 별도로 holdout한다.
- 원본 클래스에도 정상적인 크기변경·압축·색변화를 적용하여 재압축 그 자체를 정답으로 학습하는 지름길을 억제한다.
- 결과는 끝까지 simulated-recapture 라벨로 기록하며 실제 촬영 자료로 보고하지 않는다.
- 반증: synthetic 검증은 높지만 actual recapture 검증이 낮으면 합성량 증대 대신 실촬영의 장치 다양성을 우선한다.

### H4. 낮은 해상도 시각 표현과 고해상도 forensic 특징의 결합

- H1과 H2를 단독으로 평가한 뒤, 공개 검증에서 오류가 상보적일 때만 고정 가중 결합을 적용한다.
- 복잡한 stacking은 5개 원천 그룹에 적합시키지 않는다. 검증 자료 부족 시 단순한 두 후보 비교가 더 해석 가능하다.
- 추론 중 모델·threshold·정규화 통계를 갱신하지 않고 모델 파일의 고정값을 사용한다.

## 4. 검증 설계

1. baseline 5개 원천 영상에 대해 원본·공식 모사본·추가 생성본·모든 추출 프레임을 같은 group에 묶는다.
2. Leave-One-Source-Out 5회로 out-of-fold 영상 예측을 만든다. 이 결과는 합성 예제에 한정된 진단이며 실촬영 일반화 점수가 아니다.
3. 외부 실제 재촬영을 확보하면 내용 분리 외에 촬영기기·디스플레이·촬영 세션을 통째로 holdout하는 별도 평가를 한다.
4. Macro-F1은 최종 영상 단위로 측정한다. 프레임 수가 많은 영상이 점수나 임계값 선정에 과도한 영향을 주지 않게 한다.
5. threshold 후보 선택은 내부 grouped split에서 끝낸 뒤 외부 holdout을 평가한다. 같은 holdout으로 반복 튜닝하면서 독립 검증으로 보고하지 않는다.
6. 재압축 원본, 화면 테두리 없는 재촬영, 심한 야간·노이즈 원본을 각각 오류 목록으로 남긴다. 파일명·컨테이너 encoder 태그·원본 해시를 예측 특징으로 사용하지 않는다.

그룹별 종속 표본을 학습·검증에 섞지 않는 원리는 scikit-learn의 grouped CV 설명에 근거한다. 그룹이 적으면 fold 평균/분산만으로 일반화 신뢰도를 확정할 수 없다. [공식 CV 문서](https://scikit-learn.org/stable/modules/cross_validation.html)

## 5. 제출 정책

- 한 평가 파일의 prediction은 그 파일과 고정 모델만으로 결정한다. 다른 파일의 점수분포, 랭크, 클래스 비율로 threshold를 바꾸지 않는다.
- 외부 리더보드 점수는 제출모델 평가 기록으로만 사용하고 비공개 입력을 복원·탐색·추가학습하지 않는다.
- `requirements.txt`, backbone, adapter, 설정, 재현 가능한 학습/추론 코드, 라이선스 고지를 함께 보존한다.
- 모든 후보의 로컬 점수·실행시간·제출별 Stage 점수를 기록하고 전체 가중합으로 선택한다.

## 6. 우선 실행 순서

1. 실제 baseline 샘플을 시각 점검하고 Group-OOF 특징 모델을 만든다.
2. TPO 라이선스와 로컬 오프라인 로딩을 확인하여 frozen 모델의 공개 예제 판별을 측정한다.
3. 합성 모사본 결과와 실재촬영 결과를 분리 기록하고, 확보 가능한 DLC 자료로 도메인 외 검증을 보완한다.
4. 성능이 검증된 단순 후보를 첫 제출에 넣고, Stage 2·3 개선 여력과 전체 처리 제한을 함께 최적화한다.

확인되지 않은 최고점 또는 예상 Private 점수를 제시하지 않는다. 최종 Private 순위/점수의 공개 시기는 주 대회 일정에 따른다.

## 7. 구현과 실제 공개 예제 실험

`solution/stage1.py`는 영상당 균등12프레임, 프레임당 native192px 패치5개를 사용한다. 품질8개·주파수11개 특징의 영상내 median/q90 및 화면주변12개 통계를 만든다. 학습은 고정 C=.1 logistic regression이며 원천5그룹 LeaveOneGroupOut으로 검증했다. 파일명은 개발시 그룹지정에만 쓰고 추론 특징에는 쓰지 않는다. 클래스양쪽에 같은 JPEG75/95와 half-resolution 변형을 추가한 후보도 비교했다.

| 후보 | 원래 크기 OOF Macro-F1 | JPEG75 변형 | 가로세로 절반 변형 |
|---|---:|---:|---:|
| 전체 특징, 원래 자료 학습 | 0.8000 | 0.8000 | 0.4505 |
| 주파수 특징, 원래 자료 학습 | 1.0000 | 1.0000 | 0.3333 |
| 주파수 특징, 동일 클래스보존 증강 | 1.0000 | 1.0000 | 0.4505 |
| TPO 동결 사전학습모델 | 0.6970 | 0.6970 | 0.6970 |
| TPO 50% + 증강 주파수 50% | 1.0000 | 1.0000 | 0.6970 |

TPO 행은 DACON 예제로 학습하지 않은 동결모델 결과다. 나머지 학습모델의 검증 영상 원천은 해당 fold 학습에 포함하지 않았다. 다만 변형종류와 결합가중치 후보를 이 공개 예제로 비교했으므로 위 수치는 최종 독립 holdout 성능이 아니다. 특히 실제 화면 재촬영의 주행영상에서 검증한 결과가 없으며 Public/Private 점수도 아니다.

**반증된 가설:** 주파수 특징 단독의 1.0은 해상도 변화에서 유지되지 않았다. 따라서 기본 후보는 TPO와 증강 주파수의 고정 동등결합으로 설정했다. 이 후보가 실제 평가에서 가장 좋다는 주장은 하지 않는다.

TPO 코드·어댑터와 OpenAI CLIP 코드·ViT-B/16 가중치를 공식 배포원에서 받았다. CLIP SHA256이 공식URL의 checksum과 일치했다. 원본 라이선스·변경경계·소유권 구분은 `model/stage1/tpo/THIRD_PARTY_NOTICE.md`, 파일별 hash는 `asset_manifest.json`에 있다. 제3자 소스와 가중치는 수정하지 않았고, 별도 wrapper가 로컬파일 경로와 고정 RGB 전처리를 설정한다. 최종 제출 추론은 네트워크를 사용하지 않는다.

실험 산출물:

- `train_stage1.py`: 고정소스 OOF 및 원자료 후보학습.
- `evaluate_stage1.py`: 클래스보존 변형 강건성 및 증강후보학습.
- `evaluate_stage1_tpo.py`: 동결TPO 원자료·변형·crop 진단.
- `evaluate_stage1_vlm.py`: 동일프롬프트 시각근거 판정용; 실행상태는 별도 결과파일로 확인.
- `research/stage1/group_oof_metrics.json`, `robustness_metrics.json`, `tpo_judgments.json`, `fixed_blend_diagnostic.json`: 실제 점수와 영상별 예측.
- `research/stage1/training_manifest.json`: 공개baseline 학습자료별 hash/원천그룹/모사여부.
- `model/stage1/config.json`: 제출후보의 고정조합 설정.

## 8. 제출용 LoRA 병합 및 동등성 검증

제출판은 `solution/stage1_tpo_merged.py`를 사용한다. `scripts/export_tpo_merged.py`가 공식 CLIP 시각 가중치에 TPO checkpoint의 고정 q/v 업데이트를 `W += (alpha/sqrt(rank)) * B @ A`로 병합했다. 이 변환기는 TPO·FoundPAD·CLIP-LoRA·loralib 코드를 import하지 않는다. 새로운 학습·최적화는 수행하지 않았다.

추론에는 OpenAI MIT `clip_source/clip/model.py`의 VisionTransformer와 병합 가중치·분류 head만 필요하다. 원본 TPO 구현·LoRA 라이브러리·CLIP tokenizer·미병합 backbone/adapter는 개발 폴더에 보존하되 제출에서는 제외한다. TPO의 CC BY-NC-SA 가중치 조건과 병합변경 고지는 그대로 유지하며 원본 OpenAI MIT 고지도 포함한다.

CUDA에서 공개 10영상 × 12프레임 × 4변형 = **480프레임**을 기존 구현의 저장된 실측과 대조했다. 최대 확률 절대차는 **0.0000014901161193847656**, 사전에 지정한 허용오차 0.00005 이내였으며 영상단위 **40개 판정이 모두 동일**했다. 결과는 `research/stage1/merged_equivalence.json`에 기록했다. 이는 구현 동등성 검증이며 실제 재촬영에 대한 추가 성능 검증은 아니다.

병합 가중치: `model/stage1/tpo/merged_visual.pt`, 344,823,499 bytes, SHA256 `2935322b6984ba7c3f314607ae8fdf1145d2d3d7671ede13b4b324659a5c51a9`. 출처·변환·권리 고지는 `MERGED_WEIGHTS_NOTICE.md`, 원본과 출력 해시는 `merged_manifest.json`에 있다.
