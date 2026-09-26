# Stage1 입력 민감성 및 분기 진단

2026-09-17. 기존 VDmoire 개발 표본 5원천을 사용했다. 원본·운영 코드·모델은 변경하지 않았고 학습, 추가 다운로드, 전체 영상 추론을 하지 않았다. 제공 clean-reference를 대회 ORIGINAL 정답으로 변환하지 않았다.

## 실행 및 재사용 근거

기존 `scripts/data/score_vdmoire_baseline.py`는 repository solution을 직접 import한다. 현재 Mac 공식 실행은 `scripts/migration/run_stage.py` → `artifacts/submissions/verify_v6/inference.py` → `model/stage2/code/solution/stage1_v4.py`이며 이 release의 특징 추출·TPO 모듈을 사용한다. 예전 pilot 결과에는 완전한 당시 코드·모델 해시가 없으므로 동일성을 단정하지 않고 **같은 5원천만 현재 release 경로에서 재계산**했다.

설정·가중치·전처리·실행 파일 해시는 `provenance.json`에 저장했다. 이전 pipeline diagnosis freeze에 존재하는 대조 항목의 불일치는 0이다. 현재 모델은 forensic/TPO 각각 0.5, 임계값 0.5다. RGB 원본과 중앙 native 192×192 crop을 비교하며, TPO 내부에서는 모델에 저장된 크기로 bilinear resize와 저장 mean/std 정규화를 수행한다. 두 분기의 기존 점수 및 평균 방식은 그대로 사용한다.

5원천 × 제공 클래스 2개 × 2프레임 = **20개 고유 프레임**. 각 full/crop 조건의 10개 집계 결과, 총 20개 결과를 계산했다. 단 2개 프레임으로 구성한 개발 진단이며 운영 영상의 12프레임 샘플링·시간축·디코딩 전체 동작을 재검증한 결과가 아니다. 새 재계산과 예전 ensemble 확률 최대 차이는 **1.11e-16**이다. 따라서 해당 기존 수치는 재현됐다. 실행 exit 0, 계산 1.76초. 전체 20원천으로 확장하지 않았다.

## 분기별 결과

아래 분기 0.5 기준은 독립 채택 기준이 아니라 동일한 확률 기준에서 동작을 설명하기 위한 진단이다.

| 재촬영 표본 5개 | full frame | 중앙 192 crop |
|---|---:|---:|
| 현재 ensemble 미탐 | 0/5 | 4/5 |
| forensic 확률 <0.5 | 5/5 | 5/5 |
| TPO 확률 <0.5 | 0/5 | 1/5 |
| ensemble 평균 재촬영 확률 | 0.58322 | 0.41154 |

| 원천 | full ensemble | crop forensic | crop TPO | crop ensemble |
|---|---:|---:|---:|---:|
| Reds/video_16 | .54345 | .21107 | .55232 | .38169 |
| Reds/video_193 | .69817 | .20788 | .90893 | .55841 |
| Reds/video_161 | .58703 | .20921 | .75375 | .48148 |
| Reds/video_117 | .54165 | .27420 | .56374 | .41897 |
| animal/video_2 | .54580 | .14911 | .28516 | .21713 |

**관측된 구체 약점은 내부 패치에서 낮은 forensic 확률이 TPO와 평균되며 재촬영 판정을 뒤집는 현상이다.** crop의 4개 미탐 중 3개는 TPO만 0.5 이상이지만 ensemble은 0.5 미만이다. animal/video_2는 두 분기가 모두 낮다. 따라서 모두 TPO 실패라고 설명할 수 없다. full 조건에서도 forensic은 5개 모두 낮으므로 crop 때문에만 forensic이 실패한 것도 아니다.

이것은 forensic 제거·TPO 가중치 증가가 실제 대회 성능을 높인다는 증거가 아니다. clean-reference의 원본 판정 신뢰성이 불충분하여 실제 ORIGINAL 오탐 증가를 여기서 평가할 수 없기 때문이다. 이 보고서는 공식 Macro F1을 산출하지 않았다. 참고로 공식 공개 10개 만점은 별도 기존 개발 회귀 기록이며 도로 일반화의 보증이 아니다.

## 인과 해석의 한계

- full/crop은 화면 테두리, 장면 범위, 스케일과 주파수 내용, TPO 내부 확대 비율을 함께 바꾼다. 테두리만의 의존성 또는 모아레만의 원인으로 확정할 수 없다.
- 5개 원천, 각 클래스 2프레임, 기존 개발 노출, 단일 촬영 기기 표본이다. 새 독립 평가가 아니다.
- 제공 clean-reference와 재촬영의 픽셀·시간 대응이 인증된 것은 아니다. clean을 카메라 ORIGINAL로 쓰지 않는다.
- 임계값이나 앙상블 비율을 이 5개 결과에 맞춰 조정하지 않았다.

## UHDM 및 다음 작업 판단

현재 관측을 확인하는 데에는 새 UHDM 계산이 필수적이지 않다. 따라서 **이번 UHDM 추가 정제/추론/학습은 0쌍**이다. 기존 검수된 50쌍 native 중앙 패치는 다음 단계에서 내부 공간 흔적의 보조 진단에만 사용 가능하다. 4,500쌍 전부 정제할 근거는 없다.

다음 Stage1 실험 후보는 검수된 UHDM 50쌍에서 현재 두 분기의 확률과 제공 pair 내 변화가 어떻게 나타나는지 고정 조건으로 확인하는 것이다. 이것은 보조 paired 진단이며 clean=ORIGINAL 또는 독립 도로 평가로 해석하지 않는다. 실제 ensemble 변경의 진입 조건은 **카메라 원본으로 확인된 도로 자료에서 오탐 회귀를 검사할 수 있고**, 재촬영 내부 패치 약점이 여러 원천에서 반복되는 것이다. 현재는 모델 학습·가중치 조정을 승인할 근거가 부족하다.

## 산출물과 재현

- `analyze.py`: 기존 5원천만 재계산하고 간단한 수량 검증 수행. Python은 `artifacts/mac_experiments/scipy_compat/.venv/bin/python`.
- `provenance.json`: 현재 파일 해시, 이전 동결 기록 대조, 설정 및 제한.
- `predictions.json`: 원본 경로·제공 라벨·분기 확률·기존 대비 차이·개발 노출.
- `branch_comparison.json`: 원천별 full/crop 확률과 변화량.
- `summary.json`: 수량·미탐·시간·수치 재현 차이.

원본 이용 조건·출처는 `artifacts/data_pilot_20260916/vdmoire/manifest.json`, `clean_manifest.json`의 공식 원천 링크를 상속한다. 본 출력은 예측 분석이며 데이터 라이선스를 새로 선언하거나 대회 공식 정답으로 승격하지 않는다.
