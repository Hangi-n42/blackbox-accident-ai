# Stage3 비전·기하 전문가 검토 — 2026-09-20

읽기 전용 검토. stage3_tracker_foe_20260920/REPORT.md, point_motion_20260919/REPORT.md 및 measure.py, state_learning_20260919/REPORT.md·vjepa_provenance.json을 확인했다. 새 추론·학습·다운로드 없음.

## 결론

현재 **단일 FOE를 추정하고 역반경을 두 번 시간 미분하는 특징 개발은 보류**하는 것이 합리적이다. 다음 추적기 이름만 바꾸거나 FOE 기준을 완화하는 반복은 우선순위가 낮다. 이것은 영상 가감속 추정 전체의 불가능성, learned tracker 전체의 실패, 혹은 카메라 회전이 유일 원인이라는 뜻이 아니다.

실측 근거:
- 최초 28창 중 6창만 품질 통과. 검증 가속 0/2, 검증 속도매칭 유효쌍 0개. 현재 추가 특징의 점수 효용은 전혀 검증되지 않았다.
- 후속은 선택된 10창: LK 5/10→DIS 4/10. 기존 실패5개 복구0. 이 비율은 6/28과 직접 비교하면 안 된다.
- expanded_19 공통114개 시작점에서도 DIS의 반창 FOE 차이가 커져 실패. 새로운 나쁜 점 추가만의 문제로 환원할 수 없다.
- 궤적을 고정해도 FOE±5px에서 작은 특징 부호가 바뀐다. 추적 생존뿐 아니라 관측값→특징 변환의 민감성 문제가 별도로 존재한다.
- expanded_11 22.5초의 큰 오차는 작은 FOE 변화만으로 설명되지 않는다. 대수식 이상조건(고정 방향 병진, 회전 없음, 정적 배경, 일정 내부파라미터)이 실제 영상에서 충족되는지 검증되지 않았다.
- 현 구현은 21프레임 전부 살아남는 궤적, 수동 정적영역, start/end 직선 교점, quadratic fit u''/u'를 요구한다. 합성9조건 통과는 정확한 궤적을 넣은 수식시험이며 noisy camera motion 검증이 아니다.

## 후보 비교

1. LK→CoTracker 등 추가 추적기: 궤적 품질을 바꿀 가능성은 있지만 FOE와 역반경 민감성을 그대로 둔다. 이번 증거상 다음 우선 후보가 아니다.
2. 단일 FOE 고정/평활/품질기준 완화: 결과를 보고 맞추면 평가누수, 실제 회전과 병진 변화를 지울 수 있음. 진단 없이 권하지 않는다.
3. 기존 V-JEPA 공간 토큰 보존: 예전 실패는 ViT-B 86.8M, center crop384, 16frames@10Hz disjoint, 576공간토큰 평균768차원, 8시간토큰 보존이라는 설정에 한정됨. 모든 V-JEPA 방식 실패가 아니다. 하지만 현재 temporal head는 train F1≈1, 외부 날짜 악화였고 공간토큰 고차원화는 소량자료에서 추가 과적합 위험. 공간정보가 핵심누락이라는 별도 증거가 없어 1순위로 확정할 수 없다.
4. **DA3-Small 공동 geometry→카메라 이동 변화**: 원천정보 자체를 camera-center 궤적으로 바꾸므로 기존 단일 FOE/장기 코너 생존을 직접 요구하지 않는다. Stage3의 종방향 속도변화에 연결 가능한 출력이라는 장점. 다만 predicted pose의 2차 차분은 여전히 노이즈에 민감하고 depth/pose 모델은 가속도 분류모델이 아니다. 자동 성공이나 공식 점수 향상을 주장할 근거 없음.

## 권고하는 다음 단일 후보

**DA3-Small 한 모델을 동결하고, 2초 공동 추정 camera-center에서 얻은 scale-invariant 상대속도 변화 특징의 타당성을 한 번 검사. 통과할 때만 DIS864 + 같은 선형 분류기의 1개 특징블록 추가 대조로 이동.**

모델과 분류기를 함께 교체하지 않는다. encoder/geometry 추가만 후보로 고정하며, 여러 DA3 크기/pose head/시각창을 대회50정답에서 탐색하지 않는다.

단계:
1. 공식 모델·코드 버전/라이선스 고정, Mac 작은 CPU/MPS 스모크부터. 공식 README의 기본 예제는 CUDA이고 xformers 설치를 안내하므로 Mac 동작을 이미 보장하지 않는다. Small 0.08B는 Apache2.0, metric depth 모델이 아니다.
2. 기존 동일 28창, 같은 시간/프레임에서 각 창의 모든 시점을 **동시에** 입력. 입력 창마다 독립 단안depth를 계산해 서로 이어 붙이지 않는다. extrinsics의 t를 그대로 차량위치로 미분하지 말고 w2c [R|t]에서 camera center C=-R^T t로 변환한다.
3. 공동 좌표계의 C(t)에 고정 차수/시간창을 적합하고 tangential ratio q=(v·a)/(v·v)를 계산한다. 일정 양의 전역scale·회전에 불변이다. 장면/시각마다 달라지는 scale에는 불변이 아니므로 이것이 가장 먼저 반증할 조건이다. 속도0 근처는 미정으로 유지하며 STOPPED는 이 실험의 해석범위 밖.
4. 동일 프레임의 순서 반전/입력참조점 변화에 대해 좌표계 정렬 후 궤적 일관성과 q 변화를 검사한다. 분모가 작거나 궤적/시간 결과가 불안정하면 결측. 센서값으로 scale/pose를 맞추지 않는다. 센서는 평가에만 사용.
5. A/D 방향 일치, 등속 잔차, 같은속도 pair 구별, 기존 반전 사건에서 기존 DIS가 갖지 못한 정보가 있는지 검사. 유효창만 보고하지 말고 결측 포함 전체 커버리지도 보고.
6. 통과하면 DIS+선형 C/분할/학습자료/조향 고정, identity baseline vs 새 geometry feature block 한 쌍만 평가. 대회5video OOF 공식S3와 날짜별proxy F1, 사건별 새 반전을 모두 보고. 한 날짜/한 공개영상만 좋아지면 일반화 개선이라고 부르지 않는다.

중단 조건: (a) Mac 소규모 추론 불가/허용시간 초과, (b) 등속에서 일관된 가짜가감속 또는 기준프레임 변화로 q 부호가 흔들림, (c) 기존처럼 특정 A/D범주 대부분 결측, (d) 공식S3 무개선 또는 새 강한반전 사건 증가. 이 경우 모델크기·보정값을 계속 탐색하지 말고 미채택.

수량제약: 기존 검증 가속2창은 사전요구3개보다 적다. 28창은 가설 타당성의 개발검사에만 쓸 수 있다. 같은속도·차량·분할조건을 만족하는 추가 창이 기존23영상에 없으면 기존 기준을 완화해 통과시켜서는 안 됨. 독립경로 추가수집은 별도 필요사항으로 보고한다. 기하검사 자체가 목적이 되지 않도록 위 단일 pilot에 한정하고, 결과가 약하면 점수실험으로 확대하지 않는다.

## 최신 공식 근거와 제한

- https://github.com/ByteDance-Seed/Depth-Anything-3 : Main model의 joint relative depth/camera pose, Small0.08B Apache2.0, MetricLarge와의 구분 확인. 현재 README는 Large/Giant/Nested 1.1에서 street scenes 개선을 주장하지만 Small에 그 개선이 적용됐다고 쓰지 않는다.
- https://github.com/ByteDance-Seed/Depth-Anything-3/blob/main/docs/API.md : world-to-camera extrinsics, intrinsics, ray-pose option 확인.
- https://github.com/ByteDance-Seed/Depth-Anything-3/blob/main/docs/BENCHMARK.md : geometry/pose benchmark가 Stage3 가감속 MacroF1 검증은 아니다.
- https://github.com/facebookresearch/vjepa2/blob/main/evals/video_classification_frozen/models.py : encoder의 token representation 인터페이스. 로컬 spatial pooling 실패를 전체모델 무용성으로 확대 불가.

위 DA3 권고는 기하출력을 Stage3 요구에 연결한 **새 실험 설계**이며 검증된 성능개선 방법이라는 사실주장이 아니다.

## 다른 전문가와의 교차 검토 후 실행 순서

평가 전문가의 **현재2395/DIS864에서 얕은HGB 단일 대조 우선** 제안에 비용·직접 점수판정 기준으로 동의한다. 과거 RF/ExtraTrees는 old2761/20Hz 조건으로 같은 대조가 아니며, 최근 MLP는1083조건이라 현설정 HGB가 이미 반증됐다고 단정할 수 없다. 다만 기존 tree 열세와 MLP trainF1≈1/일반화실패는 모델용량 부족 설명에 반증이다. HGB를 '확인된 원인을 해결하는 최적모델'이라고 소개하면 안 된다.

따라서 종합 실행순서는 (1)현재 특징·정답·학습표본·분할 고정, 얕은 HGB한개 설정만 선형대조 → (2)실패시 FOE반경특징으로 회귀하지 않고 위 DA3공동pose pilot 한 번 조건부 검토. HGB에서는 classbalancedweight·학습량 정의가 baseline과 일치하도록 하고 추가feature, smoothing, score threshold, seed/하이퍼파라미터 탐색을 섞지 않는다. 단일 대조가 실패하면 여러 tree/MLP 이름 순환을 하지 않는다.
