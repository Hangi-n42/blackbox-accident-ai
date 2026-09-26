# 독립 harder-pair 집계 검증: PASS

학습4건의 고정14특징·PCA·정규화·P/M/53쌍과 비가중치 배열 보존을 확인했다. 새 log-mean-exp 집계는 np.logaddexp.reduce로 별도 계산했고, 실제 runner objective의 baseline/new 가중치 중앙차분(ε=1e-6)으로 gradient를 확인했다. 추가 optimizer 호출은0이다.

|사례|기존→새 프레임|새 ±0.3 판정|
|---|---:|---|
|00000|0→0|wrong_all_reference_times|
|00003|0→0|correct_all_reference_times|
|00006|0→0|correct_all_reference_times|
|00013|545→545|wrong_all_reference_times|

같은 T 기준 평균 절대오차 변화 [0.0, 0.0]초, 정확도 변화 [0.0, 0.0]. 선택 프레임 변화와 시간 정확도 개선은 없다.

동일 가중치에서 옛/새 집계의 pair gradient 계수와 비중을 전수 대조했다. 새 집계는 어려운 쌍의 상대 비중뿐 아니라 사고별 계수 총합도 바꾼다. 초기 zero 가중치에서 두 식의 손실·gradient는 같으며, 이후의 계수 분포는 달라진다. 이는 식과 결과의 산술 검증이며 일반화 효과를 입증하지 않는다.

동결 25파일·기존 모델 보존. 추가 학습·PCA·encoder·CCD/제외특징 접근0.
