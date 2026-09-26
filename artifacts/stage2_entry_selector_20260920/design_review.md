# 최소 학습 선택기 설계 검토 A

**작은 고정 pairwise ranking 파일럿은 조건부로 진행 가능하다.** 기존 Q3 12후보·원본번호 mapping·다른3출력은 유지한다. vision backbone을 학습하거나 현 CCD8건에 맞춰 설정을 고르는 근거는 없다.

사람 초안은 [human_labels.json](/Users/hyeongi/projects/blackbox-accident-ai/artifacts/mac_experiments/baseline_20260916/human_labels.json)에서 확인했다. 9건 모두 human_review_draft이며 evaluation_eligible=false, source group 감사 대기다. 00004의 진입은 미상이고 나머지8건은 점 시각이다. 00003/00006은 시작부터 같은 차로다. 학습용 약한 감독으로 쓸 수 있는지는 별도 데이터 감사로 확정하며 원본 상태를 공식 정답으로 승격시키지 않는다. 과거 저장된 unseen 표기도 이후 개발 노출을 지우지 않는다.

원래 Q3의 human-dev 진입 후보 커버는 [분해 기록](/Users/hyeongi/projects/blackbox-accident-ai/artifacts/stage2_goal_20260919/core_v2/decomposition.json)의 A0=8, A1=7, A2=4다. **4개 covered 영상만 정답 softmax로 학습하는 것보다, 8개 적격 초안에서 확실한 시각오차 우위 쌍을 학습하는 편을 권고한다.** 후보에 정답이 없어도 두 후보 중 어느 쪽의 시간오차가 작은지는 알 수 있다. 그 가장 가까운 후보를 실제 진입 정답으로 부르지는 않는다. 누락된 정답의 복원이나 ±0.3초 정확도 개선은 별도 문제다.

최소 구현은 frozen Qwen3-VL vision features의 원래 시트 타일별 pooling, 고정 저차원 투영, 공유 선형 score head 하나다. 예시 고정값은 embedding L2 정규화→seed0 random projection8차원→현재8+이전후보와의차8+PTS상대위치1+첫후보1의18차원이다. 숨은 층·LoRA·backbone 학습 없이 영상별 pair loss 평균을 다시 영상별 평균하고 L2를 적용한다. 예시 최적화는 w=0, L2계수1, full-batch L-BFGS max200/tolerance1e-6이며 최적이라는 증거가 아니라 사전고정 공학적 선택이다. Root 구현의 다른 고정값도 첫 fit/CCD결과 전에 명시해야 한다.

원래 RGB시트의 장면 영역과 processor token/patch-merge 좌표 대응을 먼저 검증한다. 헤더를 pooling에서 제외해도 전체시트 attention이 섞인다면 순수한 타일 독립 특징이라고 부를 수 없다. 전체장면 특징은 특정 사고상대 특징도 아니다. 타일/프레임 매핑이 불명확하면 fit 전에 중단한다. 정규화·PCA를 쓴다면 통계는 Nexar 학습에서만 맞춘다. 고정 random projection은 별도 통계 fit을 줄이는 선택이다.

참조가 [L,U]이고 후보시각이 t_i,t_j이면 d_i(t)=|t_i−t|로 두고, 양끝점에서 계산한 d_i−d_j의 최댓값이 −epsilon보다 작을 때만 i를 우위로 학습한다. 일차원 절대거리 차이는 단조여서 이 조건이 전구간 우위를 확인한다. epsilon은 수치 동률 배제용이지 평가 허용오차가 아니다. 순위가 바뀌는 쌍·동률·동일시각·미상은 버린다. 점 초안은 L=U로 쓰되 독립 검수에서 넓힌 구간이 있다면 fit 전에 그대로 고정한다. loss는 log(1+exp(−(s_i−s_j)))다. 영상당 쌍을 먼저 평균해 쌍이 많은 영상이 우세하지 않게 한다. 8영상×다수 쌍은 독립 표본수를 늘리지 않는다.

before_start는 물리적 시각이 아니라 제출의 첫 원본프레임 목표다. 정답이 후보범위 뒤에 있는 영상은 거의 모든 쌍이 뒤 후보를 선호할 수 있어 위치 shortcut 위험이 크다. 원래 Q3와 함께 항상 첫 후보, Nexar에서만 정한 목표 상대위치 중앙값 선택을 미리 고정해 비교한다. 이 비교를 보고 CCD에 맞는 모델/λ/seed/epoch를 다시 고르지 않는다.

Nexar 적격 자료만 fit하고, 사고 중복을 제거한 CCD AI참조8건은 단 한 번의 제한된 개발 평가로 남긴다. 데이터셋 이름이 다르다는 이유로 독립을 가정하지 않는다. 원영상·재업로드·인접클립 그룹을 분리하고 미상16건에는 라벨을 만들지 않는다. CCD는 과거 노출된 AI참조라 사람의 독립 holdout이나 공식S2가 아니다. 학습 전 입력·분할·참조·모델·특징·설정·평가규칙을 동결한다.

개발 신호 기준은 사전고정 참조에서 확정 개선≥1, 확정/가능 손실0, 새 잘못된 f0선택0, paired MAE 변화상한≤0다. 후보 oracle오차와 실제 선택오차, 구간 정확도와 ranking loss를 분리한다. 현재8건에서 이를 통과해도 제출 교체 근거는 부족하다. 별도 사람검수 사고 그룹과 실제 CUDA 제출경로·다른3출력 보존 및 비용 검증이 필요하다.

학습 적격성/사고중복/타일·PTS mapping 검증 실패, 유효 preference 부재, nonfinite·수렴실패·출력계약 위반이면 중단한다. 단지 후보에 정확한 정답이 없다는 이유로 시간오차 순위가 확실한 영상을 버릴 필요는 없다. 자료가 부족하면 배선확인 파일럿이라고 보고하고 대규모 미세조정이나 미상 라벨 확정을 추가하지 않는다.

이 역할은 설계 검토만 수행했다. 코드·데이터 원본 변경과 모델 호출은0회다. 세부 계약은 design_review.json에 기록했다.

