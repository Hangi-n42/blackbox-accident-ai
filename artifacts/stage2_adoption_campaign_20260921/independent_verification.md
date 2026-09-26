# 독립 실행·평가 검증: PASS

동결 68개 해시 전후 보존. baseline 48개 저장 응답의 동일 코드·질문·RGB·출력 재생, fresh Q3 24질의와 CPU processor 24회 텐서 해시 대조를 완료했다. 새 모델 호출·모델 가중치 로드0.

|분모|건수|기존 정확도 하한/상한|후보 정확도 하한/상한|같은 정답시각 MAE 변화(초)|
|---|---:|---|---|---|
|strict|1|0.0/0.0|0.0/0.0|[3.1, 3.1]|
|conditional|2|0.0/0.0|0.0/0.0|[0.65, 0.65]|

실제 72 model calls, 전체 36 workers exit0. 동일 control 12/12, 다른 세 출력 12/12 보존. 형식상 미유효 응답 baseline 0, candidate 0.

Dual AI references only; strict and conditional have separate fixed denominators, and nine cases remain unscored.
All eligible references concern before-start encoding. No eligible during-clip case tests false-first-frame risk.
Distinct provider compilation groups plus sampled duplicate screening do not certify original-incident independence.
Packaging changes visual boundaries/positions and text layout together; this is not a pure semantic or pixel-count causal test.
Other three outputs are copied from baseline, not newly inferred. Mac MLX validation is not submitted CUDA/NF4 or official total Stage2 performance.
