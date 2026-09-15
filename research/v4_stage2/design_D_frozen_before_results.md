# 최종 D 대조 사전 설계 — V3 기반 origin-side 문장만 변경

2026-09-13. C가 종료된 후 새로운 D 결과를 보기 전에 동결한다. C는 다섯 사례 모두 BEFORE_WINDOW라 최종값이 A와 같았고, A는 이미 차로 안이라는 S2_004 시각 관찰과 더 어긋나는 진입 15를 출력했다. 따라서 C 기반 D 계획은 구현/실행하기 전에 폐기하고, 기존 V3를 직접 대조하는 단일 D로 설계를 확정한다. 폐기한 C 기반 D 코드는 만들지 않았다.

- 기준 `solution/stage2_motion_collision.py`의 네 호출·원본 번호·motion argmax 충돌·입력 정책을 유지한다.
- 매 파일 새 wrapper를 만든다. 첫 ask만 다음 원문을 치환한다. 다른 질문, JSON 키, token budget, 이미지 객체는 그대로 전달한다.

원문: `From which side of the image did that other vehicle approach? `

치환문: `From which image side did that same vehicle originate before entering the camera car's driving corridor? Track the collision vehicle backwards. Report its image side of origin before lane entry, not its direction of travel, side of impact, or position after entry. `

- 첫 질문의 collision_frame 요청과 entry_side LEFT/RIGHT 응답 키를 유지한다. 원문이 정확히 한 번 없는 경우 예외를 내어 의도치 않은 다른 첫 질문에 적용하지 않는다.
- 나머지 세 질문은 frozen V3 정책으로 생성한다. 첫 collision 응답이 달라지면 fine 후보/entry 상한/space 문맥이 달라질 수 있다. 그러므로 마지막 side만 바뀐다는 동등성은 가정하지 않는다. 실제 질문·후보·이미지·응답 차이를 모두 기록한다.
- 최종 collision은 변함없이 motion argmax다. 원영상 시간/FPS를 가정하지 않으며 ID별 규칙/데이터셋 통계/문맥 재사용/재학습이 없다.
- CPU mocked 계약: 첫 문장만 변경·나머지 prompt 불변 조건, 원본번호/파일별 초기화/4회 유지, prompt 원문 guard, 최종 motion 충돌 보존 검증.
- 같은 저장 NF4 모델로 공개 5개를 실제 순차 실행한다. source/weights/이전 A/B/C 결과를 보존한다. baseline V3 기록과 각 호출을 직접 비교해 후보 변화와 출력 변화의 경로를 기록한다.
- 제공 collision GT는 그대로 사용한다. 예측을 보기 전에 기록된 AI 시각 관찰에서 S2_001/003 RIGHT는 합의됐지만 공식 GT가 아니다. 이미 앞차가 같은 차로에 있는 사례의 시작 방향은 불명확하므로 정답을 강제하지 않는다.
- LEFT가 기존 모델 원응답인지 parser fallback인지 구분한다. D 개선이 있더라도 이 문구가 일반적 오류 원인이라고 단정하지 않는다. 동일 공개 사례를 보고 가설을 설계했으므로 독립 검증이 아니다.
- D가 마지막 후보다. 결과를 보고 D 문구를 재조정하거나 E를 탐색하지 않는다. 패키지 선택·공식 제출은 root 담당이다.
