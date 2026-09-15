# V4 Stage2 A/B 사전 설계

2026-09-13. 이 문서는 새 A/B 추론 결과를 보기 전에 작성했다. 기존 공개 5개와 과거 결과는 이미 검토했으므로 새로운 독립 검증셋이라는 주장은 하지 않는다.

- 기준: frozen V3 motion-collision, NF4 4B, 기존 영상당 4회 질문.
- A: 기존 overview/entry/space 질문을 유지한다. collision fine 질문을 제거하고 최종 motion argmax 시점을 entry 후보의 끝과 space 문맥 중심으로 사용한다. 정확히 3회. crop, reference, 모델, 픽셀 예산, motion 식, 시간 offset은 바꾸지 않는다.
- B: A 결과를 얻은 뒤 entry fine 질문을 최대 1회 추가한다. 기존 stage2_entry_refine._refinement_indices를 재사용한다. 전체 3~4회. 이미 precontact 모든 프레임을 보여줬거나 새 후보가 없으면 생략한다.
- B guard: coarse entry가 유효한 제시 정수 번호가 아니면 refine 생략. fine 응답은 정상 JSON 객체, status가 OBSERVED, entry_frame이 제시된 정수 원본 번호일 때만 수용. BEFORE_WINDOW / AFTER_WINDOW / UNCERTAIN / 잘못된 출력 / 예외는 A entry를 보존한다. 국소창 첫 이미지가 영상 시작은 아닐 수 있음을 질문에 명시한다. 첫 이미지도 실제 접촉이 관찰되었다는 OBSERVED일 때만 수용한다.
- B는 틀린 coarse 구간 바깥의 사건을 탐색하지 않는다. 모델이 잘못 OBSERVED라고 답하는 오류도 guard가 해결하지 못한다.
- 모든 프레임 번호는 읽을 수 있는 원본 파일명 번호다. 목록 위치, 파일명 번호, 시간은 별개이며 숨겨진 FPS를 가정하지 않는다. 각 영상은 독립이다.
- 먼저 CPU mocked 계약 검사: 네/세 호출 상한, 순서/번호/offset, A/B 다른 필드 보존, pointwise independence, skip, 잘못된 JSON, before/after/uncertain fallback, 국소 시작 구분, 빈 입력/중복/불일치 검증.
- 이후 같은 로더 인스턴스로 baselineV3 -> A -> B를 순차 실행한다. 각 variant의 모든 5개를 처리한다. 매 호출 prompt, 입력 이미지/크기/해시, bounded 이미지 크기, 원 응답, 후보, 시간, GPU 메모리를 기록한다. 새 영상간 문맥 또는 통계를 전달하지 않는다.
- 공개 충돌 GT는 예측 완료 후 점수화한다. 원본 MP4의 PTS와 이미지 원본 번호를 대응시켜 검증하며 새 정책에는 GT/PTS를 주입하지 않는다. collision은 모든 arm에서 motion argmax이므로 성능 유지가 계약이며 새 충돌 향상을 주장하지 않는다.
- entry/side/space 공식 GT가 없으므로 accuracy를 주장하지 않는다. root가 별도로 만든 공개 시각 검토 라벨과 대조하더라도 AI 주석/수작업 개발진단으로 표시하고 공식 GT와 혼합하지 않는다. 비공개 평가/공식 점수로 A/B를 고르지 않는다.
- 채택 필수조건: 계약 통과, 기존 source/weights 보존, collision/side 보존, 실패 없이 범위 내 출력, 공개 개발주석상 변화를 root가 검토. B는 A보다 진입 개발진단이 낫다는 근거가 없으면 자동 채택하지 않는다. 후보 coverage 개선만으로 실제 정확도 개선을 주장하지 않는다. 전체 패키지 선택은 root 담당.
- A가 다른 필드에 영향을 주는 것은 의도한 문맥 변경이며 동등 최적화가 아니다. 시간 측정은 로컬 공개 5개에 한정하고 60분 통과를 보장하지 않는다.
