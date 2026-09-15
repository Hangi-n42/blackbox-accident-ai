# 외부 검수 자료 확보 기록

2026-09-14. 새 사람 검수 자료를 사용자에게 요청한 뒤, 공개 자료로 해당 의존성을 줄일 수 있는지 확인했다. 모델이나 AI 관찰로 정답을 생성하지 않았다.

## Nexar — 검수 후보 6개 확보

- [공식 라이선스](https://huggingface.co/datasets/nexar-ai/nexar_collision_prediction/blob/main/LICENSE)는 출처 표기와 조건 유지를 전제로 사용·복사·수정·배포를 허용한다. 재판매, 재식별 및 열거된 부적절한 용도는 제한한다. 원문을 자료 폴더와 검수 화면에 보존했다.
- [공식 데이터 설명](https://huggingface.co/datasets/nexar-ai/nexar_collision_prediction/blob/main/README.md)의 positive는 실제 충돌과 near-miss를 함께 포함한다. 따라서 time_of_event를 최초 접촉 정답으로 자동 변환하지 않는다. 사건 전에 끝나는 test 영상 대신 공개 train/positive만 취득했다.
- 고정 revision: aa97deda5a59f00bb7187739053b7c72e14374df. 예측이나 시각 정답을 보기 전에 파일명 순서의 첫 6개를 선택해 plan.json에 기록했다. 이는 도구·검수 절차를 시작하기 위한 편의 표본이며 대표성 있는 검증 집합이 아니다.
- 00000, 00003, 00004, 00005, 00006, 00007 MP4의 공개 LFS SHA-256과 크기를 대조했다. 실제 전체 프레임을 디코딩해 각 원본 PTS를 보존했다. 모든 프레임을 30fps라고 가정하지 않았다.
- 새 모델 추론·사람 검수·원천 중복 감사는 아직 없다. source_group_id와 독립성은 미확정이며 확정 GT 0개다. 검수 화면의 PENDING_SOURCE_AUDIT 식별자는 임시 이름이지 독립 원천 인증이 아니다.
- 권리자 표기: Moura, Daniel C., and Zvitia, Orly. Nexar Collision Dataset. Hugging Face, 2025. Nexar Inc.

실제 증거: nexar_review_candidates/plan.json, acquisition.json, 각 mapping.json 및 원본 MP4. acquisition의 human_review=pending/contact_ground_truth=null을 유지한다.

## MM-AU — 별도 가능성 조사 중

[공식 GitHub](https://github.com/jeffreychou777/LOTVS-MM-AU)는 t_co를 충돌 시작 프레임으로 명시한다. GitHub의 academic use 문구와 별도로, 공식 연결 [Hugging Face README](https://huggingface.co/datasets/JeffreyChou/MM-AU/blob/main/README.md)에는 CC-BY-NC-4.0가 명시돼 있다. 따라서 앞서 사용 허가가 확인되지 않았다는 이유만으로 영구 배제하지 않는다.

추가 조사에서 공식 사람 충돌 시작 주석과 CAP 한 원본의 이미지 223장을 약 77MB 전송으로 실제 확보했다. 다만 [공식 관리자 답변](https://huggingface.co/datasets/JeffreyChou/MM-AU/discussions/1)은 원본 FPS가 영상마다 다르며 현재 원본 영상/FPS를 제공할 수 없다고 명시한다. 따라서 주석 프레임을 ±0.3초 평가로 환산할 근거는 확보되지 않았다. 프레임 단위 후보 감사 가능성은 남겨두되 독립 시간 검증 집합으로 채택하지 않았다. 자세한 증거와 경로는 ../v6_stage2/mmau_feasibility.md에 있다.

## PSAD — 아직 미채택

[공식 저장소](https://github.com/Shun-Gan/PSAD-dataset)는 DoTA에서 선별한 원본과 collision timestamp/객체 주석을 제공한다고 설명한다. 검토한 페이지에서 대회 활용을 판단할 라이선스 근거를 확인하지 못했다. 원본 프레임과 10fps 객체 주석 번호의 변환도 별도로 필요하다. 현재 내려받거나 정답으로 채택하지 않았다.
