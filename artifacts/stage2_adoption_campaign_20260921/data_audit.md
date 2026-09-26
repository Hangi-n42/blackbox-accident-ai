# Stage2 캠페인 자료 감사

새 사람 정답은 확인되지 않았다. 검색한 Nexar 사람 초안은 10개 파일/9사례이고 00003 수정본 중복을 포함한다. 최신 작성시각은 2026-09-15T08:32:49.023Z이다. 모두 기존 검수·평가에 노출됐으며 원기록의 unseen은 과거 시점 표기다. 공식 정답이나 독립확정 사람GT로 승격하지 않는다.

## 참조 적격성

- 00014/15/16 AI검수3건과 00017/18/19/21/22/23 조정6건은 진입 참조 적격0건. 접촉추론구간3개는 진입정답이 아니다.
- Nexar40 최신20260917 참조는 exact entry0건·독립평가0건이다. 일부 오래된 entry0 기록보다 최신 physical left-censoring/조건부 주석을 우선한다.
- 다만 동일 상대의 시작부터 내부 관찰은 대회 before-start→첫프레임 규칙에 따른 약한 AI참조 후보가 될 수 있다. 물리적 최초진입 시점과 제출규칙을 분리해 새 참조를 예측열람 전에 고정해야 한다.

## 최대12건 중 우선9건 제안

|ID|현재 AI 진입 근거|우선도|baseline cache|
|---|---|---:|---:|
|00118|left_censored_before_clip|1|1|
|00222|left_censored_before_clip|1|1|
|00554|left_censored_before_clip|1|1|
|00362|left_censored_before_clip|1|1|
|00630|left_censored_before_clip|1|1|
|00537|uncertain|1|1|
|00199|uncertain_candidate_geometry|2|0|
|00863|uncertain_candidate_geometry|2|0|
|00658|uncertain_candidate_interval|2|0|

우선6건: 00118·00222·00554·00362·00630은 같은 상대가 시작부터 내부라는 관찰, 00537은 진입 가능구간이 있으나 미확정이다. 추가3건00199·00863·00658은 실제 접촉/상대 또는 경계가 불확실하여 현재 평가부적격이다. 새 영상검수는 이 감사에서 수행하지 않았으며 예측 내용을 열지 않았다.

## 원천·노출·재사용

로컬 Nexar 64원본의 실제 SHA를 검사했고 64개가 기록과 일치했다. ID가 다른 byte중복 그룹은 []이다. 이는 사고 독립성 인증이 아니다. 원천 drive/incident ID가 없고 재편집·재업로드 중복은 배제하지 못한다.

기존 baseline calls와 원프레임 manifest 경로는 우선6건에 존재한다. calls 내용은 읽지 않았으므로 현재 Q3 12후보·RGB·processor·코드와 일치한다고 아직 주장할 수 없다. 참조/적격성을 잠근 뒤 기존캐시 CPU 재생 또는 동일입력 대조가 필요하다.

## 평가 분모

새 공식/확정 사람 진입평가 적격분모는0이다. 약한 추가참조 후보6건도 아직 적격확정분모가 아니다. 검수한 모든 사례/unknown을 남기고, 실제 적격사고 수와 before-start/during-clip을 각각 보고한다. 기존 노출자료를 새로운 독립평가라고 부르지 않는다.

근거: [사람 초안](/Users/hyeongi/projects/blackbox-accident-ai/research/v6_stage2/user_reviews/), [round2 초안](/Users/hyeongi/projects/blackbox-accident-ai/research/v6_stage2/round2_validation/intake_20260915_actual/user_reviews/), [AI3](/Users/hyeongi/projects/blackbox-accident-ai/research/v7/ai_review_stage2/review_summary.json), [AI6조정](/Users/hyeongi/projects/blackbox-accident-ai/research/v7/next_review_adjudication.json), [최신40참조](/Users/hyeongi/projects/blackbox-accident-ai/artifacts/data_curation_20260917/nexar/annotations.json), [노출기록](/Users/hyeongi/projects/blackbox-accident-ai/artifacts/data_curation_20260917/nexar/split_manifest.json). 모든 원본 SHA·정확 경로·cache metadata는 data_audit.json에 보존했다.
