# 학습형 진입 선택기 데이터 감사

**결론: 작은 개발 파일럿은 가능하지만, 확정 사람 진입 정답과 독립 사람 평가 집합은 이번 확인 범위에서 0건이다.** Nexar 단일 사람 초안9건 중 진입 시각은8건, 미상은00004다. 원본 JSON과 baseline 사본의 해시·내용이 모두 일치한다. 전부 human_review_draft/evaluation_eligible=false이며 이후 여러 모델 비교에 노출됐다. 원래 unseen 문자열은 현재 독립성을 뜻하지 않는다. 사고 상대 설명은 있으나 두 사람의 독립 검수·조정 자료는 없다.

## 실제 원Q3 학습 가능 범위

| Nexar | 초안 진입 | 현재12후보 중 ±0.3초 후보 | 최소 후보 오차(초) |
|---|---:|---|---:|
| 00000 | 555 | [561] | 0.207612 |
| 00003 | 0 | [0] | 0.000000 |
| 00004 | None | [] | 미상 |
| 00005 | 643 | [] | 1.568627 |
| 00006 | 0 | [0] | 0.000000 |
| 00007 | 514 | [515] | 0.033333 |
| 00008 | 560 | [] | 0.433333 |
| 00010 | 572 | [] | 7.833333 |
| 00013 | 569 | [] | 0.800000 |

**원Q3는4/8 커버리지다.** 과거1/8은 temporal_v1의 마지막 세분화 후보 결과이며 혼동하면 안 된다. 후보가 없는4건도 오차 순위는 학습할 수 있지만, 기존12후보만 고르는 선택기로 후보 누락 자체를 복구하지는 못한다. 이 표는 원 사람초안 기준이며, 새 시각 검수 B에서 미상/구간으로 변경하면 실제 학습 대상과 쌍 수를 다시 고정해야 한다.

공개 S2_001의 별도 사용자 초안에는 진입2.4초가 있고 원Q3의25번이 허용오차 안이다. 그러나 공식 공개 CSV는5건 모두 t_entry=-1이다. **root 결정에 따라 이 추가 초안은 학습에 넣지 않고 공개5건은 기존 접촉 회귀용으로 유지한다.**

## 두 학습안과 선택된 범위

| 방식 | 가능한 감독 | 한계·결정 |
|---|---|---|
| 정답 후보 분류 | Nexar4건에 제시된 양성 후보 | 4영상뿐이며 missing-positive의 상대순위 정보를 버림. 선택하지 않음 |
| 공유 선형 pairwise ranker | Nexar known8 중 B 시각 검수 통과분의 확실한 오차 우위 쌍 | root 선택. 최종 적격 영상/구간/쌍 수는 검수 완료 후 확정 |

구간[L,U]에서 두 후보가 **같은 미상 시각**을 공유하게 한다. 두 끝점 모두에서 i의 절대오차가 j보다 확실히 작을 때만 i 우위를 학습한다. 일차원 절대거리 차이는 단조이므로 끝점 비교로 충분하다. 동률·순위 불확실 쌍은 제외하고, 영상별 pair loss 평균 후 영상 간 평균을 낸다. 최대66쌍/영상은 독립사고66건이 아니다. 정답 후보가 없는 영상의 최선 후보를 실제 정답으로 바꾸지 않는다.

검수에서 같은 상대/진입이 미상이면 학습에서 제외한다. 사람 초안을 AI가 재검수해도 human_adjudicated가 되지 않는다. 학습 자료만으로 특징 통계·설정·학습 종료를 고정하고 **CCD8은 gradient·정규화·하이퍼파라미터·epoch 선택에 넣지 않는다.** 원12후보와 RGB/PTS, 나머지3출력은 보존한다. 작은 고정 fit1회와 CCD8 개발평가1회가 최소 범위이며, 성공을 보장하는 표본 규모가 아니다.

## 평가와 나머지 자료

- CCD는 추가 취득24영상/24YouTube그룹/1,200PNG와 공개5영상을 실제 보유한다. 새24개 모두 이전 기준 추론·AI 검수에 노출됐다. 현재 진입 참조는8건(시작전4·영상중4), 나머지16은 미상이며 다시 억지 확정하지 않는다.
- CCD8 중 구간 전체에 맞는 단일 후보가 있는 것은6건이다.1237([8,22])·540([31,38])은 넓은 구간이라 단일 robust 후보가 없지만 가능한 후보집합/구간 전역 커버리지가 있다. 둘을 후보 누락으로 오분류하거나 정답 중점을 강제하지 않는다. 평가 분모8을 유지하며 정답/오답/불확정과 공유시각 paired 범위를 보고한다.
- 두 CCD코호트는 공개2그룹 및 서로의 그룹과 분리됐고, Nexar64·공개·DADA·MM-AU와 기존 제한된 해시/표본 지각해시/시각 대조에서 중복은 관측되지 않았다. 미탐 재편집 사고·사전학습 중복 부재의 인증은 아니다. Nexar 사람초안의 source_group=PENDING은 그대로 남긴다.
- Nexar40은 접촉 직접확정0·진입 근사구간4이며 전부 개발 노출이다. 근사구간은 보장된 포함구간이 아니다.14~16은 진입미상3,17~23선정6개도 진입미상6,24~29는 접촉/진입미상6이다. 빈 human_review 템플릿을 완료된 정답으로 세지 않는다.
- DADA16 PNG시퀀스는 과거 AI 구간 검수가 있지만 native PTS/원MP4가 없고, MM-AU223 JPEG는 시간/상대 대응이 미해결이다. CADP/DoTA는 이전 로컬 감사에서 영상이 발견되지 않았다. 현재 원Q3 지도학습용 정확 진입 풀을 즉시 늘려 주지 않는다.

**Nexar 학습 → CCD 평가의 데이터셋/그룹 분리는 가능하지만, CCD8은 이미 노출된 개발 평가다.** 학습 손실 감소는 실제 시점 정확도 개선이 아니며, 이 결과를 독립 사람 검증·공식S2·CUDA 제출 성능으로 부르지 않는다.

## 추가 CCD의 현실적 경로

전체 저장소 MP4 경로147개를 조사했고 CCD 명명 원본24개와 공개 numeric 원본을 확인했다. 알려진24개 외 추가 비공개용 CCD 원본은 발견하지 못했다. 이는 모든 일반 파일명 영상의 내용 분류가 아니다.

저장된 공식 주석1,500행 중 ego=Yes801개·132그룹이다. 공개2+기존24그룹을 제외하면 **611클립·106그룹**이 메타데이터에 남는다. 전체 CCD_ID의 SHA256 순으로 서로 다른 다음12그룹을 고르는 메타 예시는 압축합8,258,191바이트다. ID·ZIP멤버·주석행을 data_audit.json에 보존했지만 **아직 선택 동결·취득·시각 검수한 데이터가 아니다.** 기존 Range 취득 실적만 확인했고 현재 원격 접근은 다시 검사하지 않았다.

새 검증이 필요하면 원천 그룹을 취득/검수/모델출력 전에 분리하고, 기존 사고와의 제한된 중복 검사·예측 비노출 주석을 거쳐 적격 항목만 동결한다. 데이터 라이선스 범위 내 CCD 사용 허용은 통합 문서222행에 기록돼 있다. 공급자의 사고 binary 라벨은 진입 정답이 아니다. 관측 불가하면 unknown을 유지하고 결과를 본 뒤 사례를 교체하지 않는다.

현재 감사에서 다운로드·추론·특징 추출·학습·기존 자료 수정은 모두0이다. 새 검증 부재는 작은 개발 fit 자체를 막는 조건으로 사용하지 않되, 실제 성능 검증 완료와 구분한다.

## 근거

- human_labels: [artifacts/mac_experiments/baseline_20260916/human_labels.json](/Users/hyeongi/projects/blackbox-accident-ai/artifacts/mac_experiments/baseline_20260916/human_labels.json:1)
- human_review_contract: [research/v6_review_tool/README.md](/Users/hyeongi/projects/blackbox-accident-ai/research/v6_review_tool/README.md:25)
- GT_schema: [research/v6_stage2/SCHEMA.md](/Users/hyeongi/projects/blackbox-accident-ai/research/v6_stage2/SCHEMA.md:13)
- current_human_input: [artifacts/stage2_goal_20260919/current_baseline/inputs.json](/Users/hyeongi/projects/blackbox-accident-ai/artifacts/stage2_goal_20260919/current_baseline/inputs.json:1)
- historical_coverage_distinction: [docs/mac-priority01-results-20260916.md](/Users/hyeongi/projects/blackbox-accident-ai/docs/mac-priority01-results-20260916.md:44)
- public_official_partial: [Baseline/data/stage2/labels.csv](/Users/hyeongi/projects/blackbox-accident-ai/Baseline/data/stage2/labels.csv:1)
- public_human_draft: [research/v6_stage2/user_reviews/PUBLIC_DEV_001_review_1789397594082.json](/Users/hyeongi/projects/blackbox-accident-ai/research/v6_stage2/user_reviews/PUBLIC_DEV_001_review_1789397594082.json:26)
- current_CCD_refs: [artifacts/stage2_entry_path_audit_20260920/references.json](/Users/hyeongi/projects/blackbox-accident-ai/artifacts/stage2_entry_path_audit_20260920/references.json:1)
- current_CCD_decomposition: [artifacts/stage2_entry_path_audit_20260920/decomposition.json](/Users/hyeongi/projects/blackbox-accident-ai/artifacts/stage2_entry_path_audit_20260920/decomposition.json:1)
- old_CCD_input: [artifacts/stage2_goal_20260920/ccd_intake/inputs.json](/Users/hyeongi/projects/blackbox-accident-ai/artifacts/stage2_goal_20260920/ccd_intake/inputs.json:1)
- new_CCD_input: [artifacts/stage2_first_frame_policy_20260920/intake/inputs.json](/Users/hyeongi/projects/blackbox-accident-ai/artifacts/stage2_first_frame_policy_20260920/intake/inputs.json:1)
- CCD_metadata: [artifacts/stage2_goal_20260920/ccd_intake/annotation.txt](/Users/hyeongi/projects/blackbox-accident-ai/artifacts/stage2_goal_20260920/ccd_intake/annotation.txt:1)
- CCD_zip_metadata: [artifacts/stage2_goal_20260920/ccd_intake/zip_members.json](/Users/hyeongi/projects/blackbox-accident-ai/artifacts/stage2_goal_20260920/ccd_intake/zip_members.json:1)
- CCD_bounded_overlap: [artifacts/stage2_first_frame_policy_20260920/intake/overlap.md](/Users/hyeongi/projects/blackbox-accident-ai/artifacts/stage2_first_frame_policy_20260920/intake/overlap.md:3)
- Nexar40_curation: [artifacts/data_curation_20260917/REPORT.md](/Users/hyeongi/projects/blackbox-accident-ai/artifacts/data_curation_20260917/REPORT.md:19)
- Nexar6_entry_unknown: [research/v7/new_source_review_result_20260915.md](/Users/hyeongi/projects/blackbox-accident-ai/research/v7/new_source_review_result_20260915.md:16)
- Nexar3_entry_unknown: [research/v7/ai_review_stage2/00014/review.json](/Users/hyeongi/projects/blackbox-accident-ai/research/v7/ai_review_stage2/00014/review.json:1)
- Nexar_later6_screening: [artifacts/stage2_goal_20260919/acquisition/ai_screening.json](/Users/hyeongi/projects/blackbox-accident-ai/artifacts/stage2_goal_20260919/acquisition/ai_screening.json:1)
- Nexar_later6_blank_human_forms: [artifacts/stage2_goal_20260919/acquisition/review_packet.md](/Users/hyeongi/projects/blackbox-accident-ai/artifacts/stage2_goal_20260919/acquisition/review_packet.md:3)
- other_local_sources: [artifacts/stage2_goal_20260920/data_source_audit.md](/Users/hyeongi/projects/blackbox-accident-ai/artifacts/stage2_goal_20260920/data_source_audit.md:9)
- CCD_recorded_use: [대회_통합_정보.md](/Users/hyeongi/projects/blackbox-accident-ai/대회_통합_정보.md:222)
- pairwise_design: [artifacts/stage2_entry_selector_20260920/design_review.json](/Users/hyeongi/projects/blackbox-accident-ai/artifacts/stage2_entry_selector_20260920/design_review.json:1)
