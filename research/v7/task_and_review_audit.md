# V7 과제·사람 검수 적용성 감사

확인일: 2026-09-15. 기존 모델·라벨·동결 실험을 변경하지 않은 감사다. `review_applicability_evidence.json`에 최신 사람 검수 9개와 원본·case·선택 PNG의 대조 근거를 보존했다. 사람 검수와 아래 AI 관찰은 별개이며 공식 GT를 생성하지 않았다.

## 1. 공식 규격과 실제 판독 절차

공식 평가·규칙·필독 공지와 아래 운영진 답변을 재열람했다. 검색 가능한 최신 페이지 범위이며, 모든 게시물의 미검색 댓글까지 확인했다는 뜻은 아니다.

| 항목 | 확인한 규정 | 이를 구현·검수할 때 필요한 절차와 불확실성 |
|---|---|---|
| 대상·접촉 | 블랙박스 설치 차량과 그 상대의 실제 물리적 접촉이다. 근접·위험·충격처럼 보이는 흔들림과 구분한다. | 충돌 상대를 먼저 식별하고 접촉 전후를 같은 대상으로 추적한다. 접촉면이 가려지면 확정 프레임 대신 마지막 비접촉/최초 접촉 확인 구간을 AI 관찰에 남긴다. |
| entry | 피해차량 바퀴가 ego 주행 차선에 처음 닿는 시점. 시작 전에 이미 진입했다면 영상 첫 프레임. | 첫 등장, 차체 중심, 검출 box 바닥을 바퀴로 대체하지 않는다. `0`은 현재 Nexar의 첫 번호일 뿐 모든 입력의 고정 답이 아니다. |
| side | 화면 기준 상대가 들어온 방향 LEFT/RIGHT. | 현재 위치·주행 벡터 방향과 진입 전 기원측을 분리한다. 시작부터 같은 차선인 상대의 진입측을 임의 생성하지 않는다. |
| space | 충돌 당시 실제 진입 가능한 물리적 회피 공간. | 속도·반응시간에 따른 회피 성공 가능성은 기준이 아니다. 차로·차량·연석·중앙분리대·도로 경계를 함께 보며, 빈 반대차선이라고 자동 1을 주지 않는다. |

근거: [필독 예측 기준](https://dacon.io/competitions/official/236753/talkboard/417186), [진입/이미 진입 Q&A — 9월 1일 답변](https://dacon.io/competitions/official/236753/talkboard/417277), [회피 공간 Q&A — 9월 4일 답변](https://dacon.io/competitions/official/236753/talkboard/417319). 마지막 Q&A는 검색 도구가 반환한 운영진 답변 본문을 확인했고 canonical 페이지 직접 열기는 도구 오류였다.

교차로에서 차선 연장 영역을 판단하며 무충돌·제3자끼리만 충돌한 장면을 제외한다는 기존 근거는 `대회_통합_정보.md:55–57`에 있다. 이번 웹 검색에서는 이 두 세부 답변의 원문 URL을 다시 식별하지 못했다. Nexar의 positive 폴더 이름만으로 그 두 조건에 해당한다고 확정하지 않는다. 실제 ego 접촉과 상대 식별은 영상별 재검토 대상이다.

실제 Stage2 입력은 영상 폴더별 **전체 프레임 이미지**다. 이전 10Hz+endpoint 진단과 전체 native PNG 진단은 다른 시간 샘플링 조건이다. 전체 PNG는 이 변수의 대조용이며 숨겨진 평가 JPEG 인코딩까지 동일하다는 증거는 없다. [입력 Q&A — 8월 31일 답변](https://dacon.io/competitions/official/236753/talkboard/417260)

제출값은 원래 파일명 프레임 번호이며 평가 서버가 영상별 시간 대응으로 변환한다. 누락·비수치·음수·범위 밖 번호·허용 범주 밖 값은 오답이다. 기존 통합 문서가 삽입 이미지까지 확인한 S2 산식은 contact/entry Accuracy@0.3초 각각 0.35, side/space Macro-F1 각각 0.15다. 이번 웹 텍스트 추출은 시간 대응 설명을 확인했으나 수식 이미지는 다시 렌더링하지 않았다. 이 9개 초안으로 공식 S2를 만들지 않는다. [평가 페이지](https://dacon.io/competitions/official/236753/overview/evaluation)

예측은 파일마다 독립이어야 한다. 파일 내 추적·분할·종합은 가능하나 다른 파일의 통계·예측을 쓰거나 비공개 평가자료로 모델 갱신/튜닝/pseudo-labeling하는 것은 금지다. 따라서 논문의 test-time adaptation을 그대로 제출에 넣을 수 없고, 다른 영상을 누적해 임계값을 조정해서도 안 된다. [규칙 페이지](https://dacon.io/competitions/official/236753/overview/rules)

## 2. 아홉 사람 검수의 적용 범위

아래는 **사용자가 기록한 값**이다. 초는 native PTS이고 표에서만 6자리로 반올림했다. 원본 JSON의 정밀도는 보존했다. 최신 00003은 이전과 답이 같고 notes/created_at만 달라졌다.

| Nexar ID | contact 프레임 / 초 | entry 프레임 / 초 | 시작부터 진입 | side | space |
|---|---|---|---|---|---|
| 00000 | 584 / 20.207612 | 555 / 19.204152 | 아니오 | LEFT | 0 |
| 00003 | 582 / 19.400000 | 0 / 0 | 예 | 미상 | 1 |
| 00004 | 579 / 19.300000 | 미상 | 아니오 | RIGHT | 1 |
| 00005 | 643 / 21.013072 | 643 / 21.013072 | 아니오 | LEFT | 1 |
| 00006 | 579 / 19.300000 | 0 / 0 | 예 | 미상 | 1 |
| 00007 | 625 / 20.833333 | 514 / 17.133333 | 아니오 | LEFT | 1 |
| 00008 | 582 / 19.400000 | 560 / 18.666667 | 아니오 | RIGHT | 1 |
| 00010 | 637 / 21.233333 | 572 / 19.066667 | 아니오 | LEFT | 1 |
| 00013 | 662 / 22.066667 | 569 / 18.966667 | 아니오 | RIGHT | 0 |

사용 가능 필드 수는 contact 9, entry 8, side 7, space 9다. 이는 완성도 수치이지 정답 개수나 성능 수치가 아니다. 빈 side를 LEFT/RIGHT로 보충하거나 uncertain entry를 0으로 바꾸지 않는다. observed entry 중 2개는 시작 이전 진입 관례이며 영상 안의 차선 횡단 사건 관측과 구분한다.

원본 파일 SHA, 전체 `JSON.stringify(case.frames)`의 Node SHA, 선택 contact/entry의 case PTS 및 PNG SHA는 이번에 모두 재대조해 일치했다. 근거 파일에는 각 실제 경로와 SHA가 들어 있다. 기존 독립 PyAV 원본 전수 PTS 및 선택 RGB 검증은 `research/v6_stage2/nexar_review_intake_integrity.json`, `nexar_review_intake_integrity_addendum.json`, `round2_validation/intake_20260915_actual/integrity.json`이다. 마지막 검사는 접촉 중심이어서, 이번에 새 3개 entry를 원본에서 순차 디코딩하여 RGB와 native PTS를 추가 대조했고 모두 일치했다. `review_entry_integrity.json`에 source/case/review/PNG 전후 SHA와 선택 프레임 픽셀 SHA를 기록했다. 파일 동일성과 의미 판독 정확성은 다른 검사다.

9개 모두 `human_review_draft`, `evaluation_eligible=false`, `source_group_id=PENDING_SOURCE_AUDIT_*`다. 단일 사람 검수이며 annotation_blinded는 자기 신고다. 동일 SHA가 없는 것은 동일 사건·차량·재편집 원천이 없다는 증거가 아니다. 초기 6개는 이미 개발/예약 검증에 사용됐고, 새 3개도 round2 평가 완료 후에는 더 이상 unseen이 아니다. 원본 JSON의 과거 exposure 필드는 역사적 기록으로 남긴다. PUBLIC_DEV_001 사람 검수 및 DADA AI 검토를 이 9개에 섞지 않는다.

### 확인된 검수 도구의 잠재 오류

`research/v6_review_tool/dist/app.js:11`의 show는 이미지 src를 바꾸면서 번호를 즉시 갱신하고, `:12`의 mark는 이미지 load/decode 완료 확인 없이 현재 index를 저장한다. 빠른 이동 직후 클릭하면 표시되던 픽셀과 저장 인덱스의 대응을 보장하지 못한다. 기존 `test_review.cjs`는 DOM 모사 시험이며 실제 느린 이미지 로딩을 검사하지 않는다. **9개에서 실제로 발생했다는 기록은 없다.** 따라서 이 발견만으로 사람 값을 틀렸다고 처리하지 않는다. 후속 수정은 로드된 프레임과 표시 번호의 일치·기록 버튼 활성화를 묶고, 늦게 완료한 과거 요청이 새 화면을 덮지 못하도록 해야 한다.

또한 export의 hash await 중 사용자가 case를 바꾸면 객체 구성의 앞부분과 나중에 복제되는 draft가 다른 시점일 수 있다. export 시작 때 source/case/draft 전체를 원자적으로 복제하거나 UI를 잠그는 것이 필요하다. 이를 실제 발생 오류로 단정하지 않는다.

**후속 승인된 수정 완료:** 배포 디렉터리의 `app.js`가 이 도구의 직접 실행 소스이며 별도 `src/app.js`는 없다. 이미지 preload와 표시 이미지 decode가 끝난 동일 요청에서만 이미지·번호를 보이고 접촉/진입 기록을 허용하도록 수정했다. 요청 revision으로 늦은 이전 로드를 무시하며, 실패 시 기록을 막는다. 시작부터 진입 버튼도 첫 프레임 표시를 기다린다. export는 await 전에 case와 draft를 함께 복제한다. `test_frame_loading.cjs`의 통제된 느린/역순/실패 decode·빠른 이동·case 전환·export 경합 검사와 기존 `test_review.cjs`가 모두 통과했다. 실제 브라우저에서의 느린 네트워크 시각 QA까지 수행한 것은 아니다. 기존 case/PNG/사람 JSON/모델은 변경하지 않았다. 현재 무결성 재확인 및 수정 파일 SHA는 `review_ui_fix_validation.json`에 있다. 위 원인 설명은 수정 전 경로의 기록이다.

### 제한적 AI 시각 관찰 — 사람 검수를 대체하지 않음

이번에는 원본 PNG 3장만 직접 보았다. 순차 영상 전체 판독이 아니며, 기존 사람 답을 읽은 뒤의 검토이므로 blind AI annotation이라고 부르지 않는다.

- `00005/frame_000643.png`: 왼쪽 근접 차량이 화면 하단을 크게 차지하고 바퀴와 ego 접촉면이 가려져 있다. 이 한 장으로 최초 진입과 최초 접촉이 동시에 발생했다는 사실은 확인 불가다. 같은 프레임 기록은 재검토 우선 신호이며 오류 확정이 아니다.
- `00008/frame_000560.png`: 야간 교차로이며 오른쪽 아래 화면 끝에 차량 일부가 보인다. 바퀴가 ego 연장 차선에 처음 닿았는지는 이 한 장으로 확인 불가다.
- `00013/frame_000569.png`: 주차 차량이 늘어선 도로, 오른편에서 도로 쪽으로 비스듬한 차량, 불명확한 차선 표시가 보인다. 상대 추적과 ego 주행 영역 판독 없이 정확한 wheel-entry를 확정할 수 없다.

### 실행할 AI 영상 검수 계획

1. 후보 출력은 숨기고, 원본과 기존 사람 초안은 다른 열로 보존한다. 이미 사람 값을 읽은 검토자는 비맹검임을 표시한다. AI 관측 JSON의 `reviewer_kind=AI`, `official_gt=false`, `human_annotation_replaced=false`를 고정한다.
2. 모든 9개를 동일 절차로 검토한다. 전체 시간축에서 ego의 직접 접촉 상대를 식별하고, 접촉 이후 외관·위치로 다시 확인한다. 여러 상대 또는 접촉 자체가 미확인이면 applicability=uncertain으로 둔다.
3. 같은 상대의 주행 전 구간을 따라가며 차선 경계/연장 영역, 바퀴 가시성, 최초 도달 이전·이후 native 프레임을 기록한다. 사람 시점 주변만 보아 그 답을 확인하는 방식은 피한다. 정확 프레임이 안 보이면 interval/unknown을 유지한다.
4. side는 그 동일 상대의 진입 전 기원측으로 판독한다. space는 사람 contact와 별개로 AI가 관측한 접촉 구간의 주변 도로를 확인한다. 상대 ROI만 잘라 주변 차로를 없애지 않는다.
5. 사람과 AI 불일치 목록을 출력하되 기존 라벨은 변경하지 않는다. AI 합의만으로 사람 간 독립 합의가 생겼다고 표시하지 않는다. 모호한 구간 안 예측을 정답으로 세는 대신 구간과의 거리 및 명백한 불일치만 진단한다.

## 3. 공간 grounding·추적 논문에서 적용할 점

아래는 정식 주요 학회 논문의 기능을 확인한 것이며 DACON 성능·현재 GPU 소요·라이선스 적합성을 검증한 결과는 아니다. 새 모델을 다운로드하거나 실행하지 않았다.

| 논문·1차 출처 | 확인한 기능 | V7 적용 가설과 채택 전 실패 조건 |
|---|---|---|
| [Grounding DINO, ECCV 2024](https://www.ecva.net/papers/eccv_2024/papers_ECCV/html/6319_ECCV_2024_paper.php) | 언어 범주·referring expression으로 open-set object box 검출 | 상대의 색/유형/기원측을 공간 후보로 구체화할 수 있다. 여러 검은 차 중 실제 접촉 상대를 고르는 능력은 별도다. 사람이 확인한 상대와 box가 다르면 추적 전에 실패로 기록한다. |
| [TAPIR, ICCV 2023](https://openaccess.thecvf.com/content/ICCV2023/html/Doersch_TAPIR_Tracking_Any_Point_with_Per-Frame_Initialization_and_Temporal_Refinement_ICCV_2023_paper.html) | query point의 프레임별 대응과 시간적 정제, 위치/가림 불확실성 추정 | 보이는 바퀴 접지 부근과 도로 경계 점을 각각 추적하는 근거다. 가린 바퀴를 차체 점으로 대체하거나 저신뢰 궤적을 최초 접촉으로 단정하면 실패다. |
| [SAM 2, ICLR 2025](https://openreview.net/pdf?id=Ha6RTeWMd0) | promptable 영상 segmentation, streaming memory | 상대 mask 연속성·다중 차량 분리를 점검할 수 있다. mask 안정성은 상대 identity/접촉/차선 의미의 증거가 아니며, 바퀴까지 정확한 경계를 보장하지 않는다. |

세 모델을 무조건 연결하지 않는다. 필요한 중간 산출물은 **같은 상대 identity, 가시 바퀴, ego 차선 영역, 원본 시간축**이다. 기존 LK ROI 시도가 실패한 배경 추적·대상 전이 문제를 새 tracker 이름만으로 해결했다고 주장하지 않는다. 이 중간 정합성이 개선되는지 별도 관측한 후에만 사건 시점 후보를 비교한다. 마스크·track이 불명확할 때 전체 화면으로 돌아오는 정책도 결과 전에 고정해야 한다.

## 4. 추가 Stage1 감사 — 자료 없는 training-free 변경의 한계

현재 `model/stage1/config.json`은 TPO/forensic 0.5 평균, threshold 0.5다. `solution/stage1.py:70–95`는 native 최대 192px 패치 5개의 주파수 통계를 시간 순서 없이 median/q90 집계한다. 최종 artifact는 주파수 22개만 사용한다. `solution/stage1_tpo_merged.py:39–48`은 전체 프레임을 224 정사각형으로 resize하고 12프레임 확률을 평균한다. 입력 검증과 LoRA merge 동등성은 기존에 확인됐지만, 실제 도로 재촬영 양성은 아직 0개다. 얼굴/물체 PAD의 print까지 포함한 공격 점수를 순수 화면 재촬영 확률로 해석하는 데도 도메인 차이가 있다.

확인된 로컬 손익: DLC 문서 진단에서 TPO FP/FN=1/4, forensic=0/10, 결합=0/6(각 클래스12개). comma 원본23개에서 TPO FP=4, forensic/결합 FP=0. 결합 조건은 `pF+pT>=1`이므로 낮은 forensic 점수는 TPO의 검출과 오탐을 함께 억제한다. DLC의 TPO 이득만으로 단독 교체하면 원본 오탐 위험을 무시한다. 공개 합성 clean에서 높은 점수와 축소 시 하락 역시 실제 재촬영 전이를 보장하지 못한다. 근거: `research/v5_postmortem_stage1.md`, `research/v5_stage1/diagnosis.md`, `research/stage1/dlc_subset/frozen_metrics.json`, `research/stage1/comma_original_diagnostic/metrics.json`.

| 논문·1차 출처 | 논문에서 확인한 사실 | 현재 적용할 수 있는 범위 |
|---|---|---|
| [Face De-Spoofing: Anti-Spoofing via Noise Modeling, ECCV 2018](https://openaccess.thecvf.com/content_ECCV_2018/html/Yaojie_Liu_Face_De-spoofing_ECCV_2018_paper.html) | 얼굴 spoof 영상을 live 성분과 spoof noise로 분해하는 학습 모델이며 제약·감독을 사용한다. | 물체 내용과 재촬영 흔적을 분리해 검사할 근거다. 지금의 고역 필터가 논문의 학습된 spoof noise와 같지 않다. 논문은 학습 없이 도로 재촬영을 판별하는 방법을 입증하지 않는다. |
| [Single-Side Domain Generalization for Face Anti-Spoofing, CVPR 2020](https://openaccess.thecvf.com/content_CVPR_2020/html/Jia_Single-Side_Domain_Generalization_for_Face_Anti-Spoofing_CVPR_2020_paper.html) | 실제 얼굴의 도메인 공통 표현과 공격 도메인 간 차이를 비대칭 학습한다. adversarial/triplet 학습을 포함한다. | 모든 공격을 단일 분포로 압축하거나 실제 원본/공격에 무조건 같은 불변성을 강요하는 가정을 재검토할 근거다. 얼굴 benchmark 결과는 블랙박스 검증이 아니며 comma 원본만으로 이 학습을 재현할 수 없다. |

위 두 CVF 검색 원문을 확인했다. 추가 직접 열기 요청은 403이었으며 상세 구현·가중치 라이선스까지 감사했다고 주장하지 않는다. 물리 화면 재촬영에 가까운 최신 물체 논문·TPO는 별도 문헌으로 취급한다. 여기서 TPO는 기존 `gurayozgur/TPO`이며 동명의 CVPR2025 Task Preference Optimization과 다르다.

**지금 가능한 training-free 작업은 성능 개선 확정이 아니라 원인 진단이다.** 기존 native 패치와 전체화면 점수를 보존하고 프레임별 변화·패치 위치·crop/resize 민감도를 동일 영상 안에서 측정할 수 있다. 하지만 새 scale 평균, 최대값 pooling, 테두리 가중치, TPO 비율을 적용하면 학습이 없어도 새 모델 정책이다. 문서/원본만 보고 채택하면 도로 재촬영 검증 부재가 유지된다. 목표 양성이 없으므로 양성 손실을 알 수 없어 개선 채택 근거는 현재 불충분하다.

V5의 고정 logit 결합 학습도 모든 gate를 실패했다. 같은 자료에서 다른 C/threshold를 다시 고르는 것을 근본 보완으로 제안하지 않는다. 다음 유효한 분해는 `research/v6_stage1/`의 실제 촬영 schema에 따라 도로 원본과 물리 재촬영의 원천·기기 분리를 확보한 후 동결 TPO/forensic/결합의 veto 손실과 자체 미검출을 세는 것이다. 소수 고정 표본의 무회귀 조건은 보수적 선택 규칙이며 통계적 일반화 보장이 아니다.

Stage3 논문·동역학 검토는 별도 담당자의 범위다. 이 문서의 바퀴/차선 및 물리 재촬영 관측으로 Stage3 조향·가감속 정답을 생성하지 않는다.
