# Stage2 실제 보유 자료·CCD 소량 취득 근거 감사

2026-09-20. 이 감사는 메타데이터·실제 파일·공식 배포 문서를 읽었으며 모델 예측을 열거나 추론하지 않았다. 새 CCD 취득과 전문가 정답 검수는 부모 에이전트가 별도로 수행한다.

**지금 투입할 경로는 사전 선정된 CCD 자차 관여 12건이다.** 실제 MP4·50장 전체 프레임·native PTS가 각 폴더에 준비되었다. 이 12건을 검수 전에 '대회 정답 12건 확보'로 세지 않는다. Nexar 00024~29의 판독 불가 6건을 다시 검수하거나 새 Nexar를 계속 받는 경로는 중단한다.

## 실제 확보와 검수 가능성

| 원천 | 확인한 실제 보유 | 지금의 제한·용도 |
|---|---|---|
| 기존 MP4 inventory | 96개 파일 경로. public Stage1/3, 계약 fixture 중복 포함 | 96개 독립 Stage2 사고가 아님. 그중 이전 Nexar58개(40+9+3+6), 9/19 추가6개를 합쳐64개 |
| CCD 새 소량 | 부모 사전선정12건, 서로 다른 공식 YouTube 그룹, 전체600 PNG 및 원본 MP4/native PTS | 공급자 ego=Yes·실제 사고 시작 주석이 있는 검수 후보. 상대차·첫 접촉·진입·방향·공간은 검수 전 미확정 |
| DADA | `research/v5_external/dada/inputs/images/`에16개 PNG 시퀀스와 공식 XLSX, 원 MP4 미확보 | 이미 AI 검수 노출. exact GT0, native PTS 없음. 아래3개도 초 단위 새 holdout으로 승격 불가 |
| MM-AU | `research/v6_stage2/mmau_probe/cap_8_009509/images/`223 JPEG와 공식 주석 | t_ai20/t_co90/t_ae200 존재. 원시간·0/1기준·상대 설명 불일치 미해결. 반복 다운로드로 해결되지 않음 |
| CADP | 이 감사의 로컬 범위에서 영상 미발견 | 공식 CCTV 사고 자료여서 자차 dashcam 입력을 직접 대체하지 못함 |
| DoTA | 이 감사의 로컬 범위에서 영상 미발견 | 사고 bbox/trackID·ego 구분은 있지만 anomaly 시작은 실제 첫 접촉과 다른 목표. 이번에는 후순위 |

정확 원본 경로, 공급자 최초 positive 순번, 프레임 디렉터리·PTS·manifest는 같은 폴더의 `data_source_audit.json`에 각 CCD12건별로 적었다. 현재 취득 완료는 `ccd_intake/inputs.json`으로 확인했다. 공식 전체 ZIP 크기는 HTTP Content-Range 기준793,091,291bytes이며 전체 다운로드가 아니다. 실제 선택12건의 취득량·CRC·SHA는 부모 acquisition 산출물을 따른다.

## CCD 선정이 가능한 근거와 한계

[공식 CCD 저장소](https://github.com/Cogito2012/CarCrashDataset)는 dashcam 사고 분석용 데이터로 배포하고 공식 Drive 다운로드와 논문 인용을 안내한다. 각 사고 영상은50프레임/10fps이며 공개 필드는 사고 프레임 여부·원본 시작 프레임·YouTube 그룹·ego 관여·날씨·명암이다. 이번 확인한 배포 README에 로컬 연구 검수 금지나 사용 전 추가 승인 요구는 없었다. 저장소의 MIT는 software/documentation 문구이므로 모든 제3자 원영상의 권리가 MIT라고 확대 해석하지 않는다. [저장소 LICENSE](https://raw.githubusercontent.com/Cogito2012/CarCrashDataset/master/LICENSE)

[공식 논문 §4.1](https://arxiv.org/pdf/2008.00334)은 사고 시작을 실제 crash 발생 시점에 주석했다고 설명한다. 따라서 공급자 first-positive는 위험 onset/near-miss 라벨보다 이번 접촉 진단과 더 직접 연결된다. 다만 ego=Yes만으로 다른 차량과의 접촉이 보이는지, 다중 충돌 중 같은 상대인지, 대회의 최초 물리 접촉 프레임인지가 보증되지는 않는다. 공급자 라벨과 전문가 시각 라벨을 서로 덮어쓰지 않는다.

대회 통합 정보222행은 CCD 등 공개 데이터의 라이선스 범위 내 학습·검증을 허용한다는 운영진 답변을 기록한다. 이번 짧은 웹 검색으로 원 Q&A 게시글 URL은 복구하지 못했다. 그 문서213~215행의 공개·비영리 활용 및 출처/이용조건 준수 규칙과 사용자 명시 승인에 맞춰 실제 사용은 **로컬 외부 검증, 원본 비재배포**로 기록한다. 데이터 자체 권리의 포괄 검증 완료라는 법적 결론은 내리지 않는다.

기존 `research/stage2_external_data_audit.md:65,88`의 '명시적 데이터 이용범위 확인 전 진행하지 않는다'는 당시 분석가의 실행 방침이다. 이를 배포자 또는 대회 운영진의 승인 요구로 재해석해 이미 승인된 소규모 연구 검수를 자동 중단할 근거는 확인되지 않았다.

## 기존 자료에서 가장 좁혀졌던 DADA3건

| ID·정확 PNG 디렉터리 | 과거 AI 접촉 union | 과거 AI 진입 union | 판단 |
|---|---|---|---|
| `research/v5_external/dada/inputs/images/DADA_8_036/` | [119,125] | [102,120] | 조건부 후보, exact GT 아님 |
| `research/v5_external/dada/inputs/images/DADA_11_129/` | [71,83] | [25,68] | 조건부 후보, exact GT 아님 |
| `research/v5_external/dada/inputs/images/DADA_10_142/` | [166,184] | [155,169] | 조건부 후보, exact GT 아님 |

숫자는 1부터 시작하는 원본 PNG 번호다. `research/v5_external/dada/blind_review/consensus_frozen.md`의 기존 두 AI 검토 합의이며 이번에 새로 확정한 사람 GT가 아니다. `decision_external_not_run.md`는 계획된 외부 모델 추론을 실행하지 않았다고 기록한다. 예측 미노출 장점은 있지만 공급자 사건창을 보고 만든 sheet와 과거 검수에 노출됐고, native PTS 부재가 남아 있어 지금 CCD 대신 재활용하지 않는다.

## 독립성 검사와 다음 한 단계

공개 예제5개도 CCD 유래다. 공식 ZIP 전체1500개 central directory의 파일크기+CRC32와 공개5개를 비교하면 공개000001~000005가 같은 CCD 원ID와 유일 일치한다. 원천 그룹은 앞4개가 `CCD_YT_0000`, 다섯째가 `CCD_YT_0010`이다. 새 선정12건의 원천 그룹은 모두 이2개와 다르다. 이 결과는 `ccd_public_overlap.json/md`로 별도 보존한다.

CCD 12개 전체 600프레임을 공개 5개 전체 250프레임 및 기존 Nexar 64개에서 균등 12장씩과 비교 완료했다. 파일 SHA와 RGB exact match는 0건이며, 전체/좌우반전/중앙 80% crop perceptual hash의 최인접 24쌍 및 CCD 간 4쌍을 시각 확인했다. 검사한 범위에서 중복으로 제외할 사례는 없다. 같은 compilation이 아닌 두 영상에 같은 사고가 포함될 수 있으므로 그룹 분리만으로 완전 독립성을 인증하지 않는다. CCD1500개 원본 전체를 받지 않아도 공개 그룹 제외는 가능하다.

다음은 선정 ID를 유지한 채 중복 제외 결과를 반영하고 모델 출력 없이 전문가 검수를 동결하는 것이다. 정확히 관측되지 않는 필드는 unknown으로 남기고, 공급자 accident frame과 시각 first-contact를 별도 기록한다. 미리 고정한 후보의 검증에만 사용하며 결과가 나쁘다는 이유로 추가 데이터·기준·모델을 바꿔 같은 검증을 되풀이하지 않는다. CCD의5초·10fps·후반 사고 편집은 장시간 비공개 영상 전체 탐색 성능을 대표하지 않으며 접촉 개선을 Stage2 전체 개선으로 계산하지 않는다.

보조 1차 출처: [CADP 공식 프로젝트](https://ankitshah009.github.io/accident_forecasting_traffic_camera), [DoTA 공식 배포](https://github.com/MoonBlvd/Detection-of-Traffic-Anomaly), [MM-AU owner FPS 답변](https://huggingface.co/datasets/JeffreyChou/MM-AU/discussions/1). 이 감사는 새 대용량 다운로드·학습·모델 실행·제출을 하지 않았다.
