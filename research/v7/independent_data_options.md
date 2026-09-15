# 독립 접촉 시점 자료: 제한된 공개 자료 조사

2026-09-15. 기존 조사부터 확인한 뒤 MM-AU, PSAD, ACCIDENT 세 후보만 1차 자료로 확인했다. 새 영상·가중치 다운로드, 설치, 모델 실행, 사람/AI 주석 생성은 하지 않았다.

**결론: 이번 조사에서 즉시 사용할 수 있는 ‘사용권 확인 + 실제 자차 접촉 + 독립 사람 시점 주석 + 입증된 초 단위 대응 + 기존 자료와 중복 없음’을 모두 만족한 새 소규모 집합은 확보하지 못했다.** 검색 범위 안에서 확인하지 못했다는 뜻이며 그런 데이터가 존재하지 않는다는 주장이 아니다. MM-AU는 사람 충돌 프레임 감사 후보, PSAD는 조건 확인 후 검토 후보다. 새로 확인한 ACCIDENT는 최초 가시 접촉을 여러 사람이 주석하지만 CCTV 시점이라 현재 자차 접촉 검증의 대체물이 아니다.

## 기존 실패를 다시 시도하지 않은 근거

- `research/v6/external_source_audit.md`: Nexar positive에는 collision과 near-miss가 함께 있으므로 `time_of_event`를 접촉 정답으로 변환하지 않았다. 원본 영상 PTS와 사용 조건은 확보했지만 실제 접촉 여부·최초 접촉 기준은 별도 검수가 필요하다. 이미 쓴 9개 사람 초안은 새 독립 검증으로 재사용할 수 없다. [공식 Nexar 설명](https://huggingface.co/datasets/nexar-ai/nexar_collision_prediction/blob/main/README.md), [공식 라이선스](https://huggingface.co/datasets/nexar-ai/nexar_collision_prediction/blob/main/LICENSE)
- `research/v5_external/dada/blind_review/consensus_frozen.md`: 기존 DADA 12개+reserve 4개는 대상·접촉 식별의 불확실성과 PNG 시간 대응 한계 때문에 공식 정확 GT 0개로 보존했다. 조건부 기원측 6개와 넓은 entry 구간 5개를 정확 접촉 정답으로 승격하지 않는다.
- `research/v6_stage2/mmau_feasibility.md`: MM-AU CAP 한 시퀀스 223장을 이미 소량 취득했다. 새로 같은 아카이브를 받는 것은 원본 FPS 부재를 해결하지 않는다.

## 후보 1: MM-AU — 외부 사람 충돌 프레임은 있으나 초 단위 대응이 막힘

**정의·사람 주석:** CVPR 2024 논문 §3은 다섯 자원자가 사고창을 표시하고 프레임 인덱스를 평균한다고 설명한다. `t_ai`(사고창 시작), `t_co`(충돌 시작), `t_ae`(끝)를 분리한다. 따라서 t_co를 단순 위험 시작으로 취급할 이유는 없지만, 가려진 자차의 최초 물리 접촉을 어떻게 확정하는지에 대한 세부 판독 매뉴얼은 확인되지 않았다. 데이터 전체가 자차 직접 충돌만이라는 보장도 없다. [공식 논문](https://openaccess.thecvf.com/content/CVPR2024/papers/Fang_Abductive_Ego-View_Accident_Video_Understanding_for_Safe_Driving_Perception_CVPR_2024_paper.pdf), [공식 메타데이터 정의](https://github.com/jeffreychou777/LOTVS-MM-AU)

**단위·접근:** 단위는 프레임이다. 공식 HF에 이미지 시퀀스 아카이브와 GitHub 메타데이터가 공개돼 있고 소량 취득은 기존 기록에서 실제 성공했다. 그러나 저장소 owner는 원본 FPS가 영상마다 달랐고 현재 원본 영상/FPS를 제공할 수 없다고 답했다. 임의 30fps나 자연스럽게 보이는 재생률을 지정해 ±0.3초로 평가할 수 없다. [공식 owner 답변](https://huggingface.co/datasets/JeffreyChou/MM-AU/discussions/1)

**사용 조건:** 데이터 카드 CC-BY-NC-4.0, GitHub academic-use 문구를 확인했다. 코드 라이선스를 데이터 허가로 대신하지 않는다. 비상업 연구 검토 가능성과 이 상금 대회에서의 사용·재배포 적합성은 별도이며 이번 조사에서 후자를 확정하지 않았다. [공식 데이터 카드](https://huggingface.co/datasets/JeffreyChou/MM-AU/blob/main/README.md)

**선택 판단:** 이미 확보한 시퀀스의 source/ego/type·번호 체계를 확인하면 프레임 순서 기반 후보 누락 감사에는 조건부 유용하다. 독립 초 단위 채택 검증은 원본 시간 대응이 복원될 때까지 불가. 원천이 CCD/A3D/DoTA/DADA 및 공개 플랫폼이므로 이름이 다른 데이터셋이라는 이유로 중복 없음도 보장되지 않는다.

## 후보 2: PSAD — 시간 정보 안내는 있으나 라벨·권리 확인이 먼저

공식 저장소는 DoTA에서 고른 2,724개 1280×720, 30fps 영상을 설명하고 anomaly start / collision timestamp / anomaly end를 구분한다. 영상은 Baidu, scene annotations는 Google Drive에서 받도록 안내한다. object_labels의 frame_id는 10fps이고 원본 번호는 그 세 배라는 주의사항도 있다. 이 규칙이 collision timestamp에도 그대로 적용된다고 단정하지 않는다. collision timestamp의 단위·0/1 시작·자차 직접 접촉만인지·사람 최초 접촉 합의 방식은 이번에 확인한 공개 README에서 확정되지 않았다. [공식 PSAD 저장소](https://github.com/Shun-Gan/PSAD-dataset)

저장소의 공개 파일 목록과 README에서 데이터 이용 라이선스는 확인하지 못했다. 다운로드 링크의 존재는 대회 사용 허가가 아니다. 사용 불가가 확정된 것도 아니다. **현재 취득·학습·채택하지 않고, 데이터 사용 조건과 collision 필드 정의가 확인되면 소량 원본+주석 대응을 다음 단계로 삼는다.** DoTA 파생이므로 MM-AU·기존 인터넷 영상과의 source 중복 확인도 필요하다.

## 후보 3: ACCIDENT — 더 명확한 사람 접촉 주석, 그러나 자차 영상이 아님

공식 논문 §3.1은 impact frame을 ‘최초 가시 접촉’으로 정의하고 3–5명 독립 검토자의 시점 median을 설명한다. §5.1은 초 단위 GT를 여러 검토자의 average라고 써서 합의 연산 설명에 차이가 있다. 실제 배포 메타데이터/평가기 확인 전 두 연산을 동일하다고 간주하지 않는다. 시간 출력은 초이며 영상별 FPS 변환을 설명한다. 이 자료는 단순 anticipation cue보다 접촉 정의가 명시적이다. [원저자 논문](https://arxiv.org/html/2604.09819v1)

공식 프로젝트는 실제 CCTV 2,027개와 CARLA 합성 2,211개를 구분한다. 움직이는 자차가 장착한 블랙박스가 아니며 자차 접촉·ego lane·ego evasion 정의가 없다. 그러므로 전체화면 motion-jerk 모델이나 현재 Stage2의 자차 접촉 정확도를 검증하는 도메인 대체물이 될 수 없다. [공식 프로젝트](https://accidentbench.github.io/)

논문 부록 B는 주석/메타데이터 CC-BY-4.0, 코드/CARLA 자산 Apache-2.0, 원본 영상은 재배포·파생 허용된 원천만 포함한다고 설명한다. 동시에 연구 목적 제한을 명시하므로 각 영상 원천 조건과 실제 배포 약관 확인이 필요하다. 주석의 CC-BY를 모든 영상의 동일 라이선스로 확대하지 않는다. [공식 라이선스·배포 설명](https://arxiv.org/html/2604.09819v1#A2), [저자 GitHub](https://github.com/accidentbench/ACCIDENT), [저자가 연결한 Kaggle 배포](https://www.kaggle.com/datasets/picekl/accident)

Kaggle 연결은 확인했으나 이번 도구에서 파일 목록·약관 본문은 읽히지 않았다. 실제 소량 다운로드 가능 여부와 파일별 주석 완결성은 미검증이다. 향후 일반적인 가시 접촉 인식 연구에는 후보가 되지만 **현재 요청의 독립 ego-contact 채택 gate용으로는 제외**한다.

## 다음에 할 수 있는 좁은 행동

즉시 새 데이터셋을 크게 받기보다 MM-AU의 이미 보유한 시퀀스에서 source URL/ego 사고형/프레임 번호를 확인하는 읽기 감사가 가장 작은 후속이다. 이때 결과는 초 단위 정확도가 아니라 기존 사람 t_co에 대한 프레임 포함·순서 오차로 제한한다. 원본 시간 대응이 없으면 그 결과로 대회 성능 채택을 결정하지 않는다.

실제 초 단위 독립 검증을 열려면 (a) 사용 허가와 원본 시각이 있는 **아직 예측을 보지 않은** 자차 직접 충돌 원본, (b) 최초 물리 접촉 정의와 가려짐 처리 규칙, (c) 독립 복수 검수와 불일치/불확실성 보존, (d) 기존 9개 및 원천 파생과의 중복 검사까지 필요하다. 새 데이터셋 이름이나 단일 접촉 숫자만으로 이 조건을 충족했다고 쓰지 않는다. 이번 조사에서 새 GT, 새 검증 통과, 예상 개선 수치는 없다.
