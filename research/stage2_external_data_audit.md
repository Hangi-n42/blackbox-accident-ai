# Stage2 외부 사고 데이터 감사 — 2026-09-11

## 결론

**기술적 실행 후보는 CCD의 ego-involved 사고 부분집합이다.** 공식 사고 영상 묶음은 표시 용량 756.4MB이며, 공식 주석 1,500개 중 자차 관여 표시가 801개다. 논문은 사고 시작 라벨을 실제 crash 발생으로 정의한다. 다만 **데이터 영상 자체의 이용허락 범위를 확인하지 못했으므로, 현재 상태를 라이선스 검증 완료로 취급하지 않는다.** 기술적 취득 가능성과 대회 사용 허용을 구분해야 한다.

DoTA는 배포자가 원 영상 채널의 공유 허락을 받았다고 밝힌 장점이 있지만, 제공 시간 라벨은 실제 접촉이 아니라 사고가 불가피해진 시점이다. 55GB 배포를 받아도 대회용 첫 접촉 라벨을 새로 만들어야 하므로 이번 우선순위에서 제외한다.

이번 작업은 기존 `research/stage2_strategy.md`의 CCD 조사와 4B 검증 기록을 먼저 읽은 후, 빠져 있던 실제 다운로드 목록·작은 주석·라벨 정의·원 출처 이용조건을 보강했다. 영상이나 대형 압축파일은 다운로드하지 않았다. 모델·프롬프트·추론 소스는 변경하지 않았고 GPU를 사용하지 않았다. 외부인에게 문의를 보내지 않았다.

## 1. CCD — 조건부 1순위

### 원 출처와 접근

- 배포자 저장소: [Cogito2012/CarCrashDataset](https://github.com/Cogito2012/CarCrashDataset).
- 논문: [Bao et al., ACM MM 2020, §4.1](https://arxiv.org/pdf/2008.00334).
- 저장소가 연결한 [공식 Drive 루트](https://drive.google.com/drive/folders/1NUwC-bkka0-iPqhEhIgsXWtj0DA2MR-F).
- 루트에서 실제로 찾은 [videos 폴더](https://drive.google.com/drive/folders/1Rx4LCo-9AbAdPw5Zh7wpKhMKLabZ5oA8).

웹 검색 도구는 Drive 내부 목록을 표시하지 못했으나, 공식 공개 폴더의 HTML을 HTTP로 읽어 다음 항목을 확인했다. 이는 실제 다운로드 성공을 뜻하지는 않는다.

| 공식 파일 | 화면 표시 용량 | 확인 범위 |
|---|---:|---|
| [Crash-1500.zip](https://drive.google.com/file/d/1fmcwGhr8JT9YfLUrlcvuCZ3ychi2eyFP/view) | 756.4MB | 파일 목록·view 페이지 확인. 압축파일 본문 미취득 |
| [Crash-1500.txt](https://drive.google.com/file/d/13OgrD0-8cKG0X00MlA6JXr0G_JJmHYXg/view) | 270KB | 276,792 bytes 원문을 메모리에서 읽어 구조·집계 검증 |
| [Normal.zip](https://drive.google.com/file/d/11ErpWQmmV5au2JOQVxwYl3ebtuugdlan/view) | 6GB | 목록만 확인. 실제 접촉 검증의 첫 단계에는 불필요 |
| [배포 README](https://drive.google.com/file/d/11DbPw5WYXaBBm9NrxdhX0tN98aFwMlPT/view) | 5,748 bytes | 원문 확인. 별도 데이터 이용조건 발견하지 못함 |

Drive의 MB/GB는 화면 표시를 그대로 기록했다. 정확한 사고 ZIP byte 수·전송 속도·전체 다운로드 성공은 확인하지 않았다. 정상 영상 및 VGG 특징을 제외하고 사고 ZIP과 작은 주석만 취득하는 경로가 있다.

### 규모·ego 구분: 공식 주석 직접 집계

주석 URL: [공식 Crash-1500.txt 다운로드](https://drive.google.com/uc?export=download&id=13OgrD0-8cKG0X00MlA6JXr0G_JJmHYXg).

- SHA256: `eb793a9b68dc50dfbb2ce3e6be75a13e625da93098ade40e8eb4e0fc2c5c724a`.
- 총 1,500행, 서로 다른 clip ID 1,500개.
- `egoinvolve=Yes`: 801개, `No`: 699개.
- 모든 행에 50개 binary frame labels. 이번 파일에서 모두 0 이후 1로 전환되는 단조 배열이었다.
- `youtubeID` 숫자형 원천 그룹은 전체 133개, ego 부분집합 132개다. 영상 개수와 독립 원천 개수는 다르다.

이 값들은 원본 주석 직접 집계이며 모델 성능이나 영상 수작업 검수 결과가 아니다. `egoinvolve=Yes`만으로 상대가 다른 차량인지, 첫 물리 접촉이 화면에서 확인 가능한지, 다중 충돌 중 어느 사건인지까지 확정할 수 없다.

### 실제 접촉 GT와 나머지 항목

공식 논문 §4.1은 accident beginning time을 실제 car crash가 발생하는 시점에 붙였다고 설명한다. 5초 clip 안에서 사고 시작을 마지막 2초에 배치했다. 공식 배포는 10FPS이며 각 프레임의 사고 여부가 있다. 따라서 **첫 `binlabels=1`은 사고 발생시점의 공식 라벨로 사용 가능**하다. 다만 대회의 자차-상대차 **최초 물리 접촉**과 프레임 단위로 동일하게 검수됐다는 세부 지침·오차보증은 확인 불가다. [공식 논문](https://arxiv.org/pdf/2008.00334), [공식 주석 설명](https://github.com/Cogito2012/CarCrashDataset#annotation-format)

진입 첫 프레임, LEFT/RIGHT 진입, 충돌 당시 회피공간 GT는 공식 공개 필드에 **없다**. 논문 그림에 등장하는 사고 이유·참여자 tracklet을 현재 배포되는 GT로 오인하면 안 된다. 저장소는 상세 주석을 추후 공개할 항목으로 설명한다. [현재 배포 필드](https://github.com/Cogito2012/CarCrashDataset#annotation-format)

CCD는 사고 시점을 끝부분에 배치한 편집 데이터다. 이를 이용해 대회 입력에서 '마지막 2초' 또는 고정 비율을 답으로 삼으면 안 된다. 이 편집 편향은 임의 길이 영상의 실제 접촉 탐색 검증에 한계를 준다.

### 코드 라이선스와 데이터 이용조건

저장소의 [MIT LICENSE](https://raw.githubusercontent.com/Cogito2012/CarCrashDataset/master/LICENSE)는 Wentao Bao의 software 및 associated documentation에 대한 허락 문구다. 저장소 설명은 사고 영상을 YouTube에서, 정상 영상을 BDD100K에서 수집했다고 밝힌다. 공식 Drive README까지 확인했지만, **제3자 영상 자체에 적용되는 별도 이용허락·권리 정리·비영리 대회 검증 허용 조항은 발견하지 못했다.** 그러므로 “CCD 영상 모두 MIT” 또는 “데이터 이용 금지” 중 어느 쪽도 단정하지 않는다. 전자는 확인되지 않았고 후자도 명시돼 있지 않다.

대회 통합 문서의 CCD 사용 가능 답변은 라이선스 범위 준수가 전제이므로 이 빈칸을 대신 채워주지 않는다. 영상을 제출 ZIP에 넣지 않는 것만으로 이용조건 문제가 해소되는 것도 아니다. 유지관리자/데이터 권리자가 명시한 허용 범위를 확인해 기록하는 단계가 필요하다. 문의가 필요할 때 확인할 내용은 다음 두 가지다. 이 조사에서는 발송하지 않았다.

1. 공식 사고 clip을 비영리 AI 경진대회의 외부 검증 및 모델 선택에 사용할 수 있는가?
2. 데이터 영상·라벨의 적용 라이선스와 원 영상 권리 범위는 무엇이며, 수상 검증을 위한 데이터 제출/재배포 제한이 있는가?

### 논문 DOI·저작권 조건 추가 점검

- 출판 DOI [10.1145/3394171.3413827](https://doi.org/10.1145/3394171.3413827) 및 ACM Digital Library 본문 페이지는 이번 웹 조회에서 HTTP 403으로 열리지 않았다. 따라서 ACM 페이지의 추가 데이터 약관/보충자료 조건은 확인 불가다.
- [저자 arXiv 페이지](https://arxiv.org/abs/2008.00334)의 `view license`는 [arXiv 비독점 배포 허락](https://arxiv.org/licenses/nonexclusive-distrib/1.0/license.html)으로 연결된다. 이는 저자가 arXiv에 논문을 배포하도록 허락한 조건이며 CCD 영상 재사용 허락이 아니다.
- [논문 PDF 첫 페이지](https://arxiv.org/pdf/2008.00334)의 ACM 고지는 개인/수업 목적 논문 복제 조건과 타인 소유 구성요소의 권리를 존중해야 한다는 내용을 포함한다. 이 고지를 데이터 학습·검증 라이선스로 적용할 근거는 없다.
- 확인 가능한 논문·원 저장소·Drive README 범위에서 데이터 자체의 허용 범위를 추가로 확정하지 못했다. **영상 취득·학습·모델 검증은 진행하지 않았으며, 명시적인 데이터 이용범위가 확인되기 전에는 진행하지 않는다.** 이번에는 공개 메타데이터와 작은 주석을 읽고 집계한 범위에 머물렀다.

## 2. DoTA — 이번 접촉 GT 검증에는 후순위

원 출처는 [MoonBlvd/Detection-of-Traffic-Anomaly](https://github.com/MoonBlvd/Detection-of-Traffic-Anomaly)와 [저자 논문 §3](https://arxiv.org/pdf/2004.03044)이다.

| 항목 | 공식 자료에서 확인한 사실 |
|---|---|
| 규모 | 4,677개, 10FPS로 추출·주석. 해상도 1280×720 |
| ego 구분 | 사고 종류별 ego/non-ego 분리. `*`가 non-ego를 뜻함 |
| 시간 GT | anomaly 시작·종료. 시작은 사고가 불가피해진 시점이며 실제 crash 시점과 구분한다고 명시 |
| 공간 GT | 사고 참여자의 bbox/track ID. 대회 진입 첫 접촉·LEFT/RIGHT·회피공간 GT는 제공 목록에 없음 |
| 취득 | 공식 README에서 55GB Drive 배포와 10GB×5+5GB 분할 배포를 제공 |
| 현재 접근 확인 | 저장소 및 공식 배포 링크 존재 확인. Drive 링크는 웹 도구 내부 오류로 실제 파일 목록·본문 접근 확인 불가 |

[공식 저장소](https://github.com/MoonBlvd/Detection-of-Traffic-Anomaly), [논문 시간 주석 정의 및 ego 구분](https://arxiv.org/pdf/2004.03044)

배포자는 2021-11-13 업데이트에서 원 채널 작성자의 공유 허락을 받아 영상 clip을 직접 공유한다고 밝혔다. 이는 CCD 조사에서 찾지 못한 구체적인 원천 공유 허락 설명이다. 그러나 [MIT LICENSE](https://raw.githubusercontent.com/MoonBlvd/Detection-of-Traffic-Anomaly/master/LICENSE)는 역시 software/documentation 문구이고, 모든 하위 사용·대회 재배포를 허용하는 데이터 전용 계약은 이번 확인 범위에서 찾지 못했다. '원 제공자가 공유할 권한을 받음'과 '수령자의 모든 이용이 허용됨'을 동일시하지 않는다.

**부적합 사유:** `anomaly_start`를 `collision_frame`으로 쓰면 목표 자체가 달라진다. ego 차종 collision category를 골라도 첫 물리 접촉을 새로 검수해야 한다. 현재 필요한 소규모 외부 접촉 검증에 대해 CCD보다 취득량이 크고 추가 라벨 비용이 높다.

## 실행 제안: CCD 기반, 출처별로 분리한 소규모 접촉 검증

아래는 실행 계획이며 완료된 데이터 구축이 아니다. 이용범위를 확인하기 전에는 대회 학습/검증에 실제 영상을 투입하지 않는다.

1. 데이터 이용조건 기록 후 사고 ZIP과 공식 주석만 취득하고 SHA를 보존한다. 본 연구의 실제 사용범위를 외부 검증으로 한정해 별도 기록한다.
2. `egoinvolve=Yes`에서 시작하되, 자차-다른 차량의 실제 접촉이 관찰되는 clip만 사람이 검수한다. 보행자/장애물 접촉, 불명확한 ego 관여, 가려진 접촉, 다중 사건은 제외 이유를 기록한다. 실제 가용 개수는 아직 확인 불가다.
3. 대회 공개 5개와의 동일·중복 clip을 파일 해시와 영상 유사도로 확인하고, 일치하는 원천 `youtubeID` 그룹 전체를 개발 검증에서 제외한다. 서로 다른 compilation 영상에 같은 사고가 재등장할 수 있어 원천 ID 분리만으로 중복이 해결된다고 보증하지 않는다.
4. 처음에는 **서로 다른 사고 40개를 목표**로 접촉 시점만 검수하는 소규모 실험을 제안한다. 이 40은 실험 예산 제안이며 현재 확보 수량이 아니다. 모델 출력을 보지 않고 실제 접촉 frame과 근거 프레임, 불확실 구간을 기록한다. 공식 first-positive와 검수한 first-contact를 별도 컬럼으로 보존한다.
5. 출처 그룹별로 개발 부분과 untouched 검증 부분을 나누고, 2B·4B NF4의 현재 고정 소스부터 비교한다. 원본 순번과 10FPS 시간정보를 유지하며 ±0.3초를 측정한다. 라벨이 명확하지 않은 clip을 정답으로 강제하지 않는다.
6. 진입·방향·회피공간은 GT가 없는 상태로 남긴다. 충돌 점수로 Stage2 전체 개선을 주장하지 않는다. 진입 refinement 구조를 바꾸더라도 진입 GT 없이 정확도 우위를 판정하지 않는다.

현재 확보된 성과는 **실행 가능한 공식 취득 경로·주석 형식·801개 ego 후보·접촉 라벨 정의의 확인**이다. 독립 외부 검증셋 구축, 데이터 이용권 검증 완료, 첫 접촉 정답 검수 및 모델 비교는 아직 수행하지 않았다.
