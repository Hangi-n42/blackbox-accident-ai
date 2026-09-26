# CCD12 중복·출처 분리 감사

2026-09-20. 모델·예측을 읽지 않은 CPU 검사와 AI 시각적 동일사고 선별이다. contact/entry 정답 검수가 아니다.

**검사한 범위에서 중복으로 제외할 사례는 0개다. 완전한 원천·사건 독립성을 인증한 결과는 아니다.**

## 공개5 원CCD ID 대응

| 공개 파일 | 공식ZIP 멤버 | bytes | CRC32 | source group |
|---|---|---:|---:|---|
| 000001.mp4 | 000001.mp4 | 526859 | 1738312453 | CCD_YT_0000 |
| 000002.mp4 | 000002.mp4 | 286879 | 777372839 | CCD_YT_0000 |
| 000003.mp4 | 000003.mp4 | 427410 | 1691493424 | CCD_YT_0000 |
| 000004.mp4 | 000004.mp4 | 305244 | 4093236374 | CCD_YT_0000 |
| 000005.mp4 | 000005.mp4 | 363811 | 890767109 | CCD_YT_0010 |

공식 1,500개 ZIP central directory 전체에 대해 파일 크기와 CRC32를 비교했고 각 공개 파일이 위 멤버와 유일 일치한다. 공개 파일의 SHA256도 JSON에 보존했다. 원격 ZIP 멤버의 암호학적 SHA를 새로 취득·대조한 것은 아니다. 같은 공개 이름이라는 추정만으로 대응하지 않았다.

새 사전 선정 12개의 서로 다른 source group은 공개 그룹 CCD_YT_0000/0010과 겹치지 않는다. 공개 동일 클립·공식 그룹 중복으로 제외하는 사례는 없다. 원 YouTube 영상이 여러 사고의 편집본일 수 있어 source group 분리만으로 사건 분리를 보증하지 않는다.

## 실제 프레임 비교 범위

- CCD 12개: 전체 600 PNG.
- 공개 5개: 원 MP4에서 전체 250프레임.
- Nexar 64개: 각 원 MP4에서 시간상 균등 12장, 총 768프레임. 기존 40+9+3+6 및 9/19 추가 6이 포함된다. 이 검사는 6건의 접촉 재검수가 아니다.
- CCD 원 MP4 SHA256 대 공개/Nexar 69개 원 MP4: 일치 0.
- CCD 600프레임 대 비교 대상 1,018프레임의 같은 크기 RGB SHA256: 일치 0.
- 63bit DCT perceptual hash에 전체 영상·좌우반전·중앙 80% crop 변형을 비교했다. 각 CCD의 공개/Nexar 최인접 24쌍과 CCD 서로 다른 66쌍의 최인접 거리도 보존했다. pHash 값은 선별 특징이며 정답/통계적 독립성 기준이 아니다.

| CCD | 공개최소Hamming/63 | Nexar최소Hamming/63 |
|---|---:|---:|
| CCD_000174 | 10 | 12 |
| CCD_000688 | 16 | 14 |
| CCD_000801 | 14 | 14 |
| CCD_001024 | 14 | 14 |
| CCD_000453 | 16 | 12 |
| CCD_001477 | 14 | 10 |
| CCD_000703 | 14 | 14 |
| CCD_000139 | 14 | 14 |
| CCD_000196 | 14 | 12 |
| CCD_000303 | 14 | 12 |
| CCD_000815 | 12 | 14 |
| CCD_000728 | 12 | 14 |

최인접 24쌍 전부를 4장의 sheet로 시각 확인했다. 각 쌍은 도로 형태·건물·차량 배치 또는 주야 조건이 서로 달랐다. 거리 10인 CCD_000174/public000002는 수목 도로와 개방 간선도로, CCD_001477/Nexar00005는 서로 다른 교차로·주변 건물이다. CCD 간 최소 거리 10인 000703/000139는 낮 고속도로와 밤 고속도로이며, 최인접 4쌍 모두 시각상 같은 장면 근거가 없었다. 단일 프레임 대조로 같은 원 채널·기존 편집본의 모든 가능성을 소거하지는 못한다.

시각 증거: [nearest00](/Users/hyeongi/projects/blackbox-accident-ai/artifacts/stage2_goal_20260920/ccd_overlap_evidence/nearest_00.jpg), [nearest01](/Users/hyeongi/projects/blackbox-accident-ai/artifacts/stage2_goal_20260920/ccd_overlap_evidence/nearest_01.jpg), [nearest02](/Users/hyeongi/projects/blackbox-accident-ai/artifacts/stage2_goal_20260920/ccd_overlap_evidence/nearest_02.jpg), [nearest03](/Users/hyeongi/projects/blackbox-accident-ai/artifacts/stage2_goal_20260920/ccd_overlap_evidence/nearest_03.jpg), [CCD 간 4쌍](/Users/hyeongi/projects/blackbox-accident-ai/artifacts/stage2_goal_20260920/ccd_overlap_evidence/within_top4.jpg). 각 개별 원 프레임 경로와 0부터 시작하는 순번은 JSON에 있다.

## 한계와 적용

Nexar 전체 프레임·시간 정렬·임의 crop/회전·자막 제거·다른 원천 카메라를 전수 탐색하지 않았다. DADA/MM-AU와의 새 교차 비교도 본 검사 범위 밖이다. 서로 다른 편집본에 같은 사고가 포함되는 가능성은 남는다. 따라서 결론은 "공개 CCD 그룹 분리 확인 및 검사한 프레임에서 중복 미발견"이다.

부모 사전 선정 12개와 고정 평가 프로토콜을 유지해 다음 전문가 검수로 진행할 수 있다. 이 감사만으로 자차 실제 접촉·상대차 식별·진입 GT 또는 Private 점수 개선을 확정하지 않는다. 공급자 라벨과 전문가 라벨은 독립 보존한다.

재실행 스크립트: `ccd_public_overlap.py`; 실행 Python: `/Users/hyeongi/projects/blackbox-accident-ai/artifacts/mac_experiments/scipy_compat/.venv/bin/python`. 결과 JSON은 사전 선정·주석·ZIP 메타·스크립트 SHA를 보존한다.
