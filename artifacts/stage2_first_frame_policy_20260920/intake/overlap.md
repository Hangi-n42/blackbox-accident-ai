# 신규 CCD12 원천·기존 사고 중복 검사

**검사한 범위에서 중복 제외 0건. 완전한 사고 독립성을 인증하지 않는다.** 공개 CCD2그룹과 기존CCD12그룹을 제외했고 신규12건도 서로 다른 그룹이다. ID 해시 선정 후 영상·평가 결과에 따른 교체는 없었다.

| 비교 범위 | 프레임 |
|---|---:|
| 신규 CCD12 전체 | 600 |
| 기존 CCD12 전체 | 600 |
| 공개5 전체 | 250 |
| Nexar64 × 균등12 | 768 |
| DADA16 × 균등12 | 192 |
| MM-AU1 × 균등12 | 12 |

신규12 MP4 대 기존81 MP4의 SHA256 일치0건. 신규600 대 비교대상1,822프레임의 동일 크기 RGB SHA256 일치0건. 기존63bit DCT pHash의 전체·좌우반전·중앙80% crop 함수를 재사용했다. 공개5의 ZIP member 유일 CRC/크기 대응도 다시 확인했다.

신규 영상마다5원천별 최인접60쌍과 신규12내66쌍 중 최인접6쌍을 AI가 직접 시각 확인했다. 도로·건물·차량 배치 또는 날씨/주야 맥락이 다르며 같은 사고로 제외할 근거를 찾지 못했다. 최소Hamming거리8인 CCD_000643/Nexar00013은 눈 덮인 고층 아파트 도로와 가을 주택가로 구분된다. pHash 거리는 중복 판정 임계값이 아니며 제외 기준으로 단독 사용하지 않았다.

이 검사는 600×1,822 지정 프레임 비교와66개 신규 영상쌍 특징 비교다. Nexar/DADA/MM-AU 모든 프레임, 시간 정렬, 임의 crop·회전·다른 카메라를 검사하지 않았다. 같은 사고가 다른 compilation에 수록됐을 가능성은 남는다. Car Crashes Time 등 공통 유통 워터마크는 원촬영 사건의 동일성을 보증하지 않는다. 출처 그룹 분리와 검사 범위를 넘는 독립성을 주장하지 않는다.

[세부 JSON](/Users/hyeongi/projects/blackbox-accident-ai/artifacts/stage2_first_frame_policy_20260920/intake/overlap.json) · [재현 코드](/Users/hyeongi/projects/blackbox-accident-ai/artifacts/stage2_first_frame_policy_20260920/intake/overlap.py) · [검수 자료 입구](/Users/hyeongi/projects/blackbox-accident-ai/artifacts/stage2_first_frame_policy_20260920/intake/review_packet.md)

[최인접 시트0](/Users/hyeongi/projects/blackbox-accident-ai/artifacts/stage2_first_frame_policy_20260920/intake/overlap_evidence/nearest_00.jpg) · [최인접 시트1](/Users/hyeongi/projects/blackbox-accident-ai/artifacts/stage2_first_frame_policy_20260920/intake/overlap_evidence/nearest_01.jpg) · [최인접 시트2](/Users/hyeongi/projects/blackbox-accident-ai/artifacts/stage2_first_frame_policy_20260920/intake/overlap_evidence/nearest_02.jpg) · [최인접 시트3](/Users/hyeongi/projects/blackbox-accident-ai/artifacts/stage2_first_frame_policy_20260920/intake/overlap_evidence/nearest_03.jpg) · [최인접 시트4](/Users/hyeongi/projects/blackbox-accident-ai/artifacts/stage2_first_frame_policy_20260920/intake/overlap_evidence/nearest_04.jpg) · [최인접 시트5](/Users/hyeongi/projects/blackbox-accident-ai/artifacts/stage2_first_frame_policy_20260920/intake/overlap_evidence/nearest_05.jpg) · [최인접 시트6](/Users/hyeongi/projects/blackbox-accident-ai/artifacts/stage2_first_frame_policy_20260920/intake/overlap_evidence/nearest_06.jpg) · [최인접 시트7](/Users/hyeongi/projects/blackbox-accident-ai/artifacts/stage2_first_frame_policy_20260920/intake/overlap_evidence/nearest_07.jpg) · [최인접 시트8](/Users/hyeongi/projects/blackbox-accident-ai/artifacts/stage2_first_frame_policy_20260920/intake/overlap_evidence/nearest_08.jpg) · [최인접 시트9](/Users/hyeongi/projects/blackbox-accident-ai/artifacts/stage2_first_frame_policy_20260920/intake/overlap_evidence/nearest_09.jpg)
[신규 내 최인접6쌍](/Users/hyeongi/projects/blackbox-accident-ai/artifacts/stage2_first_frame_policy_20260920/intake/overlap_evidence/within_top6.jpg)

새 모델 호출0. 모델 예측·참조 정답을 중복 검사에 사용하지 않았고 기존 파일은 수정하지 않았다.
