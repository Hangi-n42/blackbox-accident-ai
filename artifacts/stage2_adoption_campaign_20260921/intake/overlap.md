# 신규 CCD 12건 출처·중복 감사: PASS_WITH_SCOPE_LIMITS

공식 metadata의 ego=Yes 및 전체 CCD_ID SHA256 순서로 선정12를 독립 재계산했다. 공개 source_group2개와 이전 CCD24개 그룹을 모두 제외했고 신규12개도 서로 다르다. 원본 영상12·PNG600·RGB600·PTS 파일 해시가 manifest와 일치하며 native PTS × time_base 600개를 원번호 시각과 정확 비교했다.

- 신규600프레임 대 이전 CCD1200 + 공개250 + Nexar768(64영상 각12) + DADA192(16클립 각12) + MM-AU12 = 참조2422프레임 검사.
- 전체 영상93개(이전 CCD24·공개5·Nexar64)와 비교한 신규 영상 SHA256 중복0. 신규12 영상 해시도 서로 다름.
- 검사 프레임의 원크기 RGB 완전일치0. 63bit DCT pHash의 원본/좌우반전/중앙80% 조합으로 비교.
- 12건×5원천 최근접60쌍 전부와 신규 내부66쌍 중 최근접6쌍을 직접 시각 검수: 확인된 동일 장면0. pHash 최소6도 도로 구조·차량이 다른 거짓 유사 후보였다.
- 공개5의 byte size+CRC32를 전체 CCD ZIP member 목록에 대조해 각각 단일 member에 대응하고 source_group0000/0010을 확인했다. 이는 전체 공식 원본과의 SHA256 동일성 증명은 아니다.

취득 압축 member 합계 8,258,191bytes, 취득 기록 전체 전송 9,044,971bytes, 추출 MP4 8,864,221bytes. 기존 감사·정답·예측·모델은 수정하지 않았다. 신규 모델 호출0, 신규 예측/정답 참조 미열람.

**한계:** 공급자 source_group은 YouTube 원천/편집 묶음이다. 원 촬영 사고 식별자가 아니므로 완전한 사고 독립성을 인증하지 않는다. Nexar/DADA/MM-AU는 균등12프레임 표본이며 미표본·재편집·다른 crop 복제는 미검증이다. 이번 비교에서 중복을 확인하지 못했다는 제한된 결론이다.

상세 좌표·경로·해시·시각근거는 [overlap.json](/Users/hyeongi/projects/blackbox-accident-ai/artifacts/stage2_adoption_campaign_20260921/intake/overlap.json)에 보존했다.
