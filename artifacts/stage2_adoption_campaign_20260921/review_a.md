신규 CCD 12건의 48개 전체 시트(600프레임), 각 f0 원본 및 JSON에 열거한 핵심 원본을 실제 시각 검수했다. 모델 출력·동료 참조·공급자 시점/개별 라벨은 열람하지 않았다. 기존 개발 사례에 노출된 AI 검수이며 인간 검수나 공식 GT가 아니다.

| ID | 진입 참조 | 근거 층 | 적격 제안 |
|---|---|---|---|
| CCD_000037 | unknown | insufficient_boundary_evidence | False |
| CCD_000041 | unknown | insufficient_boundary_evidence | False |
| CCD_000062 | unknown | insufficient_boundary_evidence | False |
| CCD_000083 | before_start [0, 0] | conditional_unmarked_lane | True |
| CCD_000295 | unknown | insufficient_boundary_evidence | False |
| CCD_000421 | during_clip [18, 28] | conditional_unmarked_lane | True |
| CCD_000923 | during_clip [28, 30] | strict_boundary_evidence | True |
| CCD_001007 | before_start [0, 0] | conditional_unmarked_lane | True |
| CCD_001046 | unknown | insufficient_boundary_evidence | False |
| CCD_001111 | unknown | insufficient_boundary_evidence | False |
| CCD_001147 | unknown | insufficient_boundary_evidence | False |
| CCD_001244 | before_start [0, 0] | strict_boundary_evidence | True |

엄격한 표시 경계 근거는 923의 [28,30], 1244의 시작 내부 상태이다. 83·421·1007은 무표시 통로/연석 연장에 의존하는 조건부 참조로 별도 취급해야 한다. 눈으로 경계가 사라지거나 상대/충돌 연결이 불충분한 7건은 UNKNOWN을 유지했다. before_start의 0/0은 제출 인코딩이며 물리 진입이 f0에 발생했다고 주장하지 않는다.

421은 원본 640×360, 나머지는 1280×720이며 모든 시간은 intake의 PTS 매핑을 사용했다. 범위의 중간점을 확정 정답으로 변환하지 않았다. JSON에 상대 연속성, 충돌 근거, 가림/불확실성과 실제 열람 경로를 기록했다. 원본과 기존 참조는 변경하지 않았다.
