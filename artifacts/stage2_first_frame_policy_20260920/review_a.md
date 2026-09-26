독립 시각 검수 A — 새 12개 사례

전체 48장 시트에서 600개 프레임을 확인하고, 각 f0 및 진입/상대 식별에 필요한 원본을 직접 확인했다. 실제로 본 원본 목록과 시트 경로는 review_a.json에 기록했다. 모델 예측·공급자 사고 라벨·동료 검수를 열람하지 않았고 추가 모델 호출 및 원본 변경은 하지 않았다.

| ID | 상대 | f0 차로 상태 | 진입 | f0 표식 |
|---|---|---|---|---|
| CCD_001498 | 미확정 | UNKNOWN | unknown | unknown |
| CCD_000585 | 미확정 | UNKNOWN | unknown | absent |
| CCD_001278 | 식별 | UNKNOWN | during_clip 13–16 | occluded |
| CCD_001454 | 식별 | UNKNOWN | unknown | absent |
| CCD_000103 | 식별 | UNKNOWN | unknown | absent |
| CCD_000470 | 식별 | UNKNOWN | unknown | unknown |
| CCD_001237 | 식별 | OUTSIDE | during_clip 12–20 | visible |
| CCD_000052 | 식별 | INSIDE | before_start 0–0 | visible |
| CCD_000540 | 식별 | OUTSIDE | during_clip 32–36 | visible |
| CCD_000643 | 식별 | UNKNOWN | unknown | visible |
| CCD_000022 | 식별 | UNKNOWN | unknown | visible |
| CCD_000493 | 식별 | UNKNOWN | unknown | absent |

INSIDE 1건, OUTSIDE 2건, UNKNOWN 9건이다. UNKNOWN은 정답 부재가 아니라 현재 시각 근거의 한계다. f0에서 상대를 식별하지 못하거나 눈·무차선 노면으로 경계/연장선을 확인하지 못한 사례를 추정으로 채우지 않았다. 사고 상대 식별과 실제 접촉면 관측은 별개이며, 일부 사례는 근접·연속 운동은 보이나 접촉면이 화면 밖이다.

before_start의 0/0은 첫 프레임 제출 인코딩이다. 물리적 진입이 f0에 발생했다는 뜻이 아니며 시작 전 정확한 시각은 미상이다. during_clip 범위는 마지막 확실한 사건 전/첫 확실한 사건 후를 포함하는 보수적 구간으로 단일 프레임 정답이 아니다.

표식 좌표는 f0 원본에서 각각 직접 정했다. 001237·000540·000643의 상대 상부는 작아 축소 시 표식 가시성이 제한된다. 000052는 바로 앞 녹색 세단이 아닌 그 한 대 앞 차량, 001237은 왼쪽 앞 별도 흰 탑차가 아닌 우측 출입구 탑차다. 원본은 000103만 960×720, 나머지는 1280×720이다.

이 문서는 독립 AI 시각 참조이며 인간 전문가 검수나 유일한 확정 정답을 대신하지 않는다. 작성 완료 후 review_a.json/md 및 원본 파일 수정을 중단한다.

