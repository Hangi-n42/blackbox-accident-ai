# CCD AI peer adjudication

두 독립 A/B 원본을 동결 보존한 뒤 B가 A와 대조한 **AI peer adjudication**이다. 독립 제3검수나 인간 전문가 검수가 아니다. 이번 단계에서 native 21장을 다시 관찰했고 B의 기존 600프레임 시각 관찰을 재사용했다. 모델 예측·공급자 시점은 계속 미열람이다.

| ID | Contact frame | Entry frame | Direction | Space |
|---|---|---|---|---|
| CCD_000174 | 32-39 | unknown | RIGHT | 1 |
| CCD_000688 | 31-40 | 0 | None | 1 |
| CCD_000801 | 15-38 | 0 | None | 0 |
| CCD_001024 | 32-38 | unknown | LEFT | 0 |
| CCD_000453 | unknown | 8-12 | RIGHT | None |
| CCD_001477 | 31-33 | unknown | LEFT | 0 |
| CCD_000703 | unknown | unknown | None | None |
| CCD_000139 | unknown | unknown | None | None |
| CCD_000196 | 32-38 | 0 | None | 1 |
| CCD_000303 | unknown | unknown | LEFT | None |
| CCD_000815 | 35-38 | unknown | LEFT | 0 |
| CCD_000728 | 29-32 | 26-30 | LEFT | 1 |

평가 가능 분모: 접촉 8/12(새 변형/파편 4, 공급자+시각 연속성 보강 4), 진입 5/12(f0 인코딩 3, 구간 2), 방향 7/12, 공간 8/12. 네 필드 모두 가능한 사례는 1/12이며 공식 S2는 null이다.

174 entry=0은 같은 도로 쪽을 같은 차로로 확대 해석하여 철회했다. 453/303은 전방 화면 소실을 확실 사건후 상한으로 인증할 수 없어 contact unknown을 유지했다. 303 공간도 접촉 시점 연결 불충분으로 미상이다. 703/139는 최초 상대 미상으로 네 필드 모두 제외했다. 지원되는 경계 차이는 hull을 유지했다. 801의 2.3초 구간을 정밀 점라벨처럼 쓰면 안 된다.

입력의 12개×50프레임 ordinal/PTS 표를 재검증했다. 모두 PTS=frame/10, 0.0–4.9초다. 진입 f0은 실제 물리 시각이 아닌 시작시 동일차로 인코딩이며 실제 진입은 영상 이전 미상이다.

조정 완료·동결. A/B 원본 및 최종 조정 파일의 추가 변경을 중단한다.
