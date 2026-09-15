# Nexar 검수 추가 파일 4개 무결성 확인

2026-09-15. Downloads에서 지정받은 신규 00004/00005/00006 및 갱신 00003 JSON을 `research/v6_stage2/user_reviews/`에 **원본 바이트 그대로** 보존했다. 새 JSON 보고서에 다운로드·사본·case·원본 영상 SHA와 모든 검사값을 기록했다. 기존 `nexar_review_intake_integrity.json`은 변경하지 않았고 검사 전후 SHA가 동일하다.

네 자료 모두 원본 영상 SHA, Node의 `JSON.stringify(case.frames)` 전체 배열 UTF-8 SHA, 전체 디코드 프레임의 native PTS 대응이 일치했다. 검수자가 지정한 서로 다른 PNG 6장도 파일 SHA 및 원본 디코드 RGB 픽셀이 일치했다. 모든 PNG를 비교한 것은 아니며, 전체 프레임 시간축과 선택 PNG를 구분했다. native PTS는 FPS로 추정하지 않고 PyAV의 정수 PTS×유리수 time_base를 사용했다. CPU 2 codec threads로 처리했으며 모델 추론·GPU·예측 파일 접근은 없었다.

| 자료 | 원본 프레임 수 | 접촉 초안과 대응 시각 | 진입 초안과 대응 시각 | 유지한 불확실성 |
|---|---:|---|---|---|
| 00004 | 1202 | frame579 / 19.3초 | uncertain, frame/PTS null | 진입 시점 미확정 |
| 00005 | 1227 | frame643 / 21.013071895424837초 | 같은 frame643 / 같은 시각 | 사람이 두 사건을 같은 프레임으로 골랐다는 사실만 보존; 동시 발생 진위 미판정 |
| 00006 | 1199 | frame579 / 19.3초 | frame0 / 0초 | side 빈값, 이미 같은 차로였다는 초안 유지 |
| 00003 갱신 | 1203 | frame582 / 19.4초 | frame0 / 0초 | side 빈값, 신규 설명 보존 |

프레임 번호는 모두 원본 순차 디코드의 0-based 번호다. 00004의 미확정 진입을 임의 프레임으로 대체하지 않았고 00006/00003의 방향도 추정하지 않았다.

00003 구버전과 새 버전의 전체 JSON 필드 비교에서 달라진 것은 `created_at`과 `review.notes`뿐이다. 새 설명은 “계속 같은 차로에서 달렸기 때문에 진입한 쪽은 판정 불가”다. 접촉·진입 프레임/초/상태, side 빈값, space=1, already_entered_at_start=true 등 수치·범주 답변은 모두 같다. 따라서 이전 단일 초안 기준 시간 대조의 참조 수치가 변경되지는 않았다. 이전 intake·진단의 해시 결합은 덮어쓰지 않고 새 버전을 별도로 보존했다.

이 결과의 PASS는 파일·프레임 무결성 PASS다. 네 자료는 여전히 단일 사람 검수 초안이며, 사람 간 독립 재검수·합의가 제공되지 않았다. 기록상 source_group은 PENDING_SOURCE_AUDIT, split은 unassigned이고 exposure/annotation_blinded는 제출된 메타데이터다. 이를 검증된 독립 holdout이나 확정 정답으로 승격하지 않았다. 전체 S2는 null이며 reserved 00005/00006/00007의 예측은 접근하지 않았다.

상세 보고서: `nexar_review_intake_integrity_addendum.json`

보고서 SHA256: `486838bfcc551d54caf2889a61cf3dba178b1c1b2e3d66af2a23c35707cbe937`

기존 intake 보존 SHA256: `23c9a7d4413e184eebd8550dd8b2c1fb1462b1d6a4916b95963485f64b90f1fe`
