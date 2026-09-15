# Stage2 감사 입력 계약 v1

이 도구는 점수 개선 실험이나 제출 코드가 아니다. 확인된 정답과 기존 추론 기록을 연결한다. 누락된 정답을 AI 관찰로 채우지 않는다. 모든 경로는 `--root` 기준 상대 경로 또는 절대 경로이며 SHA-256은 실제 파일 내용과 대조한다.

최상위 GT 문서는 `schema_version: 1`, `cohort_id: string`, `videos: array`이다. 각 영상은 다음 필드를 가진다.

| 필드 | 타입 / 의미 |
|---|---|
| ID | 비어 있지 않은 고유 문자열 |
| source_group_id | 동일 원천 영상·파생본을 묶는 비어 있지 않은 문자열 |
| source | `{dataset: string, source_uri: string, video_path: string, video_sha256: string}`. 영상 파일 해시를 실제 검증한다. URL은 출처이며 접속하지 않는다. |
| split | `development`, `validation`, `test` 중 하나. 공식 숨은 평가 접근 권한을 뜻하지 않는다. |
| exposure | `{predictions_seen: boolean, prior_model_development: boolean, annotation_blinded: boolean, notes: string}`. validation/test에 예측 노출 또는 개발 사용이 있거나 비맹검이면 거부한다. 공개 반복 실험은 development이다. |
| input_manifest_sha256 | 해당 영상의 추론 입력 manifest 해시. trace와 정확히 일치해야 한다. |
| frame_pts | 시간 순서의 `[{frame: integer >= 0, pts_seconds: finite number >= 0}, ...]`. 번호는 중복 불가, PTS는 엄격 증가. 파일 번호를 재번호화하거나 FPS로 환산하지 않는다. |
| mapping_provenance | `{artifact_path, sha256, method, time_origin}`. method/time_origin은 비어 있지 않은 문자열. 증거 JSON은 `{frame_pts: [...], source_video_sha256: string, time_origin: string}`을 포함해야 하며 inline 값·원본 영상 해시·시간 기준과 정확히 대조한다. 추가 필드는 허용한다. |
| target_vehicle_id | 영상 내 실제 충돌 상대를 구분하는 비어 있지 않은 문자열. 다른 영상과 공유하는 ID가 아니다. 공식 주석에 차량 ID가 없으면 official-only 정답에서 null을 허용한다. 인간 확정 정답은 실제 상대 ID를 요구한다. |
| labels | 아래 네 키를 반드시 포함한다. 미확인은 null. |

labels 키는 `collision_time_seconds`, `entry_time_seconds`, `entry_side`, `evasion_space`이다. 값은 null 또는 `{value, gt_type, provenance}`이다. 시간 value는 해당 frame_pts와 같은 time_origin의 유한 비음수 초, 방향은 `LEFT`/`RIGHT`, 공간은 정수 `0`/`1`이다. bool을 정수로 받지 않는다. 구간형 불확실 주석은 확정 시각으로 평가하지 않는다. 진입 전부터 차로 안인 경우에는 사람이 판정한 최초 프레임의 실제 PTS를 정답으로 기록한다. null은 오답/0/-1로 바꾸지 않고 해당 항목을 미평가한다. 하나라도 null이면 전체 S2는 null이다.

gt_type은 `official` 또는 `human_adjudicated`만 허용한다. AI/advisory/단일 인간 검수 초안은 거부한다. provenance 공통 필드는 `{artifact_path, sha256, source_ref, created_at, annotators, adjudicator}`이다. `created_at`은 timezone을 포함한 ISO8601, annotators는 문자열 배열, adjudicator는 문자열 또는 null이다.

- official: source_ref는 공식 주석의 정확한 출처·행·필드를 식별한다. annotators=[]와 adjudicator=null 허용.
- human_adjudicated: 서로 다른 최소 2명 annotators, 비어 있지 않은 adjudicator, `independent_reviews: [{annotator, artifact_path, sha256}, ...]` 및 `adjudication: {artifact_path, sha256, resolution}` 필수. 서로 다른 최소 2개 검수 증거 파일을 요구한다. resolution은 합의/불일치 해결의 구체 기록이다. 검수 증거 해시가 있다고 인간 신원이나 실제 독립성을 자동 인증하는 것은 아니다. 운영 담당자가 실제 검수 과정과 출처를 확인해야 한다.

기존 public 로그의 trace 문서는 `schema_version:1`, `scope`, `provenance`, `videos`이다. 각 영상은 ID/input_manifest_sha256/original_frame_numbers/valid_frame_numbers/prediction/diagnostics/calls를 보존한다. V5 제출에는 상세 로그가 없으므로 과거 V3의 동등 코드/동등 모델 로그는 **개발용 archived trace**이며 실제 V5 서버 실행 기록으로 부르지 않는다. 모델이나 과거 예측을 새로 실행하지 않는다.

평가는 고정된 전체 범주(방향 LEFT/RIGHT, 공간 0/1)의 Macro-F1과 공식 ±0.3초만 사용한다. 임의 채택 문턱은 없다. 정답이 모두 있는 동일 cohort에서만 전체 S2를 계산한다. 부분 정답 점수는 분모와 함께 개발 진단으로만 반환한다. 동일 원천의 파생본은 독립 표본 수로 세지 않는다.
