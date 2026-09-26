# Independent visual review B

AI 시각 검수이며 사람 정답 검증이 아니다. 차로 경계/상대/차체 내부를 확정할 근거가 없으면 UNKNOWN. before_start의 0/0은 제출 인코딩이고 물리적 진입 시각은 영상 전 미상.

All 12 clips: 48 full sheets (600 frames), all native f0 images, and the native images listed in JSON were directly viewed. No model predictions, provider labels or peer review were read. Native PNG dimensions and all 600 frame/PTS mappings were checked.

| ID | Counterpart fixed | f0 state | Entry | f0 marker |
|---|---|---|---|---|
| CCD_001498 | False | UNKNOWN | unknown UNKNOWN | unknown |
| CCD_000585 | False | UNKNOWN | unknown UNKNOWN | unknown |
| CCD_001278 | True | UNKNOWN | unknown UNKNOWN | unknown |
| CCD_001454 | True | UNKNOWN | unknown UNKNOWN | absent |
| CCD_000103 | True | UNKNOWN | unknown UNKNOWN | unknown |
| CCD_000470 | True | UNKNOWN | unknown UNKNOWN | absent |
| CCD_001237 | True | OUTSIDE | during_clip 8..21 | visible |
| CCD_000052 | True | INSIDE | before_start 0..0 | visible |
| CCD_000540 | True | OUTSIDE | during_clip 34..37 | visible |
| CCD_000643 | True | UNKNOWN | unknown UNKNOWN | visible |
| CCD_000022 | True | UNKNOWN | unknown UNKNOWN | visible |
| CCD_000493 | True | UNKNOWN | unknown UNKNOWN | unknown |

Native CCD_000103 is 960x720; other cases are 1280x720. Marker rectangles use source coordinates, not sheet coordinates. Small markers may lose visibility after reduction.

Frozen before model predictions. No further modification without an explicit new review step.
