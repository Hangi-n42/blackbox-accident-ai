# Peer adjudication by reviewer B

A/B independent records preserved. This is peer adjudication after both reviews, not a third independent expert. No model predictions or provider labels read.

| ID | Same counterpart | f0 | Entry | Native marker |
|---|---|---|---|---|
| CCD_001498 | False | UNKNOWN | unknown UNKNOWN | False |
| CCD_000585 | False | UNKNOWN | unknown UNKNOWN | False |
| CCD_001278 | True | UNKNOWN | unknown UNKNOWN | False |
| CCD_001454 | True | UNKNOWN | unknown UNKNOWN | False |
| CCD_000103 | True | UNKNOWN | unknown UNKNOWN | False |
| CCD_000470 | True | UNKNOWN | unknown UNKNOWN | False |
| CCD_001237 | True | OUTSIDE | during_clip 8..21 | True |
| CCD_000052 | True | INSIDE | before_start 0..0 | True |
| CCD_000540 | True | OUTSIDE | during_clip 32..37 | True |
| CCD_000643 | True | UNKNOWN | unknown UNKNOWN | True |
| CCD_000022 | True | UNKNOWN | unknown UNKNOWN | True |
| CCD_000493 | True | UNKNOWN | unknown UNKNOWN | False |

Denominators: counterpart 10/12; known f0 state 3/12 (INSIDE 1, OUTSIDE 2); entry 3/12 (before_start 1, during_clip 2); native marker candidates 5/12. All five rendered-input approvals remain pending.

CCD_001278 remains unknown because only A supplied a bounded entry. CCD_001237 uses f8..21 and CCD_000540 f32..37, preserving the conservative union rather than narrowing or averaging. CCD_000052 0/0 is submission encoding; physical pre-clip entry time is unknown.

Selected native rectangles are from A and directly checked against f0 originals. They avoid wheels, lane boundaries and number overlays. Small distant targets remain a visibility limitation after reduction. Model calls require the separate rendered-input QA.

All original review JSON/MD hashes were verified unchanged. AI references are provisional; unknown cases are excluded from the corresponding evaluation denominator.
