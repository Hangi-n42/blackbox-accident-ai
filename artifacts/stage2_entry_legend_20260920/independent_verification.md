# Legend / conditional state independent verification

**PASS.** CPU-only stored-record verification; zero new model calls.

Frozen files before/after: 909; previous freeze preserved: 865; source PNG/native PTS: 250.
Ten fresh Q3 workers/calls; 30 cached Q1/Q2/Q4 answers replayed; other three final outputs unchanged in all five pairs.
Marked images/pixel tensors/image grids are identical across arms. Exact prefix sentence alone changes the prompt; input_ids/attention hashes change.
Fresh marker-only raw responses and frame predictions reproduce all five historical controls.

| Case | Control | Legend | Control grade | Legend grade |
|---|---:|---:|---|---|
| CCD_000688 | 13 | 4 | wrong | wrong |
| CCD_000801 | 10 | 10 | wrong | wrong |
| CCD_000453 | 4 | 4 | wrong | wrong |
| CCD_000196 | 10 | 10 | wrong | wrong |
| CCD_000728 | 14 | 14 | wrong | wrong |

Shared-truth accuracy delta: [0.0, 0.0]. Shared-truth MAE delta seconds: [-0.18, -0.18].
Gate: {'gains': 0, 'losses': 0, 'new_false_first': 0, 'paired_MAE_nonincrease': True, 'control_matches_history': True, 'pass': False}.
Grades use exact rational seconds and closed acceptance intervals; paired MAE extrema were independently calculated at interval endpoints.

Conditional probe: 2/5 eligible AI state references; uncertain=0, invalid=0.
All five state workers/calls completed after the valid failed main comparison. Each input is the unchanged 384x256 first-tile crop; strict JSON enum parsing has no fallback.
State response success does not establish first-wheel-contact timing, causal failure attribution, or an automatic frame-zero correction.

## Limits

- Exposed five-case expert-marker development diagnostic, not independent generalization.
- AI state/timing references and marker identity correctness are not certified by this byte/math review.
- State probe is a cropped single-frame task, not the same-input temporal ablation or an entry override.
- Recorded processor hashes were checked; tensors were not independently regenerated.
- Python socket checks do not establish OS-wide network isolation; Mac does not establish CUDA equivalence.
- No official S2 was computed.

No production, frozen source, reference, model, or parent evaluation file was modified. Measurements and artifact hashes are in independent_verification.json.
