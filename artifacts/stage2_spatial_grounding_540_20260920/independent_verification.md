# Spatial diagnostic independent runtime verification

**PASS.** Two actual calls, frozen files and CPU processor reconstructions verified without model inference.

Frozen files pre/post: 1896; prior1,866 preserved. Existing source_detail RGB/pixel_values/image_grid preserved; new question and768-token cap retained.

| Frame | JSON | Spatial schema | Worker lane enum | Generated tokens | Token cap reached |
|---|---|---|---|---:|---|
| 31 | True | True | UNCERTAIN | 58 | False |
| 39 | True | True | INSIDE | 187 | False |

Runtime/input/strict basic spatial schema audit. Geometry reference scores require a separate read-only mathematical review.

## Limits

- One exposed incident, two frames and expert-agent references are not human GT or performance validation.
- New structured question and768-token cap jointly differ from the old state question.
- A valid lane_state alone does not imply a valid or accurate spatial response.
- At-token-limit and configured-EOS flags are observations; JSON/schema completeness is evaluated separately.
- No forced old INSIDE geometry, automatic coordinate repair, entry override or officialS2.
