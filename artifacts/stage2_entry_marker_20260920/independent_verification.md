# Q3 same-target marker independent verification

**Runtime, input-contract and scoring verification: PASS.** No new model calls were made by this verifier.

Frozen files: 865; fresh workers/calls: 10/10; cached answers replayed: 30; source PNG: 250; candidate tiles: 60.
Five original sheets reproduce the historical Q3 RGB exactly. Five marked sheets reproduce only the fixed yellow rectangle pixels. All10 saved bounded images match actual calls.
Within each arm pair, recorded processor hashes differ only for pixel_values. Prompts, input_ids, attention and image-grid tensors remain unchanged where present. Tensor contents were not regenerated.
Raw text, original JSON parser, offered-frame membership, snap/fallback flags, full four-call stored-answer replay and the other three final fields were independently checked.

| Case | Control entry | Marked entry | Control grade | Marked grade | Valid marked selection |
|---|---:|---:|---|---|---|
| CCD_000688 | 13 | 13 | wrong | wrong | True |
| CCD_000801 | 13 | 10 | wrong | wrong | True |
| CCD_000453 | 4 | 4 | wrong | wrong | True |
| CCD_000196 | 10 | 10 | wrong | wrong | True |
| CCD_000728 | 14 | 14 | wrong | wrong | True |

Shared-truth accuracy delta bounds: [0.0, 0.0]. Shared-truth mean absolute-error delta bounds (seconds): [-0.06, -0.06].
Frozen diagnostic gate: {'gains': 0, 'losses': 0, 'no_new_false_first': True, 'paired_MAE_nonincrease': True, 'baseline_matches_history': True, 'pass': False}. Historical control raw/prediction matches: 5/5 and 5/5.
Interval grades use exact rational seconds; paired accuracy uses closed acceptance-interval differences. Paired absolute-error extrema were computed from interval endpoints independently of the experiment evaluator.

## Scope limits

- Exposed five-case privileged-marker development diagnostic, not independent generalization or deployable policy.
- Identity and attention effects are combined; prediction-only outputs do not identify which vehicle the model tracks.
- Semantic validity of expert boxes and interval labels is not certified by this byte/math review.
- Calls reused Q1/Q2/Q4 context; ten fresh Q3 calls are not ten complete four-call model runs.
- Python socket monitoring and stored processor hashes are narrower than full OS/network or independent processor regeneration.
- Mac MLX verification does not prove CUDA/NF4 equivalence; official S2 remains null.

The source code, frozen inputs/reviews, model files and parent evaluation were not modified. Full hashes, case scores and resource measurements are in independent_verification.json.
