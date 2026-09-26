# Equal-size source-detail independent verification

**PASS.** No new model load or inference by the CPU verifier.

Frozen files pre/post: 1866; prior1,829 preserved. Six actual calls and six CPU processor reconstructions match; only pixel_values differ between each equal-size pair.

| Frame | AI reference | Upsampled | Source detail | Historical control |
|---|---|---|---|---|
| 31 | OUTSIDE | OUTSIDE | OUTSIDE | True |
| 35 | UNKNOWN | OUTSIDE | OUTSIDE | True |
| 39 | INSIDE | OUTSIDE | OUTSIDE | True |

Metrics: {'upsampled': {'n': 2, 'correct': 1, 'accuracy': 0.5, 'distinguish_definite_outside_inside': False, 'chronological_responses': [{'frame': 31, 'prediction': 'OUTSIDE'}, {'frame': 35, 'prediction': 'OUTSIDE'}, {'frame': 39, 'prediction': 'OUTSIDE'}], 'uncertain': 0, 'invalid': 0}, 'source_detail': {'n': 2, 'correct': 1, 'accuracy': 0.5, 'distinguish_definite_outside_inside': False, 'chronological_responses': [{'frame': 31, 'prediction': 'OUTSIDE'}, {'frame': 35, 'prediction': 'OUTSIDE'}, {'frame': 39, 'prediction': 'OUTSIDE'}], 'uncertain': 0, 'invalid': 0}}
Source-detail comparison: {'gained_frames': [], 'lost_frames': [], 'accuracy_delta': 0.0}
Diagnostic gate: {'gains': 0, 'losses': 0, 'historical_control_match': True, 'equal_text_grid_tokens': True, 'pass': False}

## Limits

- One exposed incident with two definite AI references and privileged counterpart marks; no independent generalization or official GT claim.
- f35 stays UNKNOWN and unscored; no temporal entry output is produced or changed.
- Source detail, blur and aliasing change together, while canvas, marker, header, text/grid/tokens are held equal.
- Source scenes are90percent native width, not untouched native resolution.
- Separate state replies do not reveal the internal cause of the old Q3 time-selection error.
- CPU replay confirms recorded preprocessing and scoring, not model reasoning or CUDA equivalence.
