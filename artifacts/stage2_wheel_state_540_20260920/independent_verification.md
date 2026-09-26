# Wheel-state diagnostic independent verification

**PASS.** No model was loaded or called by this CPU verifier.

Frozen files pre/post: 1829; previous1793 preserved. Actual calls: 6. Source frames/native PTS, approved A markers and both input rasters were reconstructed; every recorded processor hash matches.

| Frame | AI reference | Low | Upsampled |
|---|---|---|---|
| 31 | OUTSIDE | OUTSIDE | OUTSIDE |
| 35 | UNKNOWN | OUTSIDE | OUTSIDE |
| 39 | INSIDE | OUTSIDE | OUTSIDE |

Metrics: {'low': {'n': 2, 'correct': 1, 'accuracy': 0.5, 'confusion': {'reference_rows': ['OUTSIDE', 'INSIDE'], 'prediction_columns': ['OUTSIDE', 'INSIDE', 'UNCERTAIN', None], 'matrix': [[1, 0, 0, 0], [1, 0, 0, 0]]}, 'distinguish_definite_outside_inside': False, 'chronological_responses': [{'frame': 31, 'prediction': 'OUTSIDE'}, {'frame': 35, 'prediction': 'OUTSIDE'}, {'frame': 39, 'prediction': 'OUTSIDE'}], 'inside_to_outside_adjacent_pairs': [], 'uncertain_all_queried': 0, 'invalid_all_queried': 0}, 'upsampled': {'n': 2, 'correct': 1, 'accuracy': 0.5, 'confusion': {'reference_rows': ['OUTSIDE', 'INSIDE'], 'prediction_columns': ['OUTSIDE', 'INSIDE', 'UNCERTAIN', None], 'matrix': [[1, 0, 0, 0], [1, 0, 0, 0]]}, 'distinguish_definite_outside_inside': False, 'chronological_responses': [{'frame': 31, 'prediction': 'OUTSIDE'}, {'frame': 35, 'prediction': 'OUTSIDE'}, {'frame': 39, 'prediction': 'OUTSIDE'}], 'inside_to_outside_adjacent_pairs': [], 'uncertain_all_queried': 0, 'invalid_all_queried': 0}}

Enlargement: {'gained_frames': [], 'lost_frames': [], 'accuracy_delta': 0.0}

## Limits

- One previously exposed incident and two definite AI state references; not independent performance validation.
- f35 remains UNKNOWN and unscored regardless of its output; no point state is derived from the old entry interval.
- Nearest enlargement jointly changes raster size and image tokens, without adding native source detail.
- Privileged counterpart marks and AI references are not human or official ground truth.
- Chronological response reversals are descriptive only; labels cannot identify an internal model cause.
- No entry timing output changes or official S2; Mac does not establish CUDA equivalence.
