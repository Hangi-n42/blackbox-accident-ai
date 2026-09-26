# Equal-size source-detail final preflight

**PASS before all six new diagnostic calls.**

Existing refs are byte-identical: f31 OUTSIDE, f35 UNKNOWN, f39 INSIDE. Original A/B hashes are resolved against the previous experiment, without reannotation. Eligible denominator is2 and both f35 replies remain unscored.

All historical control PNG bytes match. Both arms use1152×768, the same source frame/framing/prompt and exact upper-body marker. Independent reconstruction confirms original1280×720 RGB resized1152×648 BICUBIC into[0,84,1152,732], followed by the exact historical mask. Only unmasked scene pixels change; header, padding and marker pixels remain equal. Peer B directly viewed all six inputs and three sources and reportsPASS.

Changed pixels for f31/35/39:461890/461905/467032. No source crop or full-native-resolution claim is made. Jobs contain no reference answer or interval.

The reused runner/check_frozen share the correctly rebound HERE/ARMS globals; the actual environment expression was evaluated on CPU without workers. All previous1,829 frozen files remain unchanged. Six independent calls are planned; postrun verification requires strict parsing, control historical raw/enum reproduction, equal text/grid/tokens with only pixel_values changed, and CPU tensor reconstruction before categorical scoring/gate validation.

The old pending field in copied refs is historical metadata; separate peer PASS approves the new inputs without changing those refs. verify.py is complete and syntax/input audited. No models were loaded or called.

## Limits

- One exposed incident and two definite AI states with privileged expert counterpart marks, not generalization or official S2.
- Source_detail retains90percent native width and jointly changes detail/blur/aliasing; cannot isolate one visual mechanism.
- Pixel-level QA does not establish model target understanding or runtime processor equivalence; six recorded tensor hashes will be independently reconstructed after inference.
- Readiness is not success; no new predictions have been observed by this preflight.
