# Spatial grounding final preflight

**PASS before both new model calls.**

Two existing source-detail image inputs are unchanged. Questions/jobs agree on full1152×768 pixel coordinates, painted-marking centerline and768-token cap. Independent A/B hashes are preserved; no coordinate averaging or forced historical INSIDE label.

Strict JSON/schema and limited IoU/segment/no-extrapolation sanity checks pass. The independent verifier is ready for two actual calls and CPU tensor reconstruction. Reused runner bindings/environment and all prior1,866 frozen files pass. No workers/models were invoked.

Coverage: bbox A/B both frames; visible wheels f31 A2/B1 and f39 A2/B2; f31 boundary A4/B5 anchors, f39 boundary unavailable. **Inside-point side scoring is unavailable for every expert/frame pair:** f31 reference inside-point y580 is beyond boundary coverage; f39 has no boundary. No unsupported extrapolation is introduced.

Detailed raw outputs, geometry numbers and interpretation remain to be verified after the two fixed calls. Prior state-only valid_enum is separate from full spatial JSON validity.

## Limits

- Readiness only; no diagnostic outputs observed or new model calls made.
- Manual expert tolerances are diagnostic choices, not calibrated confidence or human/official GT.
- Inside-point side metric is unavailable in all four expert/frame comparisons: f31 reference point y580 lies beyond each supplied boundary range; f39 lacks boundary anchors.
- New structured question, centerline wording and768-token cap differ from previous state classification; no original-Q3 root cause or temporal score claim.
- Independent execution verifier checks input/runtime/basic schema; detailed geometry numbers receive a separate read-only review after outputs exist.
