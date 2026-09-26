# Reserved baseline comparison — frozen before predictions

Compare production `model/stage3/motion_model.joblib` acceleration head against the full comma-only 2,395-row pipeline `artifacts/stage3_zod_20260920/models/comma_only.joblib`. Do not select a fold checkpoint. Production includes 2,761 historical comma rows and 50 public labels; research includes 2,395 revised comma rows and no public or ZOD labels. This estimates relative utility, not the causal effect of any single training change.

Use the 18 preselected centers from `reserved_windows.json` once: six accelerating, six decelerating, six constant; nine per route. Adjacent frames are context for the existing DIS864 extraction, not additional evaluation observations. Reuse existing alignment, sensor labels, integrity and decoding reports. The sensor labels are proxy truth, with unverified hardware latency. Two new routes use vehicles already represented in development; geographic independence is unverified. Prior sensor selection and 54-frame AI visual review are disclosed. Project records show no prior model predictions or training on these windows.

Both checkpoints receive identical extracted inputs and truth. Preserve each saved scaler, all 864 existing features, and four-class argmax. STOP predictions count as errors; never renormalize over the three observed truth classes. Evaluate precision, recall and F1 per A/D/C, Macro-F1 over exactly these three classes, correct counts, route metrics and all 18 individual errors. No steering truth or official S3 is supplied or computed.

Compare expanded predictions to production: genuine wrong-to-correct recoveries, correct-to-wrong regressions, newly introduced A/D inversions, existing inversions corrected to truth, and inversions changed to constant-but-still-wrong. The latter are not recoveries. Report load, shared feature extraction and prediction time, and failures separately.

## Decision rule

- Keep expanded as a follow-up validation target only if pooled Macro-F1 and correct count both strictly improve, neither route Macro-F1 decreases, and there are zero new A/D inversions.
- Otherwise retain production as the next improvement baseline if its pooled Macro-F1 and correct count are at least as high and its Macro-F1 is at least as high on both routes.
- Otherwise report that this sample cannot determine superiority.

These are conservative rules for prioritizing further work, not statistical proof of general superiority or adoption approval. No outcome-dependent threshold, label, model or feature changes; no further experiment or tuning on this comparison. The 18 windows become development-exposed after predictions. Preserve original models/results and add a superseding exposure record. Horizontal flip remains discarded. No fitting, downloads, production replacement or submission.
