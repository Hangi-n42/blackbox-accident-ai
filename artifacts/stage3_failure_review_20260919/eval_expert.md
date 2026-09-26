# Stage3 evaluation expert review — 2026-09-19

Read-only review. No training, model, feature, label, source, or existing result was changed. Derived numeric evidence is in `/tmp/stage3_eval_public_confusions_20260919.json`.

## Verified findings

1. All tested reweighting conditions, including the capped control, have EXACTLY the same public confusion rows for ACCELERATING (support 9), DECELERATING (8), and STOPPED (3): `[5,2,2,0]`, `[0,6,2,0]`, `[0,0,0,3]`. Only the CONSTANT row (support 30) changes. Baseline `[1,3,26,0]`; vehicle `[1,5,24,0]`; vehicle-route `[4,4,22,0]`; half `[2,3,25,0]`; capped `[3,4,23,0]`. Thus these interventions did not repair a single net public acceleration/deceleration confusion-row count. Row equality does not alone prove every per-case prediction is identical, so the claim is expressly about confusion rows.
2. The selected vehicle policy causes two additional CONSTANT-to-DECELERATING errors and no public correction relative to the expanded baseline: OPEN_001 sample60 and OPEN_005 sample420. Accel macro-F1 .791228 -> .766420; S3 .808951 -> .791586. The public sample consists of only 5 videos / 50 sparse points and is repeatedly development exposed. This is observed development regression, not evidence of population-wide inferiority or statistical significance.
3. The inner selection objective is mean vehicle/fold macro-F1 on strict sensor proxies, not target official-label OOF S3. OPEN_001's chosen vehicle policy had inner mean gain +.0084243187 but final public S3 loss -.017365. This directly demonstrates failure of that selection criterion to transfer on this development set; it does not identify the unique physical or labeling cause.
4. Some inner vehicle/fold groups cannot validate reversal risk: OPEN_001 inner_3 RAV4 has 633 CONSTANT truths and zero acceleration/deceleration truths. `metrics()` returns opposite_rate=0 when denominator=0, and `select()` averages those rates. All-absent classes also contribute F1=0 under fixed 4-class macro. This is not leakage or an arithmetic error, but the mean mixes groups with radically different class coverage and treats unobserved reversal exposure as zero. Such groups are not evidence that a policy is safe against reversals.
5. The vehicle-route weight maximum49.375 and Kish effective-N776.34 vs2435 indicate concentration, not independent-sample size. Capping to4 improves concentration but does not rescue public or all dates: concentration alone is not a sufficient explanation.
6. No leakage or baseline reproduction error found in inspected `run.py`: outer/inner route exclusion, scaler unweighted and equal, per-class mass and public weights invariant, policy saved before outer scoring, baseline probability equality asserted. Coverage of this review is code/read results, not an independent rerun of all training.

## What the failure does and does not prove

- Proven: particular policies change source loss allocation; inner gains do not transfer consistently; all tested policies lose public score; some reduce one date's reversals while worsening another.
- Not proven: all weighting is useless, calibration will succeed, visual representation is the unique cause, sensor proxies are wrong at all failed times, or V-JEPA/depth models cannot work.
- The no-worse-on-every-date/vehicle promotion gate is more demanding than the official S3 objective. However, rejection here is not merely a stringent gate: every tested policy already loses public S3, and none repairs public reversal counts. Do not relax the gate after seeing these outputs to manufacture success.
- Date folds overlap (not 3 independent datasets); contiguous 10Hz errors are not independent cases. Use route/video-block effect summaries and event runs; never frame-count binomial confidence claims. A paired test with only the two public changed points would itself be extremely weak, and developer exposure invalidates a fresh-confirmatory interpretation.

## Best next limited experiment: target-label calibration before another large encoder change

This is a cost/information recommendation, NOT a proven optimum or guaranteed gain. The directly observed selection mismatch justifies one tightly bounded diagnostic improvement attempt.

Freeze DIS864, expanded2395 training rows, linear recipe, labels, dates, and steering. For each outer public held video, use only the other four videos'40 official labels. Generate predictions for those40 using inner leave-one-video-out models (the respective inner video is absent from fitting; external23 retained). Fit a strongly pre-regularized low-dimensional calibrator on those inner-OOF logits, anchored at identity: shared positive temperature plus zero-sum class offsets (max4 free degrees). Fix the fitting recipe before any new outer scoring. Do not use in-sample logits from a classifier already fit to those40 labels. Refit base on all40, then apply calibrator to outer held10. No held/public threshold sweep and no weighting sweep.

Compare raw and calibrated predictions with exactly the same outer models; this isolates target-label decision calibration. Primary public official-label accel macro-F1/S3, per-class confusion, both directional reversals, newly wrong previously-correct points. External strict proxy results remain secondary diagnostics: a public-trained calibrated model must not be called an external holdout when its training base includes that external video. For date diagnostics, train the same base on only the date training subset and generate calibration OOF official logits using that subset, excluding outer date cases throughout.

Stop after one frozen recipe if no public gain, new target reversals, unstable sign across held videos, or systematic date/vehicle degradation. No broad tuning on50. All claims remain development-only. If it fails, the next branch is a single physically motivated visual feature change with current classifier, not more proxy-based candidate selection or arbitrary encoder replacement.

### Counterarguments / limits

- Only40 official calibration labels per fold, including very few STOPPED cases, create high variance. Regularization does not create missing data. Nested OOF stops direct train/test leakage but does not undo years/days of developer exposure.
- A global logit calibration cannot restore missing visual evidence or distinguish scene-dependent opposite-motion confounds. It may merely convert severe errors into CONSTANT; count genuine recovered correct predictions separately from reversals suppressed by abstention-like CONSTANT predictions.
- Temporal heads, V-JEPA pooled features, sensor auxiliary regression, and internal early stopping already failed in tested settings. A new large visual model cannot be prioritized simply because its literature results are strong. A geometry-informed representation would require a concrete mechanism, a bounded corresponding control, and no simultaneous classifier change.
- If a specialist identifies strong feature-level evidence of a specific missing invariant now, that evidence can justify moving the visual-feature experiment ahead of calibration. Current reweighting experiment alone does not supply that proof.

## Evidence paths

- artifacts/stage3_group_weighting_20260919/REPORT.md
- artifacts/stage3_group_weighting_20260919/run.py (`weighting`, `inner_splits`, `select`, `main`)
- artifacts/stage3_group_weighting_20260919/inner_metrics.csv
- artifacts/stage3_group_weighting_20260919/selections.json
- artifacts/stage3_group_weighting_20260919/metrics.csv
- artifacts/stage3_group_weighting_20260919/capped_control/metrics.csv
- artifacts/stage3_state_learning_20260919/run.py (`metrics`)
- artifacts/stage3_state_learning_20260919/REPORT.md
- artifacts/data_curation_20260917/comma/rules_and_review.md
