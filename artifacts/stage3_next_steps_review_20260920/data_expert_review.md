# Stage3 data/domain review — 2026-09-20

Read-only review. No model fitting, downloading, original edits or new annotation claims. Read `artifacts/stage3_tracker_foe_20260920/REPORT.md`, `stage3_point_motion_20260919/REPORT.md`, `stage3_official_bias_20260919/REPORT.md`, `stage3_failure_review_20260919/REPORT.md`, comma curation rules, and the earlier data expert's stored-prediction audit.

## What the latest experiment establishes

The 10 windows intentionally contain five previous failures and five successes. LK passed 5/10 and DIS passed 4/10; none of the five failures was recovered. This is a controlled mechanism diagnosis, not a representative test of all comma clips or a point improvement estimate. More surviving points are not more independent or correct observations.

Holding tracks fixed, small changes in estimated expansion center can change the feature sign. Holding starting-point membership fixed, DIS still breaks the previously valid expanded_19 deceleration window. Thus both trajectory estimates and the subsequent center-sensitive feature matter. There are no ground-truth image trajectories or camera poses, so their physical error contributions cannot be assigned percentages. The experiments do not rule out every learned tracker, every geometric representation, or every optical-flow method.

Quality-pass does not imply trustworthy supervision alignment: expanded_11 acceleration has sensor a/v +0.146363 but feature +0.007287 despite passing the earlier geometry checks. Five-pixel center perturbations do not fully explain that error. Current masks include some sky/reflection risks; sensor hardware latency remains unmeasured.

## Available data and limits for the next score experiment

- Existing 23 comma source segments provide 13,787 aligned 10 Hz timestamps; the current classifier's curated 2 Hz external subset is 2,395 rows. These are development-exposed segments, not 23 new independent evaluations.
- Existing public labels are five videos / 50 official rows. They have repeatedly informed model selection. A score difference is public development evidence, not private-score improvement.
- Point-feature diagnostic: 16 source clips, 28 windows, 11 matched-speed pairs. Train A5/D5/C7; validation A2/D4/C5. Quality-valid validation is A0/D2/C1. No validation pair has both members valid. Therefore the prerequisite to train the current feature was not met.
- There is no within-vehicle/partition complete A/D/C matched-speed triple with all speed differences <=2 m/s. RAV4 validation supplies no A/D satisfying the narrow criteria. Changing that threshold merely to fill counts after seeing results would not solve the gap.
- Each of the 28 windows has AI-reviewed/manual static rectangles. This is not the automatic masking input that a submission would have. A better feature formula on these masks alone is still not a complete deployable feature.
- Strong existing inversions are not all boundary labels: the capped-weight experiment's new reversals have |a| minima .631/1.338/.591 across dates, and the bias experiment's six new reversals have roughly -1.46 to -1.01 m/s². Confident sensor state does not imply large classifier margin.
- Source/speed confounding remains observed: training CONSTANT medians near29 m/s versus date1 Civic CONSTANT median15.48; this association does not prove a learned shortcut.

## Recommended next single-factor direction

Stop tuning the present single-center inverse-radius feature and stop swapping conventional trackers on the same ten windows. The new information needed is a camera-motion representation that does not require a manually chosen expansion center and can run on automatic input.

A conditional single-factor candidate is **within-window pose-derived relative translation change**, supplied by one frozen multi-view geometry estimator (e.g. the DA3 candidate already discussed in the project's research review), while retaining the current DIS classifier and labels. This data review does not validate that estimator's runtime/accuracy; the vision review must settle the implementation. Do not simultaneously add depth features, object masks, tracker replacements or a new head. First compare its sign/constant-speed residual and availability against the sensor-matched diagnostic windows. Depth may be an internal model output, but should not become an additional fitted feature in the pose-only arm.

Critical constraints: one common scale within each observation window; no differencing independently scaled windows; video-only inference (sensor speed cannot normalize the feature at inference); relative motion is not absolute m/s²; do not infer steering labels directly from yaw. Require valid negative controls at multiple speeds and observable baseline reversal cases, not merely good agreement on previously correct decelerations. Report missing pose confidence rather than assigning zero motion.

If this feature passes a pre-frozen gate, then a single paired score experiment is warranted: existing DIS864 baseline versus DIS864 + that feature group with identical curated rows, splits, scaler/classifier configuration, public leave-one-video-out and external date diagnostics. Preserve original evaluation rows: do not evaluate only the geometry-success subset. Missing-feature handling and any fallback must be fixed using training-only information, without modifying the existing baseline. Outcomes include class F1, public S3, reversal episodes and newly broken correct decisions. A feature that merely changes opposite errors into CONSTANT errors is not sufficient.

If the vision review cannot establish reliable scale consistency or automatic inputs, do not start another full feature extraction/classifier trial. A bounded diagnostic can reject this candidate; endless salvage of the current ten windows would amplify development selection bias.

## Minimum additional data, only where necessary

Existing assets suffice to reject or provisionally validate an implementation; a new bulk download is not necessary for that step. For a balanced functional check that current held windows cannot supply, minimally obtain two previously unused continuous routes with synchronized video/speed (and preferably actual longitudinal acceleration), each offering at least three nonoverlapping A, D and C windows in comparable speeds with visible static background: at least18 windows total. Include both vehicles if asserting cross-vehicle coverage. Do not select only easy daylight straight roads if eventual deployment contains night/turns. This is a minimum functional check, not statistical proof of generalization or an official ground-truth substitute. Independent public/private performance cannot be estimated from reused 50 rows.

No new performance gain is established by the current tracking/center experiments; their value is ruling out an unreliable feature route and defining what a score-relevant next feature must demonstrate.
