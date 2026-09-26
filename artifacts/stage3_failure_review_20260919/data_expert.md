# Stage3 data/domain expert audit — 2026-09-19

Read-only examination; no labels, training, or existing results changed. Inputs: group_weighting_20260919 all_paired_predictions.csv, training_weights.csv, run.py; training_basis_20260917 cases.json; source curated comma NPZ and rules_and_review.md. Calculations use stored predictions and sensors only.

## Confirmed findings

1. Maximum 49.375 route weight belongs to `expanded_19` CONSTANT at sample105 (10.5s). This source has training labels A17/D35/C1/S52. The sole CONSTANT point gets an entire route's CONSTANT mass; sensors a=+0.0886987 m/s², speed=16.095557 m/s do not provide evidence of a wrong label. `comma_00` has only three CONSTANT training points (110,240,245), each16.4583x. Route equality is not equivalent to reliable evidence equality. This is a mechanical reason for sensitivity, not proof of the entire failure cause.
2. Capping extremes is insufficient: date1 improves opposites278→199 but date2 39→40/date3 16→33, public S3 .808951→.779261. Thus blaming49x alone contradicts the capped control.
3. New capped opposites occur in strict strong labels: date1 abs(a) min.631, median.817, max1.452; date2 min1.338, median1.487, max1.543; date3 min.591, median.772, max1.235 m/s². These are well outside candidate ±.3 thresholds. They cannot be dismissed as threshold-border uncertainty. Sensor latency/true CAN longitudinal-vs-speed-derivative differences remain unmeasured; neither is proven responsible.
4. Date3's 19 new capped opposites all belong to `expanded_21`, DECELERATING, sample321 plus362/364–368/371/376–380/382–387. These are temporally correlated events, not19 independent driving cases. Date2's seven new cases all `expanded_19`312–318. Segment-level analysis is mandatory.
5. Date1 capped per-class F1: A .4247→.4229, D .4700→.5006, C .5501→.4328, S .9849→.9837. Opposite reduction coexists with broad CONSTANT loss. Date2 has only51 true A but baseline370 predicted A and capped391; the failure includes excessive acceleration predictions, not just A/D inversions.
6. Evaluation class composition differs radically: date1 A455/D385/C335/S437; date2 A51/D107/C764/S245; date3 A17/D124/C420/S0. Date3 Civic only5 A examples. Because fixed four-class macro F1 assigns absent STOPPED zero, date3 ceiling=.75; cross-date absolute F1 comparisons are misleading. Within-fold changes remain valid. Do not pool these overlapping folds as independent data.
7. Full external training2395: Civic A254/D278/C484/S229; RAV4 A207/D208/C701/S34. CONSTANT training median speeds are29.36m/s Civic,29.68m/s RAV4; A medians15.61/10.15; D16.66/13.44. Speed-state correlation exists in the available sample. Date1 Civic CONSTANT evaluation median speed15.48m/s. This supports a measurable domain confound to test; it does not prove the classifier uses speed rather than acceleration.
8. All23 source segments already development exposed; zero independent new test. Chronological blocks must be counted, not nominal10Hz sample counts. Official50 labels too few for reliable tuning; two additional errors already change S3 by.0174.

## What this rules out / does not

- Neither vehicle class mass imbalance alone, giant route weights alone, nor ambiguous evaluation labels is an adequate complete explanation.
- Reweighting preserves DIS representation; its failure does not prove all representations cannot transfer.
- Optical depth/camera rotation/moving objects remain plausible causes but this read-only audit does not identify a unique causal one.
- MacroF1's missing-class ceiling and small public data limit interpretation; they do not reverse observed within-fold regressions.

## Recommended next bounded experiment

Stop further vehicle/route weight sweeps. Test a single physical motion feature change while keeping curated2395 training rows, unweighted scaler, C=.03 classifier and labels fixed: derive camera-rotation-compensated background motion using consistent point tracks, and its relative change over time, preserving existing DIS baseline for paired comparison. No new dataset/model download is necessary for an OpenCV sparse-tracking prototype. Rotation compensation must first pass a geometric sanity check; a plain global affine fit is not calibrated camera motion and was already only descriptive in earlier work.

Use existing sensors strictly as supervision/diagnostics, never inference inputs. Before training, form train-only matched-speed A/D diagnostics with |a|>.5 and existing stability masks; both signs at similar speeds ask whether the new feature captures acceleration rather than speed/scene. Freeze speed-bin selection and feature parameters without the outer labels. Hold original evaluation unchanged; separately report fixed per-class F1 and episode-level opposite onset counts. If there is insufficient static support or unclear camera intrinsics, mark geometry confidence/missing rather than fabricate a corrected signal.

This differs from previous ROI masking/affine residual features: those did not establish same physical point correspondence, static-background validity, or projective rotation consistency. However the evidence does not justify calling this guaranteed best; a cheap feature/physics validation should precede another full classifier run. New data is not currently the blocker for testing this hypothesis; independent multi-vehicle labelled sequences will eventually be needed to estimate generalization.
