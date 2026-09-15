# V6C exposed reserved 00007: endpoint selection audit

Read-only CPU diagnosis, 2026-09-15. No model calls, candidate changes, new selection, or threshold search. The contact reference 625 is supplied by the parent/user review; this audit does not infer or adjudicate contact from images. Detailed values and input/code hashes: `uncapped_jerk_v6c_00007_diagnosis.json`.

## Direct numerical evidence

| Original frame | Native seconds | Raw jerk | Normalized jerk before upper clipping | Weighted residual | Weighted appearance | V5 score | C score |
|---|---:|---:|---:|---:|---:|---:|---:|
| 318 (V5 selection) | 10.600000 | 0.834321 | 10.171671 | 0.092805 | 0.356682 | 10.449488 | 10.621159 |
| 624 (near supplied contact 625) | 20.800000 | 0.129215 | 0.699452 | 0 | 0 | 0.699452 | 0.699452 |
| 627 | 20.900000 | 0.134655 | 0.772535 | 0 | 0 | 0.772535 | 0.772535 |
| 1203 | 40.100000 | 0.166917 | 1.205940 | 0 | 0.545943 | 1.751883 | 1.751883 |
| 1204 (C selection) | 40.133333 | 1.576394 | 20.140493 | 0 | 0.040546 | 10.040545 | 20.181040 |

Jerk median is 0.077147894, MAD 0.050208684, denominator 0.074439391. The 0.001 denominator floor is **not active**. The endpoint wins because its uncapped jerk exceeds the previous winner; neither residual nor appearance drives this selection. Removing the cap is sufficient to explain the change in the recorded score ranking. It does not establish that the maximum raw jerk identifies physical contact. Relative to supplied reference 625 (20.833333 seconds), old error is -10.233333 seconds and C error is +19.300000 seconds.

## Unequal-interval endpoint mechanism

The local Nexar sampler explicitly includes the first and last source frame in addition to the nearest-PTS 10Hz grid. Here the last inputs are 1197, 1200, 1203, 1204. Thus consecutive intervals are 0.1, 0.1, **0.033333 seconds**. No file is duplicated or renumbered, but the final pair spans one third of the preceding duration.

Code references: `run_nexar_reserved_baseline.py:58–61` includes endpoints; `candidates/stage2_uncapped_jerk_v6c.py:52` subtracts displacement vectors without time; line 24 removes only the jerk upper cap; line 27 preserves the first-score-zero rule; line 83 selects argmax over all valid frames, including the last. The first endpoint has an explicit zero rule, whereas the final endpoint is scored normally.

The unchanged flow computation compares displacement vectors without dividing by elapsed time. The final jerk is the norm of the difference between:

- 1200→1203: shift `[2.394762, -0.040868]`, norm 2.395110 pixels over 0.1 seconds;
- 1203→1204: shift `[0.820117, 0.033373]`, norm 0.820796 pixels over 0.033333 seconds.

The displacement magnitude drops to about one third while displacement divided by elapsed time has similar magnitudes (23.951 versus 24.624 pixels/second). Therefore the raw displacement difference is confounded by interval duration. The recorded endpoint maximum is strongly consistent with this sampling artifact. This is not proof that all motion is constant, that flow is physically correct, or that no genuine camera acceleration occurred. No revised time-aware score or alternate selection was evaluated.

Seven selected feature triples were independently recomputed with the frozen optical-flow parameters and matched the stored float32 features exactly. Source decoding with two threads reproduced the exact frozen PNG RGB hashes for frames 1197, 1200, 1203 and 1204. The final images show a continuous street scene, with no obvious black/corrupted frame or scene cut. This limited visual check cannot prove absence of subtle camera movement or infer collision timing.

## Deployment scope: not an explanation of the DACON score

This particular short endpoint interval was introduced by **our local 10Hz plus endpoints Nexar preparation**. The supplied baseline inference notebook extracts every decoded Stage2 frame sequentially (`frame_index` increases by one; no Stage2 resampling). `Baseline/data/stage2` contains videos, not a pre-existing official image input tree. The locally extracted public folders in `artifacts/public_eval/stage2/images` each contain 50 consecutive frames numbered 0–49.

`대회_통합_정보.md` lines 31, 59 and 92 describe full/consecutive Stage2 images and original filename numbers. Line 148 says evaluation converts frames to time using per-video mapping; it does not promise that this mapping is an inference input. The Stage2 predictor receives image folders and filenames, not native PTS. Stage3's explicit 10Hz rule must not be transferred to Stage2. Number spacing is not elapsed time in general, particularly for variable-frame-rate material. Consequently, this finding does **not** establish an endpoint artifact in official evaluation or explain the official V5 Stage2 aggregate score.

## One next experimentally justified direction

Audit **sampling-interval consistency** before another contact-score candidate: use a controlled constant-translation video with known native timestamps, and compare the same flow feature under the official-style full-frame extraction and the local 10Hz-plus-endpoints extraction. Keep source motion fixed and change only the sampling timeline; include unequal intervals as a negative control. A spurious endpoint response in constant motion would directly test the suspected temporal confound without tuning to contact labels. Establish what temporal information exists at deployment before proposing a correction; do not normalize frame indices as though they were timestamps without a constant-FPS contract.

This is an input/feature validity experiment, not evidence of improved contact accuracy. Even removal of this artifact would not establish success near reference 625, where both neighboring current scores are low. C remains rejected under its original reserved gate; this exposed cohort must not be reused as an independent acceptance set for a revised candidate.
