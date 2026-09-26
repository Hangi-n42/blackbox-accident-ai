# V8 package — 2026-09-20

Current package: `submit_v8.zip` in this directory. The previous `../submit_v7.zip` is preserved.

## Selection

- Stage1: TPO/CLIP visual backbone unchanged, strong anchored classifier lambda=0.01;12 full frames +12 horizontal flips, probability mean, threshold0.5, no forensic. Known validation improved; independent DLC confirmation was explicitly waived. Official score improvement is unverified.
- Stage2: previous V7/V6 uncapped-jerk policy with Qwen3-VL4B NF4. Subsequent contact/entry candidates did not pass adoption gates. Evidence: docs/stage2-goal-results-20260920.md and docs/stage2-entry-selector-results-20260920.md.
- Stage3: preserved V7 LogisticRegression acceleration + RandomForest steering model, SHA256 c52ecd15cf68856f2a7b0b37b120c7a961b3bd8889a33af4ec47f8b9719bb3b3. Newer research scores are not directly comparable and no replacement passed adoption gates. Evidence: artifacts/stage3_state_learning_20260919/REPORT.md and artifacts/stage3_followthrough_20260920/REPORT.md.

Stage1 data attribution and classifier modification are included in the ZIP. Original merge provenance is retained separately; the current merged manifest identifies the new checkpoint and training.

## Validation

Build and complete extraction hashes/CRC: see submit_v8.manifest.json. Actual extracted Stage1 and Stage3 entry runs: see stage1_public/report.json and stage3_public/report.json . Stage2 is unchanged by hash; CUDA NF4 execution is not repeated on Mac. Full three-stage server time and official scores are unverified. This task creates a ZIP only; it does not submit to DACON.

Completed checks: all54 archive files extracted with matching CRC/SHA; Stage1 10rows matched fixed candidate predictions (7.83s); Stage3 2998rows exactly matched V7 (11.90s); both offline with zero connection attempts. ZIP SHA256: `a56a51c7d525fd9cb8fbfac265896d790464442b938c48cac1b0ddbae258129b`. See validation.json.
