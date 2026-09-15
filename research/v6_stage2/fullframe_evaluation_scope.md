# Full-frame V5 diagnostic: contract and evaluation scope

2026-09-15. `evaluate_nexar_fullframe_v5.py` evaluates the six already-exposed reviewed sources. It is a CPU analysis of a frozen V5 run, not a new candidate or independent acceptance experiment.

## Input contract evidence

- `대회_통합_정보.md:31` describes Stage2 as consecutive frame images per video; line 92 distinguishes public video examples from actual evaluation folders containing full frame images. Line 59 requires original filename frame numbers. Line 148 defines temporal scoring using each video's frame–time mapping. These do not establish a universal Stage2 FPS or expose the evaluator's time mapping to inference.
- `[Baseline_Inference]_3Stage_추론및ZIP생성.ipynb`, section 3: process ordered frame images and preserve original filename numbers without renumbering. Section 7 explicitly describes preparation of **public sample inputs**, converting all original frames to JPEG, starting numbering at zero. Its extraction cell at JSON lines 593–613 reads all frames and increments the index by one.
- The notebook's sample extraction is not evidence of the hidden server's exact JPEG quality, decoder, FPS distribution or timestamps. Stage3's 10Hz statement does not apply to Stage2.
- This experiment compares full native PNG against prior sampled PNG from the same source. It changes local frame density/timeline. It does not recreate the hidden rendering byte for byte. The evaluator validates source, image and native-time bindings, but does not independently re-decode all shared image pixels; that limitation is recorded in the result.

## Known-label scoring

Exact preserved human-review hashes and intake-native frame times are checked. These are single-human drafts, not official or adjudicated GT. Contact is known for six sources; entry for five; direction for four; space for six. Unknown fields remain null. Time errors use native timestamps, not a fixed FPS or index difference. Categorical macro-F1 includes both declared classes, with an absent-class zero denominator contributing zero; the known cohort size is always reported.

Results include each video's full-frame and prior-10Hz predictions, known-label errors, and changes in correctness. Historical reserved reports contribute their frozen V5 `baseline` field only; the rejected C candidate is not a full-frame baseline. `S2`, `official_S2` and `gate_passed` remain null: differing incomplete label subsets must not be combined into a complete competition score or an acceptance decision.

Synthetic CPU checks cover timestamp mismatch rejection, missing-label preservation and complete-class macro-F1. No GPU inference is performed by the evaluator, and it refuses to overwrite an existing evaluation.
