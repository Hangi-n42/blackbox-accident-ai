# Stage1/2/3 Git checkpoint — 2026-09-26

## Submitted baseline and research

- `releases/v8/source/` preserves the submitted V8 inference source, notices, and selected provenance. `releases/v8/submission.manifest.json` records all original submission files and hashes.
- V8 Stage1 uses the anchored classifier (lambda 0.01), 12 original frames plus horizontal flips, equal probability averaging, and threshold 0.5. Forensic is disabled.
- Stage2 and Stage3 in V8 retain the V7 submitted implementation/assets. Saving later experiments does not adopt their candidates into the submitted runtime.
- `releases/v7/` preserves the preceding source snapshot and submission manifest.
- `scripts/data/`, `scripts/mac/`, and explicitly tracked files under `artifacts/stage1*`, `artifacts/stage2*`, and `artifacts/stage3*` preserve authored experiments, audits, and reports at their original paths. Consult each report for adoption/rejection and evaluation limitations.

## Assets and reproduction limits

This is a source/evidence checkpoint, not a self-contained inference distribution. Model weights, datasets, videos, feature caches, prediction tables, downloaded dependency trees, tokenizer assets, and submission ZIPs remain external. Research scripts may require those local artifacts, including intermediate JSON files not selected for this checkpoint. No claim of clean-clone end-to-end execution is made.

The V8 builder is `scripts/build_v8_current.py`; it requires the original V7 ZIP, trained head, and local validation/provenance artifacts and refuses to overwrite an existing output directory. The complete original V8 ZIP remains at `artifacts/submissions/v8_20260920/submit_v8.zip` (SHA256 `a56a51c7d525fd9cb8fbfac265896d790464442b938c48cac1b0ddbae258129b`).

## Checkpoint validation

- Parsed all 288 staged Python files without syntax errors.
- No selected file exceeded 1 MB; no known GitHub/Hugging Face/OpenAI token or private-key-header pattern was found in the staged text scan. This is a limited scan, not a security guarantee.
- Existing whitespace/CRLF warnings were retained to preserve experimental and submitted source bytes.
- No retraining, full inference, model replacement, or competition submission was performed for this Git checkpoint.
