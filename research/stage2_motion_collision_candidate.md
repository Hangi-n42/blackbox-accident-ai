# Stage 2 collision-only motion ablation

This candidate tests the already frozen per-file motion score argmax in place of the VLM collision answer. It does not establish that the strongest image motion is first physical contact.

The existing Qwen 4B NF4 policy still executes its four calls. Only the final collision_frame is replaced. Entry, direction, and space retain the original VLM answer and context; inconsistent temporal ordering is possible and is deliberately not hidden by an unvalidated correction. No offsets, filename-to-answer mappings, new training, or evaluation-data adaptation are used.

Stage 1 and Stage 3 weights remain identical to submission 87584. Stage 3 uses separately validated equivalent grouped feature reductions. The build variant is nf4_motion_fast, entrypoint inference_motion_fast.py, output submit_v3_motion_fast.zip. The scale augmentation candidate failed its development gate and is excluded.

CPU mocked contracts: 10 PASS. Independent agent source review found no blocking defect; it identified the contact/motion mismatch and retained-context limitation. Public examples contain only five collision labels and both old 4B and historical motion predictions got 4/5 within 0.3 seconds. No independent generalization improvement is established.

Pre-upload requirements: isolated offline execution of the actual extracted ZIP for all three stages; exact Stage 1 and Stage 3 output equality against V2; exact retained Stage 2 fields; original frame number membership; no model/other shared source changes in the manifest. Evidence will be recorded in artifacts/submissions/verify_v3_motion_fast_results.
