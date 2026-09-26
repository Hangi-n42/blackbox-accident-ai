# State resolution independent verification

**PASS.** Frozen inputs, actual calls, categorical scoring and CPU processor reconstruction agree.

Frozen files before/after: 969; prior freeze preserved: 909; actual workers/calls: 15/15.
All three input recipes were reconstructed; original frame0 and frozen marker-box lineage match. High changes scene pixels outside the marker only.
Fresh low raw/enum controls match all five historical state results. No model was loaded or called by this verifier.

| Case | AI reference | Low | Upsampled | High |
|---|---|---|---|---|
| CCD_000688 | INSIDE | OUTSIDE | INSIDE | INSIDE |
| CCD_000801 | INSIDE | OUTSIDE | OUTSIDE | OUTSIDE |
| CCD_000453 | OUTSIDE | OUTSIDE | OUTSIDE | OUTSIDE |
| CCD_000196 | INSIDE | OUTSIDE | INSIDE | INSIDE |
| CCD_000728 | OUTSIDE | OUTSIDE | OUTSIDE | OUTSIDE |

| Arm | Correct / eligible | Grid t,h,w | Image tokens | Total input tokens |
|---|---:|---|---:|---:|
| low | 2/5 | [[1, 16, 24]] | 96 | 177 |
| upsampled | 4/5 | [[1, 48, 72]] | 864 | 945 |
| high | 4/5 | [[1, 48, 72]] | 864 | 945 |

Gate: {'gains': 2, 'losses': 0, 'inside_gains': 2, 'historical_control_match': True, 'pass': True}.
Constant-label accuracies: {'INSIDE': 0.6, 'OUTSIDE': 0.4}.
All 15 recreated processor tensor hashes and templated prompt hashes exactly match actual calls. MLX CPU was selected; no model weights were loaded and no new model inference occurred.
High and upsampled have identical input_ids, attention masks and image grids; their pixel tensors differ. Invalid/UNCERTAIN answers remain incorrect for eligible fixed references.

## Limits

- Five exposed AI references with privileged counterpart marking; semantic label validity is not certified.
- High uses 90 percent native width; this is not untouched native-resolution input.
- Low-to-high changes source rendering, grid and token count jointly; high-to-upsampled holds grid/tokens fixed.
- Source detail, blur and aliasing are not causally separated by the equal-size control.
- Categorical first-frame state does not validate first-wheel-contact timing or automatic frame-zero correction.
- Processor tensors were independently regenerated on CPU; model inference itself was not repeated.
- Python socket checks are narrower than OS networking; Mac does not establish CUDA equivalence; official S2 was not computed.

Source, frozen reference, model, worker and parent evaluation files were not changed. Exact per-call metadata, resource measurements and hashes are in independent_verification.json.
