# Input visual peer QA B

PASS. All 6 actual input images and 3 native frames were directly viewed. New predictions unseen; model calls 0.

| Frame | Same SUV / framing | Marker / wheel-lane clearance | Exact construction |
|---|---|---|---|
| 31 | PASS | PASS | PASS |
| 35 | PASS | PASS | PASS |
| 39 | PASS | PASS | PASS |

Both arms remain 1152x768. Source detail is native 1280x720 resized by BICUBIC to 1152x648 and placed at (0,84). Pixel reconstruction passed for all three. Historical control bytes, marker mask/pixels, header and padding are exact.

References are byte-identical to the preceding wheel-state experiment: f31 OUTSIDE, f35 UNKNOWN, f39 INSIDE. No reannotation was performed.

- Source_detail is 90 percent native width (1280 to 1152), not untouched native resolution.
- Detail retention, blur and aliasing differ jointly; visual QA does not isolate a single causal mechanism.
- Small f31/f35 marks obscure some upper-window pixels in both arms, but no wheel/lane evidence is additionally covered.
- Original compression and ambiguous f35 wheel-boundary relation remain; UNKNOWN reference is unchanged and must remain unscored.
- Prior-exposed AI references and privileged counterpart markers are not human-certified or deployable target detection.
- Pixel QA does not verify future processor token/grid equality or model understanding; those need separate runtime evidence.

Protected inputs/references/protocol and previous independent reviews were SHA256-verified unchanged. Only this JSON/MD pair was created.
