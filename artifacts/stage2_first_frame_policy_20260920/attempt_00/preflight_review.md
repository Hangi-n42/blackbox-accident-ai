# First-frame policy final preflight

**PASS — before all new model predictions.**

12 fixed new-source cases;12 baseline jobs +5 state jobs;53 planned actual calls (48 baseline +5 state).600 source PNG hashes and10 rendered images checked. Previous969 frozen dependencies remain unchanged.

Independent A/B originals match their reference hashes. Compatible entry intervals use the conservative union: CCD000052 frame0 before-start; CCD001237 frames8–21; CCD000540 frames32–37. CCD001278 remains unknown because only A supplied an interval. State references:1 INSIDE,2 OUTSIDE,9 UNKNOWN. Five approved marks include two state/timing-unknown cases,643 and022.

All five marks use the approved A native boxes; reconstructed384×256 tiles and exact nearest3×1152×768 images match stored RGB bytes. Jobs contain no reference answer or interval. Input peer A independently viewed native/low/state images and reports PASS for all five.

The unchanged references.json records rendering as pending at its earlier creation time. The separate input_peer_review.json PASS plus this exact reconstruction resolves that pending execution condition; references.json and the original A/B reviews were not modified. Their SHA256 hashes are preserved in preflight_review.json.

Code/protocol hashes match the interim static review. The known-OUTSIDE false-first audit covers all queried cases independent of timing eligibility. The INSIDE-only entry override and all other-state/no-marker fallbacks preserve the other three outputs. verify.py is prepared and AST-checked; runtime CPU replay/processor verification must wait for all timed model workers to finish.

## Interpretation limits

- AI references and peer rendering QA are not human or official ground truth; B adjudication is not a third independent reviewer.
- Entry reference only3/12: before-start1 and during-clip2. CCD001237 interval0.8..2.1seconds is wider than0.6, so no single prediction can be definitely correct within0.3seconds across the entire interval; it is not narrowed.
- Two queried state-UNKNOWN/timing-unknown cases remain unscored for those labels but their policy responses must be reported.
- Counterpart marks are privileged agent-provided input; this is not an autonomous deployable target detector.
- Bounded source/incident overlap screen is not universal independence certification.
- Preflight validates readiness only, not model success, score gain, CUDA behavior or officialS2.
