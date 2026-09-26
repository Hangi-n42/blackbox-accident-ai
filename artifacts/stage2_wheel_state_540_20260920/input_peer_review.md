# Rendered input peer QA B

PASS. Three native sources and six low/upsampled images were directly viewed. No new model predictions read or model calls made.

| Frame | Same SUV / upper body | Wheel and lane unobscured | Exact nearest 3x |
|---|---|---|---|
| 31 | PASS | PASS | PASS |
| 35 | PASS | PASS | PASS |
| 39 | PASS | PASS | PASS |

A/B boxes differ slightly but all mark the same SUV upper windows. A boxes were used. Each unmarked scene exactly matches native BICUBIC 384x216 at offset (0,28), and only yellow marker pixels differ. Low 384x256 becomes upsampled 1152x768 exactly, including header and padding.

- Small f31/f35 rectangles become filled yellow blocks and obscure some upper-window pixels. They do not obscure the wheel-boundary evidence.
- Upsampling adds no source detail; original compression and low-input downsampling blur remain.
- QA PASS verifies input construction, not model target understanding or reference correctness. f35 remains UNKNOWN in unchanged references.
- Reviewers knew the prior case/entry interval; this is not fully unexposed reference validation.

This separate PASS resolves pending rendered QA without editing references. Original references/reviews/inputs were hash-verified unchanged.
