# Initial five-sequence AI visual review

Reviewed contact sheets 000000,000001,000002,000004,000005, five frames each (25 total). This is AI image review, not human ground truth or proof of sub-frame hardware synchronization.

- 000000: wet road, overcast, wipers visible, changing road geometry. Forward travel visually consistent with positive sensor speed; exact small acceleration cannot be established visually.
- 000001: bridge, nearby truck, strong wide-angle distortion. Mostly constant-speed proxy with uncertain boundary left missing. Scene flow contains relative truck motion.
- 000002: bridge approach and truck overtaking; initial speed rises in CAN and OxTS. Both independent speed streams produce the same usable labels; strict support contains only five accelerating samples. Nearby truck motion is a potential confound.
- 000004: close truck occludes much of left/central view; road and image flow change with relative motion. Late invalid sensor interval stays missing; no visual label replacement.
- 000005: same bridge infrastructure appears as000001 on another date. Both assigned training; example of why date split does not guarantee geographical independence. Initial acceleration boundary stays missing. Glare and windshield artifacts visible.

Images decode, forward-time ordering, sensor-stream UTC consistency and coarse scene consistency pass for a bounded proxy experiment. Hardware latency is not measured; no claim of exact physical synchronization. Original camera has a wide fisheye field of view unlike comma; it remains unchanged in the data-only experiment. No augmentation, rectification, ROI change or label fitting to predictions.

All five pilot videos belong to training/development. Eight heldout videos remain outside fitting. Sensor QA exclusions 000006 and000007 were preserved, not rescued by visual guess. No STOPPED examples pass the current selected raw-sensor gates; four-class coverage is incomplete.
