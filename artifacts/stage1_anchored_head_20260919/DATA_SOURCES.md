# Stage1 anchored-head data provenance

This experiment uses actual screen recaptures from VDmoire and DLC-2021. DLC originals are physical laminated mock documents, not screen captures; copy categories cc/cg are excluded. Document pairs denote content identity, not synchronized video pairs. DLC includes screen/window borders and domain-specific backgrounds. No claim of independent dashcam recapture validation is made.

DLC-2021 by Polevoy et al., original archive: https://zenodo.org/records/7467028 , recapture archive: https://zenodo.org/records/6466770 . Data licenses copied here: CC BY-SA 2.5. Generated Photos (https://generated.photos/) supplies generated face images used in the source mock documents; source annotations/attribution retained. No synthetic identity labels are used as model features.

The corrected provider CSV and archive index hashes are in helper_source_hashes.json. Each acquired frame records exact archive member, source URL, CRC32, SHA256, HTTP byte range, and reuse status. Raw downloaded JPEG bytes remain unchanged. Model inputs derive temporary full-frame/JPEG75/Gaussian blur1 views for both labels, plus horizontal flips.

VDmoire: https://github.com/CVMI-Lab/VideoDemoireing ; official REDS source: https://seungjunnah.github.io/Datasets/reds.html . Existing approved local provenance and REDS CC BY4.0 terms retained. See docs/stage1-temporal23-head-results-20260919.md for pairing and geometry limitations.

Road-original supplement: comma2k19 official https://github.com/commaai/comma2k19 and https://huggingface.co/datasets/commaai/comma2k19 . Existing source hashes/metadata and MIT notice in research/stage1/comma_original_diagnostic/ and external_data/comma2k19/LICENSE. Existing detected public-video duplicate remains excluded.

Official public examples are previously observed synthetic-replay regression fixtures. They are not training inputs and do not provide new physical-recapture confirmation. Existing VDmoire val examples are also previously observed guards.

Training uses no private competition data. Per-file inference is independent; no target-data adaptation or cross-file correction. Rules checked: https://dacon.io/competitions/official/236753/overview/rules . No new submission is authorized by this experiment.
