# Merged TPO / CLIP visual weights

`merged_visual.pt` is a transformed version of the publicly released TPO pretrained adapter combined with the official OpenAI CLIP ViT-B/16 visual weights. It is not a model newly fitted to competition evaluation data.

TPO: Guray Ozgur, Fadi Boutros and Naser Damer, “Tomatoes, Potatoes, and Onions: Questioning the Need for Faces in Face Presentation Attack Detection,” 2026, arXiv:2608.21455. Copyright (c) 2026 Fraunhofer Institute for Computer Graphics Research IGD Darmstadt. Official source: https://github.com/gurayozgur/TPO . License: Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International; original full text retained in `source/LICENSE`. The merged TPO-derived weight artifact is distributed under CC BY-NC-SA 4.0 with underlying CLIP rights preserved.

OpenAI CLIP: official source https://github.com/openai/CLIP . The original MIT license is retained in `clip_source/LICENSE`. The inference runtime loads the unmodified `clip_source/clip/model.py` directly; it does not import a tokenizer or download weights.

Modification made on 2026-09-11: the fixed query and value low-rank updates are added into each CLIP attention projection by `W_q/v += (alpha / sqrt(rank)) * B @ A`. Original key projections, other visual tensors and the classification head are retained. Floating-point weight storage is float32. This is algebraic checkpoint conversion without optimization, pseudo-labeling or training. Source/output hashes and all 24 update operations are recorded in `merged_manifest.json`. The conversion was independently implemented as matrix arithmetic in `scripts/export_tpo_merged.py` without importing or copying TPO, FoundPAD, loralib or CLIP-LoRA implementation code.

The submitted inference artifact requires neither the TPO LoRA implementation nor any CLIP-LoRA/loralib source file. Original source and unmerged checkpoints retained in the development workspace are excluded from the submission. These code exclusions do not remove the TPO pretrained weights' attribution, noncommercial, or share-alike conditions.

Pre-existing third-party copyrights are not participant-owned award deliverables. Preserve these rights and license conditions separately from any competition assignment of participant-authored work.
