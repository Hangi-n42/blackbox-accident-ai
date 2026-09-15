"""Final D ablation: clarify origin-side wording in V3's first question only."""
import json
import logging
from pathlib import Path

import pandas as pd

from . import stage2_motion_collision as baseline

base = baseline.base
COLUMNS = baseline.COLUMNS
CandidateVLM = baseline.CandidateVLM
ORIGINAL = "From which side of the image did that other vehicle approach? "
REPLACEMENT = (
    "From which image side did that same vehicle originate before entering the camera car's driving corridor? "
    "Track the collision vehicle backwards. Report its image side of origin before lane entry, "
    "not its direction of travel, side of impact, or position after entry. "
)


class _FirstQuestion:
    def __init__(self, vlm):
        self.vlm = vlm
        self.calls = 0

    def ask(self, images, prompt, max_new_tokens=128):
        self.calls += 1
        if self.calls == 1:
            if prompt.count(ORIGINAL) != 1:
                raise ValueError("Frozen V3 overview wording did not match the D replacement")
            prompt = prompt.replace(ORIGINAL, REPLACEMENT, 1)
        return self.vlm.ask(images, prompt, max_new_tokens=max_new_tokens)


def _predict_file(paths, scores, vlm):
    # The call counter belongs to this file only, even if the model is shared.
    wrapper = _FirstQuestion(vlm)
    prediction, diagnostics = baseline._predict_file(paths, scores, wrapper)
    diagnostics["first_question_ablation"] = {
        "version": "v4_D_origin_side_on_V3", "original": ORIGINAL,
        "replacement": REPLACEMENT, "calls_forwarded": wrapper.calls,
        "later_candidates_may_change_if_first_collision_answer_changes": True,
    }
    return prediction, diagnostics


def predict_stage2(data_dir, model_dir):
    image_root = Path(data_dir)/"images"
    if not image_root.is_dir():
        raise FileNotFoundError(f"Missing Stage2 image directory: {image_root}")
    rows = []
    with CandidateVLM(Path(model_dir)/"vlm", precision="nf4") as vlm:
        for folder in sorted(path for path in image_root.iterdir() if path.is_dir()):
            paths = sorted((path for path in folder.iterdir() if path.suffix.lower() in base.IMAGE_EXTENSIONS), key=base._frame_number)
            numbers = [base._frame_number(path) for path in paths]
            if len(numbers) != len(set(numbers)):
                raise ValueError(f"Duplicate Stage2 frame number in {folder.name}")
            paths, scores, _ = base._motion_scan(paths)
            prediction, diagnostics = _predict_file(paths, scores, vlm)
            logging.getLogger(__name__).info("Stage2 V4 D %s: %s", folder.name, json.dumps(diagnostics, ensure_ascii=False))
            rows.append(dict(ID=folder.name, **prediction))
    return pd.DataFrame(rows, columns=COLUMNS)
