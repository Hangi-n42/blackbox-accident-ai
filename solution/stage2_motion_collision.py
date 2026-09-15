"""Stage2 ablation: retain frozen NF4 answers except motion-argmax collision.

Motion is a proxy, not verified physical contact. The other three answers retain
the original VLM collision context, even when this replacement is earlier.
"""
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from . import stage2_v2 as base
from .vlm_candidate import CandidateVLM

COLUMNS = base.COLUMNS


def _predict_file(paths, scores, vlm):
    prediction, diagnostics = base._predict_file(paths, scores, vlm)
    replacement = base._frame_number(paths[int(np.argmax(scores))])
    diagnostics = {**diagnostics, "collision_replacement": {
        "policy": "existing_motion_scores_argmax_original_frame_number",
        "base_collision_frame": prediction["collision_frame"],
        "collision_frame": replacement,
        "other_fields_use_base_vlm_context": True,
    }}
    return {**prediction, "collision_frame": replacement}, diagnostics


def predict_stage2(data_dir, model_dir):
    # Keep the frozen stage2_nf4 folder ordering, validation and model lifecycle.
    image_root = Path(data_dir) / "images"
    if not image_root.is_dir():
        raise FileNotFoundError(f"Missing Stage2 image directory: {image_root}")
    rows = []
    with CandidateVLM(Path(model_dir) / "vlm", precision="nf4") as vlm:
        for folder in sorted(path for path in image_root.iterdir() if path.is_dir()):
            paths = sorted((path for path in folder.iterdir()
                            if path.suffix.lower() in base.IMAGE_EXTENSIONS), key=base._frame_number)
            numbers = [base._frame_number(path) for path in paths]
            if len(numbers) != len(set(numbers)):
                raise ValueError(f"Duplicate Stage2 frame number in {folder.name}")
            paths, scores, _ = base._motion_scan(paths)
            prediction, diagnostics = _predict_file(paths, scores, vlm)
            logging.getLogger(__name__).info("Stage2 motion collision %s: %s", folder.name,
                                            json.dumps(diagnostics, ensure_ascii=False))
            rows.append(dict(ID=folder.name, **prediction))
    return pd.DataFrame(rows, columns=COLUMNS)
