"""C-only ablation: original clip-start reference for guarded entry refinement.

The first three A questions remain unchanged. The additional two-image question
shares the fixed visual budget, reducing the local sheet's pixel allocation.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd

from . import stage2_v4 as ab

base = ab.base
COLUMNS = ab.COLUMNS
CandidateVLM = ab.CandidateVLM
LOG = logging.getLogger(__name__)


def _predict_file(paths, scores, vlm):
    prediction, diagnostics = ab._predict_file(paths, scores, vlm, variant="A")
    diagnostics["version"] = "v4_C_start_reference"
    numbers = [base._frame_number(path) for path in paths]
    first = numbers[0]
    original = prediction["entry_frame"]
    audit = {"coarse_entry_frame": original, "full_clip_first_frame": first,
             "additional_calls": 0, "accepted": False, "fine_candidates": [], "skip_reason": None,
             "two_image_budget_confounds_local_sheet_resolution": True}
    diagnostics["entry_refinement"] = audit
    coarse_frames = diagnostics["entry_candidates"]
    if ab._observed_integer(diagnostics["entry"], "entry_frame", coarse_frames) is None:
        audit["skip_reason"] = "invalid_or_unshown_coarse_entry"
        return prediction, diagnostics
    fine, skip, interval = ab._refinement_indices(numbers, prediction["collision_frame"], original, coarse_frames)
    audit.update(fine_candidates=[numbers[index] for index in fine], skip_reason=skip,
                 interval_positions=list(interval) if interval else None,
                 interval_frames=[numbers[index] for index in interval] if interval else None)
    if not fine:
        return prediction, diagnostics
    prompt = (
        f"Image 1 is the original full clip's first frame, numbered {first}. "
        "Image 2 is a local chronological contact sheet, ordered left to right then top to bottom. "
        "Its first tile may not be the start of the clip. "
        "Find when the SAME vehicle that collides with the camera car first enters the camera car's lane. "
        "Do not substitute a different vehicle merely because it is ahead in Image 1. "
        "Entry is that vehicle's first wheel touching the lane boundary; at intersections extend the camera car's lane forward. "
        f"If that same vehicle is already inside this lane in Image 1, use status ALREADY_IN_LANE_AT_START and entry_frame {first}. "
        "Use status OBSERVED only when the actual lane entry is visible in Image 2. "
        "Otherwise use BEFORE_WINDOW or AFTER_WINDOW for entry outside the local window; "
        "use UNCERTAIN if the collision vehicle or its lane entry cannot be identified. "
        f"Lane entry candidates: {audit['fine_candidates']}. "
        "Return JSON with status and entry_frame. OBSERVED requires a shown local integer frame; "
        "ALREADY_IN_LANE_AT_START requires the original first frame number; other statuses use null."
    )
    audit.update(prompt=prompt, additional_calls=1)
    diagnostics["calls"] += 1
    try:
        answer = vlm.ask([base._read_rgb(paths[0]), base._sheet(paths, fine, columns=4)], prompt, max_new_tokens=64)
        audit["raw_output"] = answer
        try:
            parsed = json.loads(answer)
        except (ValueError, TypeError):
            parsed = {}
        if not isinstance(parsed, dict):
            parsed = {}
        audit["parsed"] = parsed
        status, choice = parsed.get("status"), parsed.get("entry_frame")
        valid_start = status == "ALREADY_IN_LANE_AT_START" and type(choice) is int and choice == first
        valid_local = status == "OBSERVED" and ab._observed_integer(parsed, "entry_frame", audit["fine_candidates"]) is not None
        if valid_start or valid_local:
            prediction["entry_frame"] = choice
            audit.update(accepted=True, accepted_rule="full_clip_start" if valid_start else "observed_local",
                         selected_local_window_start=valid_local and choice == numbers[fine[0]] and fine[0] > 0)
        else:
            audit["fallback_reason"] = "status_or_frame_not_authorized_by_observed_inputs"
    except Exception as error:
        audit.update(fallback_reason="refinement_call_failed", error=f"{type(error).__name__}: {error}")
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
            LOG.info("Stage2 V4 C %s: %s", folder.name, json.dumps(diagnostics, ensure_ascii=False))
            rows.append(dict(ID=folder.name, **prediction))
    return pd.DataFrame(rows, columns=COLUMNS)
