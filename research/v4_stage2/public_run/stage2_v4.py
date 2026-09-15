"""Isolated V4 A/B experiments; unchanged motion collision, optional entry fine.

A executes the original overview/entry/space questions with motion collision
context. B adds one guarded entry question. Original filename numbers, not FPS,
identify outputs. No cross-file answer state, training, crops, or tracking.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from . import stage2_v2 as base
from .stage2_entry_refine import _refinement_indices
from .vlm_candidate import CandidateVLM

COLUMNS = base.COLUMNS
LOG = logging.getLogger(__name__)


def _observed_integer(result, key, offered):
    value = result.get(key)
    return value if type(value) is int and value in offered else None


def _validate(paths, scores):
    if not paths:
        raise ValueError("At least one original frame is required")
    numbers = [base._frame_number(path) for path in paths]
    if numbers != sorted(set(numbers)):
        raise ValueError("Frame numbers must be unique and chronologically ordered")
    if np.asarray(scores).shape != (len(paths),) or not np.isfinite(scores).all():
        raise ValueError("Finite motion scores must match the original frame list")
    return numbers


def _refine_entry(paths, numbers, prediction, diagnostics, vlm):
    original = prediction["entry_frame"]
    coarse_frames = diagnostics["entry_candidates"]
    audit = {"coarse_entry_frame": original, "additional_calls": 0,
             "accepted": False, "fine_candidates": [], "skip_reason": None}
    diagnostics["entry_refinement"] = audit
    if _observed_integer(diagnostics["entry"], "entry_frame", coarse_frames) is None:
        audit["skip_reason"] = "invalid_or_unshown_coarse_entry"
        return
    fine, skip, interval = _refinement_indices(
        numbers, prediction["collision_frame"], original, coarse_frames)
    audit.update(fine_candidates=[numbers[index] for index in fine], skip_reason=skip,
                 interval_positions=list(interval) if interval else None,
                 interval_frames=[numbers[index] for index in interval] if interval else None)
    if not fine:
        return
    prompt = (
        "These dashcam frames are chronological, left to right then top to bottom. "
        "This is a local time window of the full clip; its first image may not be the start of the clip. "
        "Find the collision vehicle's first wheel touching the camera car's lane boundary. "
        "At an intersection continue the camera car's lane boundaries forward. "
        "Return status OBSERVED only if this entry is visible in these images. "
        "If entry happened before this window use BEFORE_WINDOW; if after it use AFTER_WINDOW; "
        "if the event or the collision vehicle cannot be determined use UNCERTAIN. "
        "Do not call the first shown image the actual entry merely because the vehicle is already inside. "
        f"Lane entry candidates: {audit['fine_candidates']}. "
        "Return JSON with status and entry_frame; entry_frame must be a shown integer only when status is OBSERVED."
    )
    audit.update(prompt=prompt, additional_calls=1)
    diagnostics["calls"] += 1
    try:
        answer = vlm.ask([base._sheet(paths, fine, columns=4)], prompt, max_new_tokens=64)
        audit["raw_output"] = answer
        try:
            parsed = json.loads(answer)
        except (ValueError, TypeError):
            parsed = {}
        if not isinstance(parsed, dict):
            parsed = {}
        audit["parsed"] = parsed
        choice = _observed_integer(parsed, "entry_frame", audit["fine_candidates"])
        if parsed.get("status") == "OBSERVED" and choice is not None:
            prediction["entry_frame"] = choice
            audit.update(accepted=True, selected_local_window_start=choice == numbers[fine[0]] and fine[0] > 0)
        else:
            audit["fallback_reason"] = "entry_not_observed_or_invalid_frame"
    except Exception as error:
        audit.update(fallback_reason="refinement_call_failed", error=f"{type(error).__name__}: {error}")


def _predict_file(paths, scores, vlm, *, variant="B"):
    if variant not in {"A", "B"}:
        raise ValueError("V4 Stage2 variant must be A or B")
    numbers = _validate(paths, scores)
    collision = int(np.argmax(scores))
    overview = base._uniform_indices(0, len(paths)-1, 10)
    coarse = base._json_object(vlm.ask(
        [base._sheet(paths, overview)],
        "These are frames from one dashcam clip, ordered left to right, top to bottom. "
        "Which numbered frame first shows physical contact of the camera vehicle with another vehicle? "
        "From which side of the image did that other vehicle approach? "
        "Return JSON with collision_frame and entry_side (LEFT or RIGHT). "
        f"Available frames: {[numbers[index] for index in overview]}.", max_new_tokens=64))
    side = str(coarse.get("entry_side", "")).upper().strip()
    if side not in {"LEFT", "RIGHT"}:
        LOG.warning("V4 entry_side invalid; retaining frozen LEFT fallback")
        side = "LEFT"
    entry_candidates = base._uniform_indices(0, collision, 12)
    entry_result = base._json_object(vlm.ask(
        [base._sheet(paths, entry_candidates, columns=4)],
        "These frames are chronological, left to right then top to bottom. "
        "The clip ends at a collision. When did the other collision vehicle first enter the camera car's driving lane? "
        "Entry is its first wheel touching the lane boundary. "
        "If it was already inside this lane at the first image, select the first image. "
        "At an intersection continue the camera car's lane boundaries forward. "
        f"Lane entry candidates: {[numbers[index] for index in entry_candidates]}. "
        "Return JSON with entry_frame only.", max_new_tokens=40))
    entry = base._choice(entry_result, "entry_frame", paths, entry_candidates, 0)
    context = sorted(set([max(0, collision-2), collision, min(len(paths)-1, collision+2)]))
    space_result = base._json_object(vlm.ask(
        [base._sheet(paths, context, columns=3)],
        "These images show just before, during, and after a dashcam collision. "
        "At contact, is there usable road space for the camera car to continue or steer around the other vehicle? "
        "Account for nearby traffic, road edges, curbs and barriers. "
        "Return JSON with evasion_space: integer 1 if space exists, otherwise integer 0.", max_new_tokens=40))
    space = base._integer(space_result.get("evasion_space"))
    if space not in {0, 1}:
        LOG.warning("V4 evasion_space invalid; retaining frozen zero fallback")
        space = 0
    prediction = dict(collision_frame=numbers[collision], entry_frame=numbers[entry], entry_side=side, evasion_space=space)
    diagnostics = dict(version=f"v4_{variant}", calls=3, coarse=coarse, entry=entry_result, space=space_result,
                       motion_proposal=numbers[collision], collision_context_frame=numbers[collision],
                       overview_candidates=[numbers[index] for index in overview],
                       entry_candidates=[numbers[index] for index in entry_candidates],
                       space_context_frames=[numbers[index] for index in context])
    if variant == "B":
        _refine_entry(paths, numbers, prediction, diagnostics, vlm)
    return prediction, diagnostics


def predict_stage2(data_dir, model_dir, *, variant="B"):
    if variant not in {"A", "B"}:
        raise ValueError("V4 Stage2 variant must be A or B")
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
            prediction, diagnostics = _predict_file(paths, scores, vlm, variant=variant)
            LOG.info("Stage2 V4 %s %s: %s", variant, folder.name, json.dumps(diagnostics, ensure_ascii=False))
            rows.append(dict(ID=folder.name, **prediction))
    return pd.DataFrame(rows, columns=COLUMNS)
