"""Unadopted Stage2 entry refinement candidate: four original calls plus at most one.

No entry ground truth or measured accuracy supports adoption. Only the entry
field may change. Sampling uses positions in the original ordered frame list;
filenames remain the public output identifiers and are never treated as FPS.
"""
from __future__ import annotations

import json

from . import stage2_v2 as base
from .stage2 import _frame_number, _uniform_indices


def _refinement_indices(numbers, collision_frame, entry_frame, coarse_frames, max_count=12):
    """Cover the two adjacent coarse intervals, retaining their selected center."""
    positions = {number: index for index, number in enumerate(numbers)}
    collision = positions[collision_frame]
    entry = positions[entry_frame]
    coarse = sorted({positions[number] for number in coarse_frames if number in positions
                     and positions[number] <= collision})
    if set(coarse) == set(range(collision+1)):
        return [], "all_precontact_frames_already_shown", None
    if entry not in coarse:
        return [], "coarse_entry_not_in_candidates", None
    selected = coarse.index(entry)
    lower = coarse[selected-1] if selected else 0
    upper = coarse[selected+1] if selected+1 < len(coarse) else collision
    if upper-lower+1 <= max_count:
        fine = set(range(lower, upper+1))
    elif entry in {lower, upper}:
        fine = set(_uniform_indices(lower, upper, max_count))
    else:
        # Allocate intervals on each side by available-frame count. The shared
        # center is included exactly, without displacing a neighbor and widening a gap.
        intervals = max_count-1
        left = min(intervals-1, max(1, round(intervals*(entry-lower)/(upper-lower))))
        fine = set(_uniform_indices(lower, entry, left+1))
        fine.update(_uniform_indices(entry, upper, intervals-left+1))
    if fine.issubset(coarse):
        return [], "no_new_frames_in_selected_interval", (lower, upper)
    return sorted(fine), None, (lower, upper)


def _predict_file(paths, scores, vlm):
    """Return the frozen base prediction, optionally refining entry once."""
    if not paths:
        raise ValueError("At least one original frame is required")
    numbers = [_frame_number(path) for path in paths]
    if numbers != sorted(set(numbers)):
        raise ValueError("Frame numbers must be unique and chronologically ordered")
    if len(scores) != len(paths):
        raise ValueError("Motion scores must match the original frame list")
    prediction, diagnostics = base._predict_file(paths, scores, vlm)
    prediction, diagnostics = dict(prediction), dict(diagnostics)
    original_entry = prediction["entry_frame"]
    fine, skip_reason, interval = _refinement_indices(
        numbers, prediction["collision_frame"], original_entry, diagnostics["entry_candidates"])
    audit = {"coarse_entry_frame": original_entry, "additional_calls": 0,
             "fine_candidates": [numbers[index] for index in fine],
             "interval_positions": list(interval) if interval is not None else None,
             "interval_frames": [numbers[index] for index in interval] if interval is not None else None,
             "skip_reason": skip_reason, "accepted": False}
    diagnostics.update(base_version=diagnostics["version"], version="candidate_entry_refine_once",
                       entry_refinement=audit)
    if not fine:
        return prediction, diagnostics
    audit["additional_calls"] = 1
    diagnostics["calls"] += 1
    prompt = (
        "These dashcam frames are chronological, left to right then top to bottom. "
        "They show a local time window of the full clip; the first image may not be the start of the clip. "
        "Which numbered frame first shows the other collision vehicle entering the camera car's lane? "
        "Entry is its first wheel touching the lane boundary. "
        "If it entered before this window, choose the first shown frame as the window boundary, "
        "not as proof of the actual entry time. "
        "At an intersection continue the camera car's lane boundaries forward. "
        f"Lane entry candidates: {audit['fine_candidates']}. "
        "Return JSON with entry_frame only."
    )
    audit["prompt"] = prompt
    try:
        answer = vlm.ask([base._sheet(paths, fine, columns=4)], prompt, max_new_tokens=40)
        audit["raw_output"] = answer
        try:
            result = json.loads(answer)
        except (ValueError, TypeError):
            result = {}
        if not isinstance(result, dict):
            result = {}
        audit["parsed"] = result
        choice = result.get("entry_frame")
        # A missing, fractional, string, boolean or unshown frame is not a valid choice.
        if type(choice) is int and choice in audit["fine_candidates"]:
            prediction["entry_frame"] = choice
            audit["accepted"] = True
            audit["selected_local_window_start"] = choice == numbers[fine[0]] and fine[0] > 0
        else:
            audit["fallback_reason"] = "invalid_or_unshown_entry_frame"
    except Exception as error:
        audit.update(fallback_reason="refinement_call_failed", error=f"{type(error).__name__}: {error}")
    return prediction, diagnostics
