"""Unadopted V5: target reference, global/local entry search, terminal classes.

No learned state, FPS assumptions, cross-file answers, or extra VLM retries.
Motion remains a collision proxy. ROI tracking is geometric, not identity proof.
"""
from __future__ import annotations

import json
import logging
import math
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont, ImageOps

from . import stage2_v2 as base
from .vlm_candidate import CandidateVLM

COLUMNS = base.COLUMNS
LOG = logging.getLogger(__name__)
CANVAS = (1280, 928)  # 1,187,840 pixels, under the frozen 1.2 MP ask budget.
REFERENCE_HEIGHT = 160
TOKENS = (64, 96, 48, 48)
TRACK_STEP_BUDGET = 300


def _strict_index(value, paths, offered):
    if type(value) is not int:
        return None
    return next((i for i in offered if base._frame_number(paths[i]) == value), None)


def _bbox(value):
    if not isinstance(value, list) or len(value) != 4:
        return None
    if any(type(x) not in (int, float) or not math.isfinite(x) for x in value):
        return None
    a, b, c, d = map(float, value)
    if not (0 <= a < c <= 1 and 0 <= b < d <= 1):
        return None
    if c-a < .01 or d-b < .01:
        return None
    return [a, b, c, d]


def _gray(path):
    image = base._read_rgb(path)
    size = image.size
    small = image.resize((320, max(16, round(size[1]*320/size[0]))))
    return cv2.cvtColor(np.asarray(small), cv2.COLOR_RGB2GRAY), size


def _advance_roi(previous, current, box):
    """One adjacent-frame step. Reject rather than retain a stale box."""
    if previous.shape != current.shape:
        return None, "resolution_change"
    height, width = previous.shape
    mask = np.zeros_like(previous)
    x1, y1, x2, y2 = np.asarray(box)*[width, height, width, height]
    mask[max(0, int(y1)):min(height, math.ceil(y2)),
         max(0, int(x1)):min(width, math.ceil(x2))] = 255
    points = cv2.goodFeaturesToTrack(previous, maxCorners=64, qualityLevel=.01,
                                    minDistance=3, mask=mask, blockSize=3)
    if points is None or len(points) < 6:
        return None, "too_few_corners"
    params = dict(winSize=(21, 21), maxLevel=3,
                  criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, .01))
    forward, sf, _ = cv2.calcOpticalFlowPyrLK(previous, current, points, None, **params)
    if forward is None or sf is None or not np.isfinite(forward).all():
        return None, "forward_failed"
    backward, sb, _ = cv2.calcOpticalFlowPyrLK(
        current, previous, forward, points.copy(), flags=cv2.OPTFLOW_USE_INITIAL_FLOW,
        **{**params, "maxLevel": 0})
    if backward is None or sb is None:
        return None, "backward_failed"
    old, new = points.reshape(-1, 2), forward.reshape(-1, 2)
    error = np.linalg.norm(old-backward.reshape(-1, 2), axis=1)
    good = ((sf.ravel() == 1) & (sb.ravel() == 1) & (error <= 1.5)
            & np.isfinite(error) & (new[:, 0] >= 0) & (new[:, 0] < width)
            & (new[:, 1] >= 0) & (new[:, 1] < height))
    if good.sum() < 6 or good.sum() <= len(points)/2:
        return None, "forward_backward_inconsistent"
    old, new = old[good], new[good]
    delta = np.median(new-old, axis=0)
    # A single median translation is insufficient when matched points disagree.
    residual = np.linalg.norm(new-old-delta, axis=1)
    consistent = residual <= 3.0
    if consistent.sum() < 6 or consistent.sum() <= len(points)/2:
        return None, "nonrigid_or_wrong_matches"
    old, new = old[consistent], new[consistent]
    oc, nc = np.median(old, axis=0), np.median(new, axis=0)
    radii = np.linalg.norm(old-oc, axis=1)
    usable = radii >= 2
    scale = float(np.median(np.linalg.norm(new[usable]-nc, axis=1)/radii[usable])) if usable.sum() >= 4 else 1.
    if not .8 <= scale <= 1.25 or np.linalg.norm(nc-oc) > .15*width:
        return None, "implausible_step"
    corners = (np.array([[x1, y1], [x2, y2]])-oc)*scale+nc
    raw_area = float(np.prod(corners[1]-corners[0]))
    corners[:, 0] = np.clip(corners[:, 0], 0, width)
    corners[:, 1] = np.clip(corners[:, 1], 0, height)
    clipped_area = float(np.prod(corners[1]-corners[0]))
    if raw_area <= 0 or clipped_area/raw_area < .75:
        return None, "leaves_frame"
    result = _bbox((corners.reshape(-1)/[width, height, width, height]).tolist())
    return result, "tracked" if result is not None else "invalid_box"


def _track_rois(paths, reference_index, box):
    boxes, events = {}, []
    if box is None:
        return boxes, [{"reason": "no_valid_reference_bbox"}]
    boxes[reference_index] = box
    if len(paths)-1 > TRACK_STEP_BUDGET:
        # Deterministic whole-video fallback; never truncate the entry timeline.
        return boxes, [{"reason": "tracking_budget_full_frame_fallback",
                        "required_steps": len(paths)-1, "budget": TRACK_STEP_BUDGET}]
    for step in (-1, 1):
        previous, native_size = _gray(paths[reference_index])
        current_box = box
        for index in range(reference_index+step, len(paths) if step == 1 else -1, step):
            current, current_size = _gray(paths[index])
            if native_size != current_size:
                events.append({"index": index, "direction": step, "reason": "resolution_change"})
                break
            updated, reason = _advance_roi(previous, current, current_box)
            if updated is None:
                events.append({"index": index, "direction": step, "reason": reason})
                break
            boxes[index] = updated
            previous, current_box = current, updated
    return boxes, events


def _crop(image, box, margin=.35):
    if box is None:
        return image.copy(), [0, 0, image.width, image.height]
    x1, y1, x2, y2 = np.asarray(box)*[image.width, image.height, image.width, image.height]
    dx, dy = (x2-x1)*margin, (y2-y1)*margin
    bounds = [max(0, int(x1-dx)), max(0, int(y1-dy)),
              min(image.width, math.ceil(x2+dx)), min(image.height, math.ceil(y2+dy))]
    return image.crop(bounds), bounds


def _paste(canvas, image, bounds):
    x, y, width, height = bounds
    shown = ImageOps.contain(image, (width, height))
    canvas.paste(shown, (x+(width-shown.width)//2, y+(height-shown.height)//2))
    return {"native_size": list(image.size), "shown_size": list(shown.size),
            "scale": shown.width/image.width}


def _canvas(paths, indices, reference=None, boxes=None, fine=False):
    """Original RGB crops, with explicit per-frame tracking/fallback metadata."""
    if not indices or len(indices) > 12:
        raise ValueError("V5 canvas requires 1..12 distinct candidate frames")
    canvas = Image.new("RGB", CANVAS, "#181818")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default(size=18)
    offset = REFERENCE_HEIGHT if reference is not None else 0
    metadata = {"canvas_size": list(CANVAS), "pixels": CANVAS[0]*CANVAS[1], "tiles": []}
    if reference is not None:
        refindex, refbox = reference
        original = base._read_rgb(paths[refindex])
        crop, bounds = _crop(original, refbox, .1)
        metadata["reference"] = {"frame": base._frame_number(paths[refindex]), "bbox": refbox,
                                 "crop_bounds": bounds, **_paste(canvas, crop, (0, 24, 320, 132))}
        draw.text((8, 2), f"REFERENCE frame {base._frame_number(paths[refindex])}", fill="white", font=font)
        draw.text((336, 28), "Same collision vehicle reference; not a timeline candidate.", fill="white", font=font)
        if refbox is None:
            draw.text((336, 60), "Target not confirmed. Full-frame fallback.", fill="orange", font=font)
    columns = 2 if 3 <= len(indices) <= 4 else min(4, len(indices))
    rows = math.ceil(len(indices)/columns)
    width, height = CANVAS[0]//columns, (CANVAS[1]-offset)//rows
    for slot, index in enumerate(indices):
        x, y = slot % columns*width, offset+slot//columns*height
        original = base._read_rgb(paths[index])
        box = (boxes or {}).get(index) if fine else None
        crop, bounds = _crop(original, box)
        tile = {"frame": base._frame_number(paths[index]), "index": index,
                "roi_tracked": box is not None, "bbox": box, "crop_bounds": bounds,
                **_paste(canvas, crop, (x, y+28, width, height-28))}
        if fine and box is not None:
            tile["context"] = _paste(canvas, original, (x+width-106, y+height-66, 104, 64))
        draw.text((x+4, y+3), f"frame {base._frame_number(paths[index])}"+ (" ROI" if box else " FULL"), fill="white", font=font)
        metadata["tiles"].append(tile)
    return canvas, metadata


def _coarse_selection(result, paths, offered, collision):
    first, best, bracket = 0, None, None
    status = result.get("status")
    if status == "ALREADY_AT_START" and result.get("same_target") is True:
        if _strict_index(result.get("best_frame"), paths, [0]) == 0:
            return 0, None, "already_at_start"
    if status == "BRACKET":
        before = _strict_index(result.get("before_frame"), paths, offered)
        after = _strict_index(result.get("after_frame"), paths, offered)
        best = _strict_index(result.get("best_frame"), paths, offered)
        valid_order = before is not None and after is not None and before < after
        if valid_order:
            bracket = (before, after)
            if best is not None and before <= best <= after:
                return best, bracket, "valid_bracket"
            return first, bracket, "invalid_coarse_best"
    return first, None, "invalid_or_uncertain_coarse"


def _entry_candidates(n, bracket, best):
    if bracket is None:
        return base._uniform_indices(0, n-1, 12)
    local = set(base._uniform_indices(bracket[0], bracket[1], 9))
    if bracket[0] <= best <= bracket[1]:
        local.add(best)
    # At most ten local candidates and two independent global anchors.
    return sorted(local | {0, n-1})


def _fine_selection(result, paths, offered, collision, coarse):
    if result.get("status") == "ALREADY_AT_START" and result.get("same_target") is True:
        selected = _strict_index(result.get("entry_frame"), paths, [0])
    elif result.get("status") == "OBSERVED":
        selected = _strict_index(result.get("entry_frame"), paths, offered)
    else:
        return coarse, "rejected_status"
    if selected is None:
        return coarse, "not_an_offered_integer"
    return selected, "accepted"


def _predict_file(paths, scores, vlm):
    if not paths or len(paths) != len(scores):
        raise ValueError("V5 requires nonempty paths and one score per path")
    numbers = [base._frame_number(path) for path in paths]
    if numbers != sorted(set(numbers)):
        raise ValueError("V5 frame numbers must be distinct and ascending")
    n, collision = len(paths), int(np.argmax(scores))
    diagnostics = {"version": "v5_reference_global_local_terminal", "calls": [], "render": [],
                   "collision_policy": "existing_motion_scores_argmax_original_frame_number"}

    def ask(indices, prompt, tokens, reference=None, boxes=None, fine=False):
        start = time.perf_counter()
        canvas, render = _canvas(paths, indices, reference, boxes, fine)
        render_seconds = time.perf_counter()-start
        start = time.perf_counter()
        raw = vlm.ask([canvas], prompt, max_new_tokens=tokens)
        result = base._json_object(raw)
        diagnostics["calls"].append({"prompt": prompt, "raw": raw, "parsed": result,
                                     "tokens": tokens, "vlm_seconds": time.perf_counter()-start,
                                     "render_seconds": render_seconds})
        diagnostics["render"].append(render)
        return result

    identity_indices = sorted({i for p in base._motion_peaks(scores) for i in (max(0, p-2), p)})
    identity = ask(identity_indices,
        "These original dashcam frames show candidate collision regions. Identify the OTHER vehicle "
        "that actually contacts the camera car, not a nearby uninvolved vehicle. Choose one shown frame "
        "where that vehicle is visible. Return JSON status IDENTIFIED, reference_frame, bbox [left,top,right,bottom] "
        "normalized to that full frame (0..1). If identity is unclear return status UNCERTAIN. "
        f"Shown original frame numbers: {[numbers[i] for i in identity_indices]}.", TOKENS[0])
    refindex = _strict_index(identity.get("reference_frame"), paths, identity_indices)
    refbox = _bbox(identity.get("bbox"))
    if identity.get("status") != "IDENTIFIED" or refindex is None or refbox is None:
        refindex, refbox = collision, None
    reference = (refindex, refbox)
    diagnostics["reference"] = {"frame": numbers[refindex], "bbox": refbox, "identified": refbox is not None}
    overview = base._uniform_indices(0, n-1, 12)
    coarse_result = ask(overview,
        "The REFERENCE identifies the collision vehicle. Timeline panels cover the WHOLE clip in order. "
        "Track that same vehicle backwards to its FIRST entry into the camera car's driving corridor. "
        "Entry is its first wheel touching the lane boundary; extend the corridor through intersections. "
        "Do not label a later re-entry or post-collision reappearance. Return JSON status BRACKET and "
        "before_frame (last shown before first entry), after_frame (first shown after entry), best_frame. "
        "If that same vehicle is already inside at the original clip start, use status ALREADY_AT_START, "
        "same_target true and best_frame equal to the first original frame. Otherwise use REENTRY or UNCERTAIN. "
        f"Original clip first frame: {numbers[0]}. Timeline candidates: {[numbers[i] for i in overview]}.",
        TOKENS[1], reference)
    coarse, bracket, reason = _coarse_selection(coarse_result, paths, overview, collision)
    candidates = _entry_candidates(n, bracket, coarse)
    diagnostics["coarse"] = {"candidates": [numbers[i] for i in overview], "best_frame": numbers[coarse],
                              "bracket": None if bracket is None else [numbers[i] for i in bracket], "reason": reason}
    entry = coarse
    entry_origin = "observed_already_at_start" if reason == "already_at_start" else (
        "coarse_retained" if reason == "valid_bracket" else "invalid_fallback_first")
    if n > 1:
        start = time.perf_counter()
        boxes, events = _track_rois(paths, refindex, refbox)
        diagnostics["tracking"] = {"seconds": time.perf_counter()-start,
                                   "tracked_frames": {str(numbers[i]): b for i, b in boxes.items()}, "events": events}
        fine_result = ask(candidates,
            "REFERENCE is the same collision vehicle. ROI panels track it; FULL panels retain scene context. "
            "The offered frames mix a local search window and original clip start/end anchors; the first "
            "local image may NOT be the start of the full clip. Find this vehicle's FIRST wheel contact "
            "with the camera car's lane boundary, never a later re-entry. Return JSON status OBSERVED and "
            "entry_frame from offered frames only. If the SAME target was already inside at the original "
            "clip start use ALREADY_AT_START, same_target true and that original first frame. "
            "If entry is before/after the observed window, re-entry, or unclear, use BEFORE_WINDOW, "
            "AFTER_WINDOW, REENTRY, or UNCERTAIN without guessing a time. "
            f"Original first frame: {numbers[0]}. Offered frames: {[numbers[i] for i in candidates]}.",
            TOKENS[2], reference, boxes, True)
        entry, fine_reason = _fine_selection(fine_result, paths, candidates, collision, coarse)
        if fine_reason == "accepted":
            entry_origin = "observed_already_at_start" if fine_result.get("status") == "ALREADY_AT_START" else "fine_observed"
        diagnostics["fine"] = {"candidates": [numbers[i] for i in candidates], "reason": fine_reason}
    else:
        diagnostics["fine"] = {"candidates": [numbers[0]], "reason": "single_frame_skip"}
    # Commit temporal state before classification: response time/bbox keys are ignored.
    committed = {"collision_frame": numbers[collision], "entry_frame": numbers[entry]}
    diagnostics["entry_origin"] = entry_origin
    diagnostics["entry_after_motion_collision"] = entry > collision
    context = sorted({max(0, entry-1), entry, collision})
    classes = ask(context,
        "Use the REFERENCE to identify the SAME collision vehicle. Time choices are already fixed. "
        "Report entry_side LEFT or RIGHT: the image side where it originated BEFORE entering the camera "
        "car's driving corridor, not travel direction or impact side. At the fixed collision frame, "
        "is there usable road space to continue or steer around it, considering traffic, edges and barriers? "
        "Return JSON entry_side and evasion_space (integer 1 if usable space exists, otherwise 0) only. "
        f"Fixed entry frame: {numbers[entry]}; fixed collision frame: {numbers[collision]}.",
        TOKENS[3], reference)
    side = classes.get("entry_side")
    space = classes.get("evasion_space")
    diagnostics["classification_fallback"] = {"entry_side": side not in {"LEFT", "RIGHT"} if isinstance(side, str) else True,
                                                "evasion_space": type(space) is not int or space not in (0, 1)}
    if side not in ("LEFT", "RIGHT"):
        side = "LEFT"
    if type(space) is not int or space not in (0, 1):
        space = 0
    return {**committed, "entry_side": side, "evasion_space": space}, diagnostics


def predict_stage2(data_dir, model_dir):
    image_root = Path(data_dir)/"images"
    if not image_root.is_dir():
        raise FileNotFoundError(f"Missing Stage2 image directory: {image_root}")
    rows = []
    with CandidateVLM(Path(model_dir)/"vlm", precision="nf4") as vlm:
        for folder in sorted(p for p in image_root.iterdir() if p.is_dir()):
            paths = sorted((p for p in folder.iterdir() if p.suffix.lower() in base.IMAGE_EXTENSIONS), key=base._frame_number)
            numbers = [base._frame_number(p) for p in paths]
            if len(numbers) != len(set(numbers)):
                raise ValueError(f"Duplicate Stage2 frame number in {folder.name}")
            paths, scores, _ = base._motion_scan(paths)
            prediction, diagnostics = _predict_file(paths, scores, vlm)
            LOG.info("Stage2 V5 %s: %s", folder.name, json.dumps(diagnostics, ensure_ascii=False))
            rows.append(dict(ID=folder.name, **prediction))
    return pd.DataFrame(rows, columns=COLUMNS)
