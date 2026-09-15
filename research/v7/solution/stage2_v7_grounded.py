"""Research-only four-call Stage2 candidate; no labels, tracking or new weights.

Import with the frozen V6 `solution` package on sys.path. Paths are the complete,
ordered, decoded-image output of V6._dual_motion_scan. Original numbers are never
renumbered. All windows use input positions, not assumed FPS or elapsed seconds.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

from solution import stage2 as primitives
from solution import stage2_uncapped_jerk_v6c as frozen

PARAMETERS = dict(version="v7_grounded_1", calls=4, max_new_tokens=[120, 24, 24, 24],
                  overview_uniform=16, motion_regions=4, window_fraction=1/8,
                  dense_count=16, reference_bbox_scale=1000, tile_size=[384, 256],
                  renderer_columns=4, final_context_radius=2,
                  motion_proposal_excluded_indices=[1], pixel_budget_per_call=1_200_000)


def _strict_json(raw):
    if not isinstance(raw, str):
        return {}
    text = raw.strip()
    if text.startswith("```json") and text.endswith("```"):
        text = text[7:-3].strip()
    elif text.startswith("```") and text.endswith("```"):
        text = text[3:-3].strip()
    try:
        result = json.loads(text)
    except (ValueError, TypeError):
        return {}
    return result if isinstance(result, dict) else {}


def _select(value, allowed, fallback):
    valid = type(value) is int and value in allowed
    return (allowed[value] if valid else fallback), valid


def _motion_regions(scores):
    peaks = []
    gap = max(1, len(scores)//16)
    # Stable ties, and do not treat the first shift-minus-zero as independent
    # evidence. The unchanged V6 argmax remains an explicitly retained candidate.
    for index in sorted(range(len(scores)), key=lambda i: (-float(scores[i]), i)):
        if index == 1:
            continue
        if all(abs(index - other) > gap for other in peaks):
            peaks.append(index)
        if len(peaks) == PARAMETERS["motion_regions"]:
            break
    return peaks


def _window(value, numbers, offered, fallback):
    lookup = {numbers[i]: i for i in offered}
    cap = max(0, math.ceil((len(numbers)-1)/8))
    sequence = isinstance(value, list) and len(value) == 3
    coarse, coarse_valid = _select(value[1] if sequence else None, lookup, fallback)
    reason = "invalid_bracket"
    if sequence and all(type(v) is int and v in lookup for v in value):
        lower, center, upper = [lookup[v] for v in value]
        if lower <= center <= upper:
            if upper-lower <= cap:
                return (lower, upper), center, dict(status="accepted", coarse_valid=True,
                                                       supplied=value, width_limit=cap)
            reason = "oversized_bracket"
        else:
            reason = "unordered_bracket"
    lower = max(0, min(coarse-cap//2, len(numbers)-1-cap))
    upper = lower + cap
    return (lower, upper), coarse, dict(status=reason, coarse_valid=coarse_valid,
                                       supplied=value, width_limit=cap)


def _dense(window, essential):
    lo, hi = window
    return sorted(set(primitives._uniform_indices(lo, hi, PARAMETERS["dense_count"])) | set(essential))


def _reference(plan, numbers, offered):
    lookup = {numbers[i]: i for i in offered}
    index, frame_valid = _select(plan.get("reference_frame"), lookup, None)
    box = plan.get("bbox")
    valid_box = (isinstance(box, list) and len(box) == 4
                 and all(type(v) is int and 0 <= v <= 1000 for v in box)
                 and box[0] < box[2] and box[1] < box[3])
    target = plan.get("target")
    if not isinstance(target, str) or len(target) > 80:
        target = ""
    # A valid rectangle is NOT a validated vehicle identity.
    return dict(index=index, frame=numbers[index] if frame_valid else None,
                bbox=box if frame_valid and valid_box else None, target=target,
                frame_valid=frame_valid, bbox_valid=bool(frame_valid and valid_box),
                identity_verified=False)


def _panel(image, title, box=None):
    width, height = PARAMETERS["tile_size"]
    tile = Image.new("RGB", (width, height), "#171717")
    fitted = ImageOps.contain(image, (width, height-28))
    x, y = (width-fitted.width)//2, 28+(height-28-fitted.height)//2
    tile.paste(fitted, (x, y))
    draw = ImageDraw.Draw(tile)
    draw.text((5, 3), title, fill="white", font=ImageFont.load_default(size=17))
    if box:
        bounds = (x+box[0]*fitted.width/1000, y+box[1]*fitted.height/1000,
                  x+box[2]*fitted.width/1000, y+box[3]*fitted.height/1000)
        draw.rectangle(bounds, outline="#ffff00", width=2)
    return tile


def _render(paths, entries, reference=None):
    """One composite so references share, not double, the existing pixel budget."""
    panels, manifest = [], []
    for index, role in entries:
        panels.append(_panel(primitives._read_rgb(paths[index]),
                             f"{role} frame {primitives._frame_number(paths[index])}"))
        manifest.append(dict(role=role, input_index=index,
                             frame=primitives._frame_number(paths[index]), crop=False))
    if reference and reference["frame_valid"]:
        source = primitives._read_rgb(paths[reference["index"]])
        box = reference["bbox"]
        panels.append(_panel(source, f"REFERENCE frame {reference['frame']}", box))
        manifest.append(dict(role="reference_full", input_index=reference["index"],
                             frame=reference["frame"], crop=False))
        if box:
            x0, y0 = math.floor(box[0]*source.width/1000), math.floor(box[1]*source.height/1000)
            x1, y1 = math.ceil(box[2]*source.width/1000), math.ceil(box[3]*source.height/1000)
            panels.append(_panel(source.crop((x0, y0, x1, y1)), f"REF CROP frame {reference['frame']}"))
            manifest.append(dict(role="reference_crop", input_index=reference["index"],
                                 frame=reference["frame"], crop=True, source_bounds=[x0,y0,x1,y1]))
    columns = min(PARAMETERS["renderer_columns"], len(panels))
    width, height = PARAMETERS["tile_size"]
    canvas = Image.new("RGB", (columns*width, math.ceil(len(panels)/columns)*height), "#171717")
    for slot, panel in enumerate(panels):
        canvas.paste(panel, ((slot % columns)*width, (slot//columns)*height))
    return canvas, dict(panels=manifest, canvas_size=list(canvas.size),
                       rgb_sha256=hashlib.sha256(canvas.tobytes()).hexdigest(),
                       panel_occupancy=len(panels)/(columns*math.ceil(len(panels)/columns)))


def predict_from_scan(paths, base_scores, new_scores, vlm):
    """Four independent calls, one file at a time; return prediction and trace."""
    paths = [Path(path) for path in paths]
    numbers = [primitives._frame_number(path) for path in paths]
    if not numbers or numbers != sorted(set(numbers)):
        raise ValueError("Expected nonempty ordered unique original frame numbers")
    base_scores, new_scores = np.asarray(base_scores), np.asarray(new_scores)
    if any(values.shape != (len(paths),) or not np.isfinite(values).all()
           for values in (base_scores, new_scores)):
        raise ValueError("One finite score per valid input frame is required")
    if hasattr(vlm, "pixel_budget") and vlm.pixel_budget != PARAMETERS["pixel_budget_per_call"]:
        raise ValueError("Candidate requires the frozen 1.2MP VLM pixel budget")
    motion = int(np.argmax(new_scores))
    peaks = _motion_regions(new_scores)
    overview = sorted(set(primitives._uniform_indices(0, len(paths)-1, 16)) | set(peaks))
    trace = []

    def ask(stage, indices, prompt, tokens, reference=None, entries=None):
        image, rendered = _render(paths, entries if entries is not None else [(i, stage) for i in indices], reference)
        raw = vlm.ask([image], prompt, max_new_tokens=tokens)
        parsed = _strict_json(raw)
        trace.append(dict(stage=stage, offered_frames=[numbers[i] for i in indices],
                          input_indices=list(indices), prompt=prompt, raw=raw, parsed=parsed,
                          max_new_tokens=tokens, rendering=rendered))
        return parsed

    plan = ask("overview", overview,
        "One dashcam clip; images run left to right, top to bottom. Locate the SAME vehicle involved in the first physical contact with the camera car. "
        "Return a JSON plan: contact:[before_frame,best_frame,after_frame], entry:[before_frame,best_frame,after_frame], "
        "reference_frame, bbox:[left,top,right,bottom], target (short visual description). "
        "All frame numbers must be offered below. Bbox uses 0..1000 coordinates of the original reference image, not this sheet. "
        "Use the closest surrounding offered frames for each event. Entry is its first wheel touching the camera car lane boundary; "
        "extend that lane through an intersection. If already inside at the true clip start use the first frame for entry. "
        "Search the whole timeline for both events; later reentry is not first entry. Use null for unknown fields. "
        f"True first frame: {numbers[0]}. Offered frames: {[numbers[i] for i in overview]}.", 120)
    reference = _reference(plan, numbers, overview)
    contact_window, coarse_contact, contact_window_status = _window(plan.get("contact"), numbers, overview, motion)
    entry_window, coarse_entry, entry_window_status = _window(plan.get("entry"), numbers, overview, 0)
    contact_candidates = _dense(contact_window, [coarse_contact, motion])
    entry_candidates = _dense(entry_window, [coarse_entry, 0])
    ref_text = (f"Reference frame {reference['frame']} suggests the vehicle: {reference['target']}. "
                "REFERENCE panels identify a possible counterpart, not additional event candidates. Verify against the whole frames; the proposed reference may be wrong. "
                if reference["frame_valid"] else "No reliable vehicle reference is available; use the whole frames. ")
    contact = ask("contact", contact_candidates,
        "Event candidates are chronological full dashcam frames. " + ref_text +
        "Choose the first physical contact with the camera car, not a later maximum shake. "
        "Contact can be outside the image; use the before/after vehicle motion evidence, not shake or proximity alone. "
        f"Allowed event frames: {[numbers[i] for i in contact_candidates]}. Return JSON with collision_frame only, or null if unknown.",
        24, reference)
    collision, contact_valid = _select(contact.get("collision_frame"),
                                        {numbers[i]: i for i in contact_candidates}, motion)
    entry = ask("entry", entry_candidates,
        "Event candidates are chronological dashcam frames, sampled from a local window plus the true clip start. " + ref_text +
        "Choose when that collision vehicle FIRST touches the camera car lane boundary with a wheel. "
        "Extend the camera car lane through an intersection. A local-window first image may be later than the beginning. "
        f"True clip first frame is {numbers[0]}; choose it only if already inside at the true beginning. "
        "Later reentry is not first entry. "
        f"Allowed event frames: {[numbers[i] for i in entry_candidates]}. Return JSON with entry_frame only, or null if unknown.",
        24, reference)
    entry_index, entry_valid = _select(entry.get("entry_frame"),
                                        {numbers[i]: i for i in entry_candidates}, coarse_entry)
    # The same input can appear in both groups: role labels disambiguate event context.
    context_entry = sorted({max(0, entry_index-2), entry_index, min(len(paths)-1, entry_index+2), 0})
    context_contact = sorted({max(0, collision-2), collision, min(len(paths)-1, collision+2)})
    context = sorted(set(context_entry+context_contact))
    spatial = ask("spatial", context,
        "Two labeled event groups from one dashcam clip: ENTRY context and CONTACT context. " + ref_text +
        "entry_side is the image LEFT or RIGHT from which the SAME vehicle first entered the camera car driving corridor; "
        "trace backwards, not its direction of travel or impact position. "
        "evasion_space is 1 if usable physical road space exists for the camera car to continue or steer around it at CONTACT, otherwise 0. "
        "Account for traffic, barriers, curbs and road edges; do not judge speed, reaction time or avoidance success. "
        "Use full scenes, not the reference crop, to judge space. "
        "Return JSON with entry_side and evasion_space only; use null if unknown.", 24, reference,
        [(i,"ENTRY") for i in context_entry]+[(i,"CONTACT") for i in context_contact])
    side = spatial.get("entry_side")
    side_valid = type(side) is str and side in {"LEFT", "RIGHT"}
    space = spatial.get("evasion_space")
    space_valid = type(space) is int and space in {0, 1}
    prediction = dict(collision_frame=numbers[collision], entry_frame=numbers[entry_index],
                      entry_side=side if side_valid else "LEFT", evasion_space=space if space_valid else 0)
    entry_route = ("valid_selected_first" if entry_index == 0 else "valid_selected") if entry_valid else (
        "valid_coarse_retained" if entry_window_status["coarse_valid"] else "invalid_fallback_first")
    return prediction, dict(version=PARAMETERS["version"], parameters=PARAMETERS, calls=4,
        frame_numbers=numbers, motion_proposal=numbers[motion], retained_motion_peaks=[numbers[i] for i in peaks],
        base_score_sha256=frozen._score_hash(base_scores), new_score_sha256=frozen._score_hash(new_scores),
        overview_frames=[numbers[i] for i in overview], reference=reference,
        coarse_collision_frame=numbers[coarse_contact], coarse_entry_frame=numbers[coarse_entry],
        contact_window=[numbers[i] for i in contact_window], contact_window_indices=list(contact_window),
        entry_window=[numbers[i] for i in entry_window], entry_window_indices=list(entry_window),
        contact_window_status=contact_window_status, entry_window_status=entry_window_status,
        collision_candidates=[numbers[i] for i in contact_candidates], entry_candidates=[numbers[i] for i in entry_candidates],
        contact_selection_route="valid_selected" if contact_valid else "invalid_fallback_v6_motion",
        entry_selection_route=entry_route, valid_first_is_not_verified_already_entered=True,
        side_selection_route="valid_selected" if side_valid else "invalid_fallback_LEFT",
        space_selection_route="valid_selected" if space_valid else "invalid_fallback_zero",
        entry_after_contact=entry_index > collision,
        spatial_context_contact_frame=numbers[collision], spatial_context_entry_frame=numbers[entry_index],
        temporal_outputs_final_before_spatial=True, trace=trace)
