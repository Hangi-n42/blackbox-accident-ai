"""Stage 2 hypothesis V2: short, separate questions and retained motion proposals.

Uses frozen VLM weights with four calls per file. No answers from other files or
public labels are consulted. Existing Stage 2 V1 remains untouched for comparison.
"""
from __future__ import annotations

import json
import logging
import math
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont, ImageOps

from .stage2 import (
    COLUMNS, IMAGE_EXTENSIONS, _choice, _frame_number, _integer, _json_object,
    _motion_scan, _read_rgb, _uniform_indices,
)

LOG = logging.getLogger(__name__)


def _sheet(paths, indices, *, columns=5):
    """Consistent ten/twelve-frame contact sheet, no sample JSON/frame answers."""
    columns = min(columns, len(indices))
    width, height = 384, 256
    canvas = Image.new("RGB", (columns*width, math.ceil(len(indices)/columns)*height), "#171717")
    draw = ImageDraw.Draw(canvas)
    for slot, index in enumerate(indices):
        x, y = slot % columns * width, slot // columns * height
        image = ImageOps.contain(_read_rgb(paths[index]), (width, height-28))
        canvas.paste(image, (x+(width-image.width)//2, y+28))
        draw.text((x+8, y+2), f"frame {_frame_number(paths[index])}",
                  fill="white", font=ImageFont.load_default(size=20))
    return canvas


def _motion_peaks(scores, count=2):
    """Distinct motion regions; never discard them because coarse VLM disagrees."""
    peaks = []
    separation = max(3, len(scores)//10)
    for index in np.argsort(scores)[::-1]:
        if all(abs(int(index)-peak) > separation for peak in peaks):
            peaks.append(int(index))
        if len(peaks) >= min(count, len(scores)):
            break
    return peaks


def _fine_candidates(coarse, overview, scores, max_count=15):
    n = len(scores)
    previous = [index for index in overview if index < coarse]
    following = [index for index in overview if index > coarse]
    start = previous[-1] if previous else 0
    end = following[0] if following else n-1
    candidates = set(_uniform_indices(start, end, 5))
    peaks = _motion_peaks(scores)
    # Exact local frames at each retained motion peak, including preceding
    # frames because physical contact can precede maximum camera displacement.
    for peak in peaks:
        candidates.update(range(max(0, peak-3), min(n, peak+2)))
    if len(candidates) > max_count:
        essential = set(peaks+[coarse])
        rest = sorted(candidates-essential)
        keep = _uniform_indices(0, len(rest)-1, max_count-len(essential))
        candidates = essential | {rest[index] for index in keep}
    return sorted(candidates), peaks


def _predict_file(paths, scores, vlm):
    n = len(paths)
    motion = int(np.argmax(scores))
    overview = _uniform_indices(0, n-1, 10)
    coarse_result = _json_object(vlm.ask(
        [_sheet(paths, overview)],
        "These are frames from one dashcam clip, ordered left to right, top to bottom. "
        "Which numbered frame first shows physical contact of the camera vehicle with another vehicle? "
        "From which side of the image did that other vehicle approach? "
        "Return JSON with collision_frame and entry_side (LEFT or RIGHT). "
        f"Available frames: {[_frame_number(paths[index]) for index in overview]}.",
        max_new_tokens=64,
    ))
    coarse = _choice(coarse_result, "collision_frame", paths, overview, motion)
    side = str(coarse_result.get("entry_side", "")).upper().strip()
    if side not in {"LEFT", "RIGHT"}:
        LOG.warning("V2 entry_side invalid; using unvalidated LEFT fallback")
        side = "LEFT"
    collision_candidates, peaks = _fine_candidates(coarse, overview, scores)
    collision_result = _json_object(vlm.ask(
        [_sheet(paths, collision_candidates)],
        "These dashcam frames are ordered left to right, top to bottom. "
        "Which numbered frame first shows actual contact of the camera car with the other vehicle? "
        "Look for contact, not the later strongest shake. "
        f"Allowed frames: {[_frame_number(paths[index]) for index in collision_candidates]}. "
        "Return JSON with collision_frame.",
        max_new_tokens=48,
    ))
    collision = _choice(collision_result, "collision_frame", paths, collision_candidates, motion)

    # Entry has its own complete pre-contact chronology. It is not narrowed to
    # the coarse collision window and never consumes V1's already_entered flag.
    entry_candidates = _uniform_indices(0, collision, 12)
    entry_result = _json_object(vlm.ask(
        [_sheet(paths, entry_candidates, columns=4)],
        "These frames are chronological, left to right then top to bottom. "
        "The clip ends at a collision. When did the other collision vehicle first enter the camera car's driving lane? "
        "Entry is its first wheel touching the lane boundary. "
        "If it was already inside this lane at the first image, select the first image. "
        "At an intersection continue the camera car's lane boundaries forward. "
        f"Lane entry candidates: {[_frame_number(paths[index]) for index in entry_candidates]}. "
        "Return JSON with entry_frame only.",
        max_new_tokens=40,
    ))
    entry = _choice(entry_result, "entry_frame", paths, entry_candidates, 0)
    context = sorted(set([max(0, collision-2), collision, min(n-1, collision+2)]))
    space_result = _json_object(vlm.ask(
        [_sheet(paths, context, columns=3)],
        "These images show just before, during, and after a dashcam collision. "
        "At contact, is there usable road space for the camera car to continue or steer around the other vehicle? "
        "Account for nearby traffic, road edges, curbs and barriers. "
        "Return JSON with evasion_space: integer 1 if space exists, otherwise integer 0.",
        max_new_tokens=40,
    ))
    space = _integer(space_result.get("evasion_space"))
    if space not in {0, 1}:
        LOG.warning("V2 evasion_space invalid; using conservative zero fallback")
        space = 0
    prediction = dict(collision_frame=_frame_number(paths[collision]),
                      entry_frame=_frame_number(paths[entry]), entry_side=side, evasion_space=space)
    diagnostics = dict(version="v2_final_separate_entry_direction", calls=4, coarse=coarse_result,
                       collision=collision_result, entry=entry_result, space=space_result,
                       coarse_collision_frame=_frame_number(paths[coarse]),
                       motion_proposal=_frame_number(paths[motion]),
                       retained_motion_peaks=[_frame_number(paths[index]) for index in peaks],
                       collision_candidates=[_frame_number(paths[index]) for index in collision_candidates],
                       entry_candidates=[_frame_number(paths[index]) for index in entry_candidates])
    return prediction, diagnostics


def predict_stage2(data_dir, model_dir):
    from .vlm import LocalVLM
    data_dir, model_dir = Path(data_dir), Path(model_dir)
    image_root = data_dir/"images"
    if not image_root.is_dir():
        raise FileNotFoundError(f"Missing Stage2 image directory: {image_root}")
    rows = []
    with LocalVLM(model_dir/"vlm") as vlm:
        for folder in sorted(path for path in image_root.iterdir() if path.is_dir()):
            paths = sorted((path for path in folder.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS), key=_frame_number)
            numbers = [_frame_number(path) for path in paths]
            if len(numbers) != len(set(numbers)):
                raise ValueError(f"Duplicate Stage2 frame number in {folder.name}")
            paths, scores, _ = _motion_scan(paths)
            prediction, diagnostics = _predict_file(paths, scores, vlm)
            LOG.info("Stage2 V2 %s: %s", folder.name, json.dumps(diagnostics, ensure_ascii=False))
            rows.append(dict(ID=folder.name, **prediction))
    return pd.DataFrame(rows, columns=COLUMNS)
