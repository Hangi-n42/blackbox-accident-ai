"""Unadopted four-call global/local entry hypothesis, no ROI or status schema."""
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from . import stage2_v2 as base
from .vlm_candidate import CandidateVLM

COLUMNS=base.COLUMNS
TOKENS=(64,40,48,40)


def _select(result, paths, offered, fallback):
    value=result.get("entry_frame")
    if type(value) is int:
        selected=next((i for i in offered if base._frame_number(paths[i])==value),None)
        if selected is not None:
            return selected,True
    return fallback,False


def _fine_indices(n, overview, coarse, coarse_valid):
    if not coarse_valid:
        return base._uniform_indices(0,n-1,12)
    previous=[i for i in overview if i<coarse]
    following=[i for i in overview if i>coarse]
    low=previous[-1] if previous else 0
    high=following[0] if following else n-1
    # Nine interval samples plus mandatory chosen coarse and two global anchors.
    return sorted(set(base._uniform_indices(low,high,9)) | {coarse,0,n-1})


def _predict_file(paths,scores,vlm):
    if not paths or len(paths)!=len(scores):
        raise ValueError("Nonempty ordered paths and matching scores required")
    numbers=[base._frame_number(p) for p in paths]
    if numbers!=sorted(set(numbers)):
        raise ValueError("Original frame numbers must be distinct and ascending")
    n=len(paths)
    motion=int(np.argmax(scores))
    overview=base._uniform_indices(0,n-1,10)
    # Exact frozen V3 first prompt, renderer, token budget and side fallback.
    first=base._json_object(vlm.ask([base._sheet(paths,overview)],
        "These are frames from one dashcam clip, ordered left to right, top to bottom. "
        "Which numbered frame first shows physical contact of the camera vehicle with another vehicle? "
        "From which side of the image did that other vehicle approach? "
        "Return JSON with collision_frame and entry_side (LEFT or RIGHT). "
        f"Available frames: {[base._frame_number(paths[index]) for index in overview]}.",
        max_new_tokens=TOKENS[0]))
    side=str(first.get("entry_side","")).upper().strip()
    side_fallback=side not in {"LEFT","RIGHT"}
    if side_fallback:side="LEFT"
    global_candidates=base._uniform_indices(0,n-1,12)
    coarse_result=base._json_object(vlm.ask([base._sheet(paths,global_candidates,columns=4)],
        "These frames cover the whole dashcam clip in chronological order, left to right then top to bottom. "
        "When did the other collision vehicle first enter the camera car's driving lane? "
        "Entry is its first wheel touching the lane boundary. "
        "If it was already inside this lane at the original clip start, select the original first frame. "
        "At an intersection continue the camera car's lane boundaries forward. "
        f"Original first frame: {numbers[0]}. Lane entry candidates: {[numbers[i] for i in global_candidates]}. "
        "Return JSON with entry_frame only.",max_new_tokens=TOKENS[1]))
    coarse,coarse_valid=_select(coarse_result,paths,global_candidates,0)
    fine_candidates=_fine_indices(n,global_candidates,coarse,coarse_valid)
    fine_result=base._json_object(vlm.ask([base._sheet(paths,fine_candidates,columns=4)],
        "These original dashcam frames are chronological, left to right then top to bottom. "
        "They contain a local search window plus the original clip start and end anchors. "
        "The first image of the local window may not be the start of the full clip. "
        "When did the other collision vehicle FIRST enter the camera car's driving lane? "
        "Select the first wheel touching the lane boundary, not a later re-entry. "
        "If it was already inside at the ORIGINAL clip start, select that original first frame. "
        "At intersections continue the camera car's lane boundaries forward. "
        f"Original first frame: {numbers[0]}. Lane entry candidates: {[numbers[i] for i in fine_candidates]}. "
        "Return JSON with entry_frame only.",max_new_tokens=TOKENS[2]))
    entry,fine_valid=_select(fine_result,paths,fine_candidates,coarse)
    origin=("fine_valid_selected" if fine_valid else "coarse_valid_retained" if coarse_valid else "invalid_fallback_first")
    context=sorted(set([max(0,motion-2),motion,min(n-1,motion+2)]))
    # Frozen V3 space question, but explicitly motion-based input context.
    space_result=base._json_object(vlm.ask([base._sheet(paths,context,columns=3)],
        "These images show just before, during, and after a dashcam collision. "
        "At contact, is there usable road space for the camera car to continue or steer around the other vehicle? "
        "Account for nearby traffic, road edges, curbs and barriers. "
        "Return JSON with evasion_space: integer 1 if space exists, otherwise integer 0.",
        max_new_tokens=TOKENS[3]))
    space=base._integer(space_result.get("evasion_space"))
    space_fallback=space not in (0,1)
    if space_fallback:space=0
    pred=dict(collision_frame=numbers[motion],entry_frame=numbers[entry],entry_side=side,evasion_space=space)
    diag=dict(version="v5_simple_global_local_motion_space",calls=4,first=first,
        first_collision_ignored_for_all_later_inputs=True,motion_frame=numbers[motion],
        coarse=coarse_result,fine=fine_result,space=space_result,
        coarse_candidates=[numbers[i] for i in global_candidates],fine_candidates=[numbers[i] for i in fine_candidates],
        coarse_frame=numbers[coarse],coarse_valid=coarse_valid,fine_valid=fine_valid,entry_origin=origin,
        valid_model_selected_first=entry==0 and (fine_valid or coarse_valid),
        semantic_observation_not_verified=True,entry_after_motion_collision=entry>motion,
        space_context=[numbers[i] for i in context],space_context_policy="final_motion_argmax_plus_minus_two_indices",
        space_context_not_equivalent_to_V3_internal_VLM_collision=True,
        side_fallback=side_fallback,space_fallback=space_fallback)
    return pred,diag


def predict_stage2(data_dir,model_dir):
    image_root=Path(data_dir)/"images"
    if not image_root.is_dir():raise FileNotFoundError(f"Missing Stage2 image directory: {image_root}")
    rows=[]
    with CandidateVLM(Path(model_dir)/"vlm",precision="nf4") as vlm:
        for folder in sorted(p for p in image_root.iterdir() if p.is_dir()):
            paths=sorted((p for p in folder.iterdir() if p.suffix.lower() in base.IMAGE_EXTENSIONS),key=base._frame_number)
            numbers=[base._frame_number(p) for p in paths]
            if len(numbers)!=len(set(numbers)):raise ValueError(f"Duplicate Stage2 frame number in {folder.name}")
            paths,scores,_=base._motion_scan(paths)
            pred,diag=_predict_file(paths,scores,vlm)
            logging.getLogger(__name__).info("Stage2 V5 simple %s: %s",folder.name,json.dumps(diag,ensure_ascii=False))
            rows.append(dict(ID=folder.name,**pred))
    return pd.DataFrame(rows,columns=COLUMNS)
