"""One local contact-verification call after the frozen four-call baseline.

Only collision_frame may change. The other fields retain the original baseline
context and values; this module does not claim that context is correct.
"""
import json
import logging
from pathlib import Path

from . import stage2_motion_collision as baseline

COLUMNS = baseline.COLUMNS


def _reject_nonfinite_json(value):
    raise ValueError(f'Nonstandard JSON constant: {value}')


def _window_indices(center, length):
    start, end = max(0, center - 10), min(length - 1, center + 10)
    chosen = baseline.base._uniform_indices(start, end, 9)
    if center not in chosen:
        # Keep at most nine tiles; nearest non-center, earlier index on ties.
        removed = min(chosen, key=lambda i: (abs(i - center), i))
        chosen = [i for i in chosen if i != removed] + [center]
    return sorted(set(chosen))


def refine_collision(paths, prediction, diagnostics, vlm):
    """Refine a saved baseline result without repeating its first four calls."""
    numbers = [baseline.base._frame_number(path) for path in paths]
    if not numbers or numbers != sorted(set(numbers)):
        raise ValueError('Expected nonempty, unique, increasing original frame numbers')
    internal = diagnostics['collision_replacement']['base_collision_frame']
    old = prediction['collision_frame']
    if type(internal) is not int or type(old) is not int or internal not in numbers or old not in numbers:
        raise ValueError('Both baseline centers must be original numbers in valid paths')
    centers = [numbers.index(internal), numbers.index(old)]
    windows = [_window_indices(center, len(paths)) for center in centers]
    indices = sorted(set(windows[0]) | set(windows[1]), key=lambda i: numbers[i])
    offered = [numbers[i] for i in indices]
    prompt = (
        'These dashcam frames are chronological, left to right, top to bottom. '
        'They come from two local time windows and do not show the whole clip; '
        'there may be a time gap between the windows. '
        'Which offered numbered frame first shows actual physical contact between '
        'the camera vehicle and the other vehicle? '
        'Do not select mere close proximity, braking, or camera shake without visible contact evidence. '
        'If the first physical contact cannot be determined from these images, return null. '
        f'Allowed frames: {offered}. '
        'Return JSON only with collision_frame: an offered integer frame number, or null.'
    )
    raw = vlm.ask([baseline.base._sheet(paths, indices, columns=5)], prompt, max_new_tokens=48)
    parsed = None
    parse_error = None
    try:
        parsed = json.loads(raw, parse_constant=_reject_nonfinite_json)
    except (ValueError, TypeError) as error:
        parse_error = type(error).__name__
    value = parsed.get('collision_frame') if isinstance(parsed, dict) else None
    accepted = type(value) is int and value in offered
    if parse_error:
        reason = 'invalid_json'
    elif not isinstance(parsed, dict):
        reason = 'not_json_object'
    elif type(value) is not int:
        reason = 'collision_not_strict_integer'
    elif value not in offered:
        reason = 'collision_not_offered'
    else:
        reason = None
    new = value if accepted else old
    result = {**prediction, 'collision_frame': new}
    detail = dict(
        version='contact_verify_v6a', added_calls=1, max_new_tokens=48,
        window_half_width_in_valid_path_indices=10, max_per_window=9,
        centers_original_frames=[internal, old], centers_valid_path_indices=centers,
        window_indices=windows, window_original_frames=[[numbers[i] for i in w] for w in windows],
        offered_original_frames=offered, offered_valid_path_indices=indices,
        raw_output=raw, parsed_output=parsed, parse_error=parse_error,
        accepted=accepted, fallback=not accepted, fallback_reason=reason,
        old_collision_frame=old, new_collision_frame=new,
        other_prediction_fields_unchanged=all(result[k] == v for k, v in prediction.items() if k != 'collision_frame'),
        other_fields_keep_baseline_context=True,
    )
    return result, {**diagnostics, 'contact_verification': detail}


def _predict_file(paths, scores, vlm):
    prediction, diagnostics = baseline._predict_file(paths, scores, vlm)
    return refine_collision(paths, prediction, diagnostics, vlm)


def predict_stage2(data_dir, model_dir):
    image_root = Path(data_dir) / 'images'
    if not image_root.is_dir():
        raise FileNotFoundError(f'Missing Stage2 image directory: {image_root}')
    rows = []
    with baseline.CandidateVLM(Path(model_dir) / 'vlm', precision='nf4') as vlm:
        for folder in sorted(path for path in image_root.iterdir() if path.is_dir()):
            paths = sorted((path for path in folder.iterdir()
                            if path.suffix.lower() in baseline.base.IMAGE_EXTENSIONS), key=baseline.base._frame_number)
            numbers = [baseline.base._frame_number(path) for path in paths]
            if len(numbers) != len(set(numbers)):
                raise ValueError(f'Duplicate Stage2 frame number in {folder.name}')
            paths, scores, _ = baseline.base._motion_scan(paths)
            prediction, diagnostics = _predict_file(paths, scores, vlm)
            logging.getLogger(__name__).info('Stage2 contact verify %s: %s', folder.name,
                                             json.dumps(diagnostics, ensure_ascii=False))
            rows.append(dict(ID=folder.name, **prediction))
    return baseline.pd.DataFrame(rows, columns=COLUMNS)
