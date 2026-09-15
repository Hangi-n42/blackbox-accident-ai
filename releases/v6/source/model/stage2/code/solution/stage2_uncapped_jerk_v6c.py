"""V5 four-call policy; remove only the jerk robust-score upper cap for final collision."""
import hashlib
import json
import logging
from pathlib import Path

import numpy as np

from . import stage2_motion_collision as baseline
from . import stage2 as primitives

COLUMNS = baseline.COLUMNS
LOG = logging.getLogger(__name__)


def _scores_from_features(features):
    """Original expression preserved; only the new branch's jerk cap differs."""
    base_scores = (primitives._robust_scale(features[:, 0])
                   + .6 * primitives._robust_scale(features[:, 1])
                   + .25 * primitives._robust_scale(features[:, 2]))
    jerk = features[:, 0]
    median = np.median(jerk)
    scale = np.median(np.abs(jerk - median)) * 1.4826
    uncapped = np.maximum((jerk - median) / max(float(scale), 1e-3), 0)
    new_scores = (uncapped + .6 * primitives._robust_scale(features[:, 1])
                  + .25 * primitives._robust_scale(features[:, 2]))
    base_scores[0] = 0
    new_scores[0] = 0
    return base_scores, new_scores


def _dual_motion_scan(paths):
    """Mirror V5 stage2._motion_scan feature extraction; one decode/flow pass."""
    valid, features = [], []
    previous = None
    previous_shift = np.zeros(2, dtype=np.float32)
    for path in paths:
        try:
            image = primitives._read_rgb(path)
        except (OSError, ValueError) as error:
            LOG.warning('Skipping unreadable Stage2 frame %s: %s', path, error)
            continue
        gray = primitives.cv2.cvtColor(np.asarray(image.resize((160, 96))), primitives.cv2.COLOR_RGB2GRAY)
        valid.append(path)
        if previous is None:
            features.append((0., 0., 0.))
        else:
            flow = primitives.cv2.calcOpticalFlowFarneback(previous, gray, None, .5, 2, 11, 2, 5, 1.1, 0)
            core = flow[8:76, 8:152]
            shift = np.median(core.reshape(-1, 2), axis=0)
            residual = np.linalg.norm(core - shift, axis=2)
            jerk = float(np.linalg.norm(shift - previous_shift))
            residual_change = float(np.percentile(residual, 90))
            appearance_change = float(np.mean(np.abs(gray.astype(np.float32) - previous)))
            features.append((jerk, residual_change, appearance_change))
            previous_shift = shift
        previous = gray
    if not valid:
        raise ValueError('Stage2 folder contains no decodable numbered images')
    features = np.asarray(features, dtype=np.float32)
    base_scores, new_scores = _scores_from_features(features)
    return valid, base_scores, new_scores, features


def _score_hash(scores):
    values = np.ascontiguousarray(scores)
    return hashlib.sha256(values.tobytes()).hexdigest()


def apply_collision(paths, prediction, diagnostics, new_scores, base_scores=None):
    """Pure cached-output helper; optionally bind base-score bytes when supplied."""
    numbers = [primitives._frame_number(path) for path in paths]
    new_scores = np.asarray(new_scores)
    if not numbers or len(numbers) != len(set(numbers)) or new_scores.shape != (len(numbers),) or not np.isfinite(new_scores).all():
        raise ValueError('Expected unique valid paths and one finite new score per path')
    if base_scores is not None:
        base_scores = np.asarray(base_scores)
        if base_scores.shape != new_scores.shape or not np.isfinite(base_scores).all():
            raise ValueError('Base scores must match valid paths')
    old = prediction['collision_frame']
    if type(old) is not int or old not in numbers:
        raise ValueError('Baseline collision is not a valid original frame')
    new = numbers[int(np.argmax(new_scores))]
    result = {**prediction, 'collision_frame': new}
    detail = dict(version='uncapped_jerk_v6c', old_collision_frame=old, new_collision_frame=new,
                  base_score_sha256=None if base_scores is None else _score_hash(base_scores),
                  base_score_hash_available=base_scores is not None,
                  new_score_sha256=_score_hash(new_scores),
                  base_score_dtype=None if base_scores is None else str(base_scores.dtype), new_score_dtype=str(new_scores.dtype),
                  score_hash_method='SHA256 of contiguous array bytes in recorded dtype; valid-path order',
                  changed_target_only=all(result[k] == value for k, value in prediction.items() if k != 'collision_frame'),
                  collision_changed=old != new, baseline_context_for_other_fields_retained=True,
                  modification='Jerk lower truncation retained, jerk upper cap 10 removed; other feature scaling and weights unchanged')
    return result, {**diagnostics, 'uncapped_jerk': detail}


def _predict_file(paths, base_scores, new_scores, vlm):
    prediction, diagnostics = baseline._predict_file(paths, base_scores, vlm)
    return apply_collision(paths, prediction, diagnostics, new_scores, base_scores)


def predict_stage2(data_dir, model_dir):
    image_root = Path(data_dir) / 'images'
    if not image_root.is_dir():
        raise FileNotFoundError(f'Missing Stage2 image directory: {image_root}')
    rows = []
    with baseline.CandidateVLM(Path(model_dir) / 'vlm', precision='nf4') as vlm:
        for folder in sorted(path for path in image_root.iterdir() if path.is_dir()):
            paths = sorted((path for path in folder.iterdir()
                            if path.suffix.lower() in primitives.IMAGE_EXTENSIONS), key=primitives._frame_number)
            numbers = [primitives._frame_number(path) for path in paths]
            if len(numbers) != len(set(numbers)):
                raise ValueError(f'Duplicate Stage2 frame number in {folder.name}')
            valid, base_scores, new_scores, _ = _dual_motion_scan(paths)
            prediction, diagnostics = _predict_file(valid, base_scores, new_scores, vlm)
            LOG.info('Stage2 uncapped jerk %s: %s', folder.name, json.dumps(diagnostics, ensure_ascii=False))
            rows.append(dict(ID=folder.name, **prediction))
    return baseline.pd.DataFrame(rows, columns=COLUMNS)
