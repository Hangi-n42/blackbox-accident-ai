"""Research candidate: preserve all four VLM calls; remove final auxiliary score sum."""
import json
import logging
from pathlib import Path

import numpy as np
from solution import stage2_uncapped_jerk_v6c as reference

COLUMNS = reference.COLUMNS


def scores_from_features(features):
    features = np.asarray(features)
    if features.ndim != 2 or features.shape[1] != 3 or not len(features) or not np.isfinite(features).all():
        raise ValueError('Expected nonempty finite motion features with three columns')
    jerk = features[:, 0]
    median = np.median(jerk)
    scale = np.median(np.abs(jerk - median)) * 1.4826
    scores = np.maximum((jerk - median) / max(float(scale), 1e-3), 0)
    scores[0] = 0
    return scores


def predict_file(paths, base_scores, new_scores, features, vlm):
    # Four calls still use the original base scores, candidates, and contact context.
    before, diagnostics = reference._predict_file(paths, base_scores, new_scores, vlm)
    scores = scores_from_features(features)
    prediction, detail = reference.apply_collision(paths, before, {}, scores, base_scores)
    detail = detail['uncapped_jerk']
    detail.update(version='research_final_jerk_only',
                  modification='Remove residual and appearance only from final score; no initialization or VLM changes')
    return prediction, {**diagnostics, 'baseline_prediction': before, 'contact_ablation': detail}


def predict_stage2(data_dir, model_dir, *, trace_dir=None):
    image_root = Path(data_dir) / 'images'
    if not image_root.is_dir():
        raise FileNotFoundError(f'Missing Stage2 image directory: {image_root}')
    rows = []
    with reference.baseline.CandidateVLM(Path(model_dir) / 'vlm', precision='nf4') as vlm:
        for folder in sorted(path for path in image_root.iterdir() if path.is_dir()):
            paths = sorted((path for path in folder.iterdir()
                            if path.suffix.lower() in reference.primitives.IMAGE_EXTENSIONS),
                           key=reference.primitives._frame_number)
            numbers = [reference.primitives._frame_number(path) for path in paths]
            if len(numbers) != len(set(numbers)):
                raise ValueError(f'Duplicate Stage2 frame number in {folder.name}')
            valid, base, new, features = reference._dual_motion_scan(paths)
            prediction, diagnostics = predict_file(valid, base, new, features, vlm)
            logging.getLogger(__name__).info('Stage2 research jerk-only %s: %s', folder.name, json.dumps(diagnostics))
            if trace_dir is not None:
                Path(trace_dir).mkdir(parents=True, exist_ok=True)
                with (Path(trace_dir) / (folder.name + '.json')).open('x') as stream:
                    json.dump(dict(ID=folder.name, prediction=prediction, diagnostics=diagnostics), stream, indent=2)
            rows.append(dict(ID=folder.name, **prediction))
    return reference.baseline.pd.DataFrame(rows, columns=COLUMNS)
