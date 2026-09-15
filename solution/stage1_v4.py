"""Stage1 candidate: fast seeks with bounded fault recovery; frozen models unchanged."""
from pathlib import Path
import json
import cv2
import numpy as np
import pandas as pd
from .stage1 import VIDEO_EXTS, extract_features, feature_probability


def _decode_pass(path, count, total_hint=None):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        cap.release()
        raise RuntimeError(f'Cannot open video: {path}')
    try:
        reported = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        if total_hint is None:
            total_hint = int(reported) if np.isfinite(reported) and reported > 0 else 0
        targets = (np.unique(np.linspace(0, total_hint - 1, min(count, total_hint))
                             .round().astype(int)).tolist() if total_hint > 0 else [])
        desired = set(targets)
        frames = []
        retained = []
        decoded = 0
        while cap.grab():
            if decoded in desired:
                ok, frame = cap.retrieve()
                if ok and frame is not None:
                    frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    retained.append(decoded)
            decoded += 1
        return frames, retained, decoded, total_hint, reported
    finally:
        cap.release()


def _sequential_fallback(path, count):
    frames, indices, actual, hint, reported = _decode_pass(path, count)
    if actual == 0:
        raise RuntimeError(f'No decodable frames: {path}')
    passes = 1
    if actual != hint:
        frames, indices, second_actual, _, _ = _decode_pass(path, count, actual)
        passes = 2
        if second_actual != actual:
            raise RuntimeError(f'Inconsistent sequential frame count: {path}')
    if not frames:
        raise RuntimeError(f'No decodable selected frames: {path}')
    return frames, {'method': 'sequential_decode_actual_count_verified',
                    'reported_frame_count': float(reported) if np.isfinite(reported) else None,
                    'decoded_frame_count': actual, 'selected_indices': indices, 'passes': passes,
                    'complete': len(frames) == min(count, actual)}


def sample_video_diagnostic(path, count=12):
    """Preserve normal seek cost; recover only observable faults within this video.

    A plausible wrong frame count or a backend lying about seek position cannot
    be detected here without decoding the full stream. Identical pixel frames
    are not treated as failed seeks because static video is valid input.
    """
    if not isinstance(count, (int, np.integer)) or count < 1:
        raise ValueError('count must be a positive integer')
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        cap.release()
        raise RuntimeError(f'Cannot open video: {path}')
    frames, indices, faults = [], [], []
    try:
        reported = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        if not np.isfinite(reported) or reported <= 0:
            faults.append('invalid_frame_count')
        else:
            total = int(reported)
            targets = np.unique(np.linspace(0, total - 1, min(count, total)).round().astype(int)).tolist()
            previous_position = None
            for index in targets:
                moved = cap.set(cv2.CAP_PROP_POS_FRAMES, index)
                ok, frame = cap.read()
                position = cap.get(cv2.CAP_PROP_POS_FRAMES)
                if not moved:
                    faults.append('seek_rejected')
                if not ok or frame is None:
                    faults.append('selected_read_failed')
                    continue
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                indices.append(index)
                if (not np.isfinite(position) or abs(position - (index + 1)) > .5
                        or (previous_position is not None and position <= previous_position)):
                    faults.append('reported_seek_position_mismatch')
                previous_position = position
    finally:
        cap.release()
    if not faults and frames:
        return frames, {'method': 'seek_positions_checked', 'reported_frame_count': float(reported),
                        'selected_indices': indices, 'passes': 0, 'complete': True, 'fallback_reasons': []}
    try:
        recovered, info = _sequential_fallback(path, count)
        info['fallback_reasons'] = sorted(set(faults))
        return recovered, info
    except RuntimeError as exc:
        # Preserve usable partial input if recovery cannot decode it; avoid
        # introducing a new whole-stage failure for a previously usable video.
        if frames:
            return frames, {'method': 'retained_partial_seek_frames', 'selected_indices': indices,
                            'complete': False, 'fallback_reasons': sorted(set(faults)),
                            'recovery_error': str(exc)}
        raise


def sample_video(path, count=12):
    return sample_video_diagnostic(path, count)[0]


def predict_stage1(data_dir, model_dir):
    # The feature, model, aggregation and decision code follows the frozen baseline.
    data, model = Path(data_dir), Path(model_dir)
    video_dir = data / 'stage1' / 'videos'
    if not video_dir.is_dir():
        video_dir = data / 'videos' if (data / 'videos').is_dir() else data
    stage_model = model / 'stage1' if (model / 'stage1').is_dir() else model
    config_path = stage_model / 'config.json'
    config = json.loads(config_path.read_text(encoding='utf-8')) if config_path.is_file() else {}
    artifact_path = stage_model / config.get('forensic_artifact', 'forensic.json')
    if not artifact_path.is_file():
        artifact_path = model / 'forensic.json'
    artifact = json.loads(artifact_path.read_text(encoding='utf-8'))
    detector = None
    if config.get('mode') == 'tpo_forensic_ensemble':
        from .stage1_tpo_merged import TPODetector
        detector = TPODetector(stage_model / 'tpo')
    results = []
    try:
        for path in sorted(p for p in video_dir.iterdir() if p.suffix.lower() in VIDEO_EXTS):
            frames = sample_video(path, artifact.get('sample_frames', 12))
            features, _ = extract_features(frames)
            probability = feature_probability(features, artifact)
            if detector is not None:
                weight = float(config.get('forensic_weight', .5))
                probability = weight * probability + (1 - weight) * float(detector.score(frames).mean())
            answer = 'RERECORDED' if probability >= config.get('threshold', artifact['threshold']) else 'ORIGINAL'
            results.append({'ID': path.stem, 'answer': answer})
    finally:
        if detector is not None:
            detector.close()
    return pd.DataFrame(results, columns=['ID', 'answer'])
