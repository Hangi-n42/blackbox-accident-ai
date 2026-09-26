"""Frozen TPO full-frame/horizontal-flip average, no forensic inference."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from .stage1_v4 import sample_video, VIDEO_EXTS
from .stage1_tpo_merged import TPODetector


def predict_stage1(data_dir, model_dir):
    data, model = Path(data_dir), Path(model_dir)
    video_dir = data / 'stage1' / 'videos'
    if not video_dir.is_dir():
        video_dir = data / 'videos' if (data / 'videos').is_dir() else data
    stage_model = model / 'stage1' if (model / 'stage1').is_dir() else model
    config = json.loads((stage_model / 'config.json').read_text(encoding='utf-8'))
    if (config['mode'], config['sample_frames'], config['original_weight'],
        config['hflip_weight'], config['forensic_weight'], config['threshold']) != (
            'tpo_hflip', 12, .5, .5, 0., .5):
        raise ValueError('Unexpected V7 Stage1 configuration')
    detector = TPODetector(stage_model / 'tpo')
    rows = []
    try:
        for path in sorted(p for p in video_dir.iterdir() if p.suffix.lower() in VIDEO_EXTS):
            frames = sample_video(path, config['sample_frames'])
            flipped = [np.ascontiguousarray(frame[:, ::-1, :]) for frame in frames]
            scores = detector.score(frames + flipped)
            probability = .5 * float(scores[:len(frames)].mean()) + .5 * float(scores[len(frames):].mean())
            rows.append({'ID': path.stem, 'answer': 'RERECORDED' if probability >= .5 else 'ORIGINAL'})
    finally:
        detector.close()
    return pd.DataFrame(rows, columns=['ID', 'answer'])
