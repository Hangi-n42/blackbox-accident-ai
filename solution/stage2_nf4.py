"""Frozen Stage 2 V2 policy using a locally serialized Qwen 4B NF4 model."""
import json
import logging
from pathlib import Path
import pandas as pd
from .stage2_v2 import (
    COLUMNS, IMAGE_EXTENSIONS, _frame_number, _motion_scan, _predict_file,
)
from .vlm_candidate import CandidateVLM


def predict_stage2(data_dir, model_dir):
    image_root = Path(data_dir) / 'images'
    if not image_root.is_dir():
        raise FileNotFoundError(f'Missing Stage2 image directory: {image_root}')
    rows = []
    with CandidateVLM(Path(model_dir) / 'vlm', precision='nf4') as vlm:
        for folder in sorted(path for path in image_root.iterdir() if path.is_dir()):
            paths = sorted((path for path in folder.iterdir()
                            if path.suffix.lower() in IMAGE_EXTENSIONS), key=_frame_number)
            numbers = [_frame_number(path) for path in paths]
            if len(numbers) != len(set(numbers)):
                raise ValueError(f'Duplicate Stage2 frame number in {folder.name}')
            paths, scores, _ = _motion_scan(paths)
            prediction, diagnostics = _predict_file(paths, scores, vlm)
            logging.getLogger(__name__).info('Stage2 NF4 %s: %s', folder.name,
                                            json.dumps(diagnostics, ensure_ascii=False))
            rows.append(dict(ID=folder.name, **prediction))
    return pd.DataFrame(rows, columns=COLUMNS)
