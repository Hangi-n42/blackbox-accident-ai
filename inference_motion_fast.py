"""DACON 236753: Qwen 4B NF4, frozen motion collision, equivalent Stage 3 reductions."""
from pathlib import Path
import sys


def _runtime(model_dir):
    code = Path(model_dir).resolve().parent / 'stage2' / 'code'
    if code.is_dir() and str(code) not in sys.path:
        sys.path.insert(0, str(code))


def predict_stage1(data_dir, model_dir):
    _runtime(model_dir)
    from solution.stage1 import predict_stage1 as predict
    return predict(data_dir, model_dir)


def predict_stage2(data_dir, model_dir):
    _runtime(model_dir)
    from solution.stage2_motion_collision import predict_stage2 as predict
    return predict(data_dir, model_dir)


def predict_stage3(data_dir, model_dir):
    _runtime(model_dir)
    from solution.stage3_fast import predict_stage3 as predict
    return predict(data_dir, model_dir)
