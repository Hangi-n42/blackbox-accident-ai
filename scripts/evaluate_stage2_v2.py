"""Use the shared public evaluator with V2 predictor, preserving V1 files/results."""
from pathlib import Path
import hashlib
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import evaluate_stage2_vlm as evaluator
from solution.stage2_v2 import _predict_file

evaluator._predict_file = _predict_file
evaluator.CALL_NAMES = ("overview", "collision_refinement", "entry_refinement", "evasion_space")
original_write_json = evaluator.write_json


def write_json(path, value):
    if isinstance(value, dict) and "selected_ids" in value:
        value["predictor_variant"] = "v2_final_separate_entry_direction"
        value["stage2_v2_source_sha256"] = hashlib.sha256((ROOT/"solution/stage2_v2.py").read_bytes()).hexdigest()
    original_write_json(path, value)


evaluator.write_json = write_json

if __name__ == "__main__":
    evaluator.main()
