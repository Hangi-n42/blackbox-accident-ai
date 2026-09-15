"""V6 policy unchanged; external runner owns the separate official8B checkpoint."""
from solution import stage2_uncapped_jerk_v6c as v6
def predict_from_scan(paths,base_scores,new_scores,vlm):
    return v6._predict_file(paths,base_scores,new_scores,vlm)
