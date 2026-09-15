"""V6 contact scores with unsupported first shift-difference contribution removed."""
import json,logging
from pathlib import Path
import numpy as np
from . import stage2_uncapped_jerk_v6c as v6
LOG=logging.getLogger(__name__)
def corrected_scores(features,new_scores):
    features=np.asarray(features,dtype=np.float32)
    scores=np.asarray(new_scores).copy()
    if features.shape!=(len(scores),3) or not len(scores) or not np.isfinite(features).all() or not np.isfinite(scores).all():
        raise ValueError('Expected finite frozen motion features and scores')
    if len(scores)>1:
        # Both components use the real first frame pair. A difference between two
        # observed flow shifts is not defined until input index2.
        residual=v6.primitives._robust_scale(features[:,1])
        appearance=v6.primitives._robust_scale(features[:,2])
        scores[1]=(.6*residual+.25*appearance)[1]
    return scores
def predict_stage2(data_dir,model_dir):
    root=Path(data_dir)/'images'
    if not root.is_dir():raise FileNotFoundError(f'Missing Stage2 images: {root}')
    rows=[]
    with v6.baseline.CandidateVLM(Path(model_dir)/'vlm',precision='nf4') as model:
        for folder in sorted(p for p in root.iterdir() if p.is_dir()):
            paths=sorted((p for p in folder.iterdir() if p.suffix.lower() in v6.primitives.IMAGE_EXTENSIONS),key=v6.primitives._frame_number)
            numbers=[v6.primitives._frame_number(p) for p in paths]
            if len(numbers)!=len(set(numbers)):raise ValueError('Duplicate original Stage2 frame number')
            valid,base,scores,features=v6._dual_motion_scan(paths)
            prediction,diagnostics=v6._predict_file(valid,base,scores,model)
            corrected=corrected_scores(features,scores)
            old=prediction['collision_frame'];prediction['collision_frame']=v6.primitives._frame_number(valid[int(np.argmax(corrected))])
            LOG.info('Stage2 V7 motion initialization %s: %s',folder.name,json.dumps(dict(old_collision=old,new_collision=prediction['collision_frame'],changed_score_indices=[1] if len(valid)>1 else [],score_sha256=v6._score_hash(corrected))))
            rows.append(dict(ID=folder.name,**prediction))
    return v6.baseline.pd.DataFrame(rows,columns=v6.COLUMNS)
