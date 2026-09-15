"""Frozen-model image-subset diagnostic. Does not manufacture or evaluate videos."""
from pathlib import Path
import sys
import hashlib
import json
import time
import csv
from collections import Counter
import numpy as np
from PIL import Image
from sklearn.metrics import confusion_matrix, f1_score

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from solution.stage1 import extract_features, feature_probability
from solution.stage1_tpo_merged import TPODetector

BASE=Path(__file__).resolve().parent
OUT=BASE/'dlc_subset'
MODEL=REPO/'model/stage1'


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()


def main():
    cfg=json.loads((MODEL/'config.json').read_text())
    artifact=MODEL/cfg['forensic_artifact']
    forensic=json.loads(artifact.read_text())
    assert cfg['forensic_weight']==.5 and cfg['threshold']==.5
    plan=json.loads((OUT/'selection_plan.json').read_text())
    acquired=json.loads((OUT/'acquisition_manifest.json').read_text())
    indexed={x['archive_path']:x for x in acquired}
    assert len(indexed)==sum(len(v['frames']) for v in plan['videos'])
    for x in acquired:
        assert sha(BASE/x['local_path'])==x['sha256']
    frozen={'evaluation_unit':'one ordered distributed-JPEG subset per source video; not decoded-video evaluation',
            'selection_plan_sha256':sha(OUT/'selection_plan.json'),
            'artifact_sha256':{str(p.relative_to(REPO)):sha(p) for p in [MODEL/'config.json',artifact,MODEL/'tpo/merged_visual.pt',REPO/'solution/stage1.py',REPO/'solution/stage1_tpo_merged.py']},
            'configuration':cfg, 'thresholds':{'tpo':.5,'forensic':.5,'blend':.5},
            'training_or_tuning':False}
    (OUT/'frozen_diagnostic_manifest.json').write_text(json.dumps(frozen,indent=2),encoding='utf-8')
    detector=TPODetector(MODEL/'tpo')
    rows=[];start=time.monotonic()
    try:
        for v in plan['videos']:
            frames=[np.asarray(Image.open(BASE/indexed[x['path']]['local_path']).convert('RGB')) for x in v['frames']]
            features,_=extract_features(frames)
            fp=feature_probability(features,forensic)
            tp=float(detector.score(frames).mean())
            row={k:v[k] for k in ['document_id','video_id','label','camera','condition_or_display']}
            row.update(n_images=len(frames),image_shapes=[list(x.shape) for x in frames],
                       y_true=int(v['label']=='re'),tpo=tp,forensic=fp,blend=.5*tp+.5*fp)
            rows.append(row)
            (OUT/'frozen_predictions.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')
            print(v['video_id'],v['camera'], 'TPO',round(tp,4),'forensic',round(fp,4),'blend',round(row['blend'],4),flush=True)
    finally:
        detector.close()
    def metrics(selected):
        y=[r['y_true'] for r in selected]
        return {mode:{'macro_f1':float(f1_score(y,[int(r[mode]>=.5) for r in selected],average='macro',zero_division=0)),
                      'confusion_rows_original_recaptured':confusion_matrix(y,[int(r[mode]>=.5) for r in selected],labels=[0,1]).tolist()}
                for mode in ['tpo','forensic','blend']}
    result={'scope':frozen['evaluation_unit'],'n_documents':len(plan['documents']),'n_video_subsets':len(rows),
            'n_images':sum(r['n_images'] for r in rows),'seconds':time.monotonic()-start,
            'image_resolution_counts_by_label':{label:dict(Counter(f'{s[1]}x{s[0]}' for r in rows if r['label']==label for s in r['image_shapes'])) for label in ['or','re']},
            'overall':metrics(rows), 'by_camera':{c:metrics([r for r in rows if r['camera']==c]) for c in ['iphone','android']},
            'by_document':{d:metrics([r for r in rows if r['document_id']==d]) for d in plan['documents']},
            'limitation':'Six document-content groups from an ID-document domain, distributed JPEG sampling and image rescaling differ from dashcam video decoding. No competition-score extrapolation.'}
    (OUT/'frozen_metrics.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result['overall'],indent=2),flush=True)


if __name__=='__main__':main()
