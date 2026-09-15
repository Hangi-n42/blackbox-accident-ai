"""Frozen TPO validation. No adaptation to validation labels or hidden inputs."""
from pathlib import Path
import argparse
import json
import time
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, confusion_matrix
from solution.stage1 import sample_video
from solution.stage1_tpo import TPODetector
from evaluate_stage1 import jpeg, resize


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='model/stage1/tpo')
    parser.add_argument('--data', default='Baseline/data/stage1')
    parser.add_argument('--output', default='research/stage1/tpo_judgments.json')
    args = parser.parse_args()
    data = Path(args.data)
    labels = pd.read_csv(data / 'labels.csv')
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    report = {'scope': 'Frozen TPO pretrained detector on public simulated examples; actual dashcam recapture generalization unverified.',
              'threshold': .5, 'aggregation': 'arithmetic mean of within-video recapture probabilities',
              'samples': []}
    detector = TPODetector(args.model)
    try:
        for row in labels.itertuples():
            start = time.monotonic()
            frames = sample_video(data / row.path)
            variants = {'clean': frames, 'jpeg75': jpeg(frames, 75), 'half_resolution': resize(frames)}
            variants['center_crop'] = [f[int(f.shape[0]*.1):int(f.shape[0]*.9), int(f.shape[1]*.1):int(f.shape[1]*.9)] for f in frames]
            results = {}
            for name, altered in variants.items():
                scores = detector.score(altered)
                results[name] = {'probability': float(np.mean(scores)), 'frame_probabilities': scores.tolist(),
                                 'answer': 'RERECORDED' if np.mean(scores) >= .5 else 'ORIGINAL'}
            result = {'ID': row.ID, 'label': row.label, 'results': results, 'seconds': time.monotonic() - start}
            report['samples'].append(result)
            output.write_text(json.dumps(report, indent=2), encoding='utf-8')
            print(json.dumps({'ID': row.ID, 'label': row.label, 'probabilities': {k: v['probability'] for k, v in results.items()}}), flush=True)
    finally:
        detector.close()
    report['metrics'] = {}
    for variant in ['clean', 'jpeg75', 'half_resolution', 'center_crop']:
        truth = [r['label'] for r in report['samples']]
        pred = [r['results'][variant]['answer'] for r in report['samples']]
        report['metrics'][variant] = {'macro_f1': f1_score(truth, pred, labels=['ORIGINAL', 'RERECORDED'], average='macro'),
                                      'confusion_matrix': confusion_matrix(truth, pred, labels=['ORIGINAL', 'RERECORDED']).tolist()}
    output.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report['metrics'], indent=2))


if __name__ == '__main__':
    main()
