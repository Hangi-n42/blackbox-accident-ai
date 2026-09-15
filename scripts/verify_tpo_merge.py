"""Verify the merged detector against recorded original-runtime public scores."""
from pathlib import Path
import argparse
import json
import sys
import time
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from solution.stage1 import sample_video
from solution.stage1_tpo_merged import TPODetector
from evaluate_stage1 import jpeg, resize


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default=None)
    parser.add_argument('--tolerance', type=float, default=5e-5)
    args = parser.parse_args()
    reference_path = ROOT / 'research/stage1/tpo_judgments.json'
    reference = json.loads(reference_path.read_text(encoding='utf-8'))
    by_id = {sample['ID']: sample for sample in reference['samples']}
    labels = pd.read_csv(ROOT / 'Baseline/data/stage1/labels.csv')
    detector = TPODetector(ROOT / 'model/stage1/tpo', device=args.device)
    records = []
    try:
        for row in labels.itertuples():
            frames = sample_video(ROOT / 'Baseline/data/stage1' / row.path)
            variants = {'clean': frames, 'jpeg75': jpeg(frames, 75), 'half_resolution': resize(frames),
                        'center_crop': [f[int(f.shape[0]*.1):int(f.shape[0]*.9), int(f.shape[1]*.1):int(f.shape[1]*.9)] for f in frames]}
            for variant, images in variants.items():
                start = time.monotonic()
                actual = detector.score(images)
                expected = np.asarray(by_id[row.ID]['results'][variant]['frame_probabilities'])
                if actual.shape != expected.shape:
                    raise ValueError('Original and merged runtime evaluated different frame counts')
                record = {'ID': row.ID, 'variant': variant, 'frames': len(actual),
                          'max_abs_probability_difference': float(np.max(np.abs(actual - expected))),
                          'original_video_probability': float(expected.mean()),
                          'merged_video_probability': float(actual.mean()),
                          'video_answer_unchanged': bool((actual.mean() >= .5) == (expected.mean() >= .5)),
                          'seconds': time.monotonic() - start}
                records.append(record)
            print(json.dumps({'ID': row.ID, 'max_abs_difference': max(r['max_abs_probability_difference'] for r in records if r['ID'] == row.ID)}), flush=True)
    finally:
        detector.close()
    maximum = max(row['max_abs_probability_difference'] for row in records)
    unchanged = all(row['video_answer_unchanged'] for row in records)
    report = {'reference': str(reference_path.relative_to(ROOT)),
              'reference_scope': 'Previously measured original TPO wrapper, identical local source weights and deterministic frame preprocessing',
              'device': detector.device, 'tolerance': args.tolerance, 'frames_compared': sum(r['frames'] for r in records),
              'max_abs_probability_difference': maximum, 'all_video_answers_unchanged': unchanged,
              'passed': maximum <= args.tolerance and unchanged, 'samples': records}
    target = ROOT / 'research/stage1/merged_equivalence.json'
    target.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps({k: v for k, v in report.items() if k != 'samples'}, indent=2))
    if not report['passed']:
        raise RuntimeError('Merged detector does not match the original within the declared tolerance')


if __name__ == '__main__':
    main()
