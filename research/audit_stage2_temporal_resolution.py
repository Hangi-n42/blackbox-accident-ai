"""Measure the current entry candidate grid on explicitly synthetic timelines.

This is a numerical coverage diagnostic, not an accuracy or dataset statistic.
No video, label, model, or hidden evaluation input is accessed.
"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from solution.stage2 import _uniform_indices


def main():
    rows = []
    for fps in (10, 30):
        for seconds in (5, 30, 60):
            count = fps * seconds
            candidates = np.asarray(_uniform_indices(0, count - 1, 12))
            distance = np.min(np.abs(np.arange(count)[:, None] - candidates), axis=1)
            rows.append({
                'synthetic_fps': fps,
                'synthetic_frames_before_and_including_contact': count,
                'synthetic_duration_seconds': seconds,
                'entry_candidate_count': len(candidates),
                'candidate_indices': candidates.tolist(),
                'maximum_nearest_candidate_error_seconds': float(distance.max() / fps),
                'possible_target_frames_with_candidate_within_0_3_seconds': int((distance <= 0.3 * fps).sum()),
                'all_possible_target_frames': count,
                'interpretation': 'Coverage of a synthetic time grid; NOT expected accuracy. Actual target distribution and evaluation FPS are unknown.',
            })
    source = ROOT / 'solution/stage2_v2.py'
    report = {
        'scope': 'Current one-pass 12-frame entry candidate grid, assuming collision already correct and at timeline end.',
        'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
        'synthetic_cases': rows,
        'conclusion': 'Long pre-contact timelines can contain target frames farther than the metric tolerance from every offered entry candidate. A more accurate VLM alone cannot remove this grid limitation.',
        'not_proven': 'This does not establish that the actual evaluation inputs have long timelines, or that quantization caused the observed Stage 2 score.',
    }
    output = ROOT / 'research/stage2_temporal_resolution.json'
    output.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
