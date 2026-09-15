"""Verify the intended packaged ablation against the submitted V2 outputs."""
import json
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def main():
    directory = ROOT / 'artifacts/submissions'
    old = directory / 'verify_v2_nf4_results'
    new = directory / 'verify_v3_motion_fast_results'
    assert json.loads((new / 'report.json').read_text())['status'] == 'PASS'
    result = {'scope': 'Public examples and packaged change isolation; not Private performance.'}
    for stage in (1, 3):
        left, right = [pd.read_csv(d / f'stage{stage}.csv') for d in (old, new)]
        pd.testing.assert_frame_equal(left, right)
        result[f'stage{stage}_identical_rows'] = len(left)
    left, right = [pd.read_csv(d / 'stage2.csv').sort_values('ID').reset_index(drop=True)
                   for d in (old, new)]
    retained = ['ID', 'entry_frame', 'evasion_space', 'entry_side']
    pd.testing.assert_frame_equal(left[retained], right[retained])
    # This expected vector is a historical diagnostic, never imported by inference.
    expected = [35, 47, 33, 42, 33]
    assert right.collision_frame.tolist() == expected
    labels = pd.read_csv(ROOT / 'Baseline/data/stage2/labels.csv').sort_values('ID')
    truth = labels.t_collision.to_numpy()
    assert left.ID.tolist() == labels.ID.tolist() == right.ID.tolist()
    result['stage2_retained_fields_identical'] = retained
    result['stage2_collision_before'] = left.collision_frame.tolist()
    result['stage2_collision_after'] = right.collision_frame.tolist()
    result['public_10hz_collision_correct_within_3_frames'] = {
        'before': int((abs(left.collision_frame.to_numpy() - truth) <= 3).sum()),
        'after': int((abs(right.collision_frame.to_numpy() - truth) <= 3).sum()),
        'total': len(truth),
    }
    result['entry_after_new_collision_ids'] = right.loc[
        right.entry_frame > right.collision_frame, 'ID'].tolist()
    manifests = [json.loads((directory / name).read_text()) for name in
                 ('submit_v2_nf4.manifest.json', 'submit_v3_motion_fast.manifest.json')]
    a, b = [{f['path']: f['sha256'] for f in m['files']} for m in manifests]
    added, removed = sorted(b.keys() - a.keys()), sorted(a.keys() - b.keys())
    changed = sorted(k for k in a.keys() & b.keys() if a[k] != b[k])
    assert added == ['model/stage2/code/solution/stage2_motion_collision.py',
                     'model/stage2/code/solution/stage3_fast.py']
    assert removed == [] and changed == ['inference.py']
    result['manifest_delta'] = dict(added=added, removed=removed, changed=changed)
    result['status'] = 'PASS'
    (new / 'ablation_comparison.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
