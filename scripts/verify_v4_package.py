"""Verify V4 archive bytes, frozen change scope, and real offline results."""
import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--selection', type=Path, required=True)
    parser.add_argument('--package', type=Path, required=True)
    parser.add_argument('--results', type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    submissions = root / 'artifacts/submissions'
    selected = json.loads(args.selection.read_text(encoding='utf-8'))
    old = json.loads((submissions / 'submit_v3_motion_fast.manifest.json').read_text(encoding='utf-8'))
    new = json.loads((submissions / 'submit_v4.manifest.json').read_text(encoding='utf-8'))
    zip_path = Path(new['zip'])
    assert digest(zip_path) == new['sha256']
    assert zip_path.stat().st_size == new['zip_bytes']
    previous = {row['path']: row for row in old['files']}
    current = {row['path']: row for row in new['files']}
    added = sorted(current.keys() - previous.keys())
    removed = sorted(previous.keys() - current.keys())
    changed = sorted(name for name in current.keys() & previous.keys()
                     if current[name]['sha256'] != previous[name]['sha256'])
    assert added == sorted(selected['allowed_added_paths']), added
    assert removed == sorted(selected.get('allowed_removed_paths', [])), removed
    assert changed == sorted(selected['allowed_changed_paths']), changed
    actual = {path.relative_to(args.package).as_posix() for path in args.package.rglob('*') if path.is_file()}
    assert actual == current.keys(), sorted(actual ^ current.keys())
    for name, row in current.items():
        path = args.package / name
        assert path.stat().st_size == row['bytes'] and digest(path) == row['sha256'], name
    report = json.loads((args.results / 'report.json').read_text(encoding='utf-8'))
    assert report['status'] == 'PASS'
    equivalence = {}
    for stage in selected['unchanged_output_stages']:
        baseline = pd.read_csv(submissions / 'verify_v3_motion_fast_results' / f'stage{stage}.csv')
        candidate = pd.read_csv(args.results / f'stage{stage}.csv')
        pd.testing.assert_frame_equal(baseline, candidate)
        equivalence[str(stage)] = {'rows': len(candidate), 'exact': True}
    if selected.get('expected_stage2_csv'):
        expected = pd.read_csv(root / selected['expected_stage2_csv'])
        actual_predictions = pd.read_csv(args.results / 'stage2.csv')
        pd.testing.assert_frame_equal(expected[actual_predictions.columns], actual_predictions)
    output = {'status': 'PASS', 'sha256': new['sha256'], 'zip_bytes': new['zip_bytes'],
              'uncompressed_bytes': new['uncompressed_bytes'], 'added': added,
              'removed': removed, 'changed': changed, 'unchanged_outputs': equivalence,
              'files_checked': len(current), 'scope': 'Integrity, change scope and public output contract; no hidden-set accuracy claim'}
    (args.results / 'package_audit.json').write_text(json.dumps(output, indent=2), encoding='utf-8')
    print(json.dumps(output, indent=2))


if __name__ == '__main__':
    main()
