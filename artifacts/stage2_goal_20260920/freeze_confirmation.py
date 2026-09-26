"""Freeze adjudicated labels and the sole fixed candidate before CCD inference."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def read(path):
    return json.loads(path.read_text())


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


if __name__ == '__main__':
    assert not (HERE / 'mac_run').exists() and not (HERE / 'freeze.json').exists()
    prior = ROOT / 'artifacts/stage2_goal_20260919'
    files = dict(read(prior / 'candidate_runtime/freeze.json')['files'])
    files.update(read(prior / 'expert_mac_review/runtime_precheck.json')['model_files'])
    files[str((prior / 'expert_mac_review/run_paired_mac.py').relative_to(ROOT))] = sha(prior / 'expert_mac_review/run_paired_mac.py')
    for name, digest in files.items():
        assert sha(ROOT / name) == digest, name
    inputs = read(HERE / 'ccd_intake/inputs.json')
    labels = read(HERE / 'ccd_adjudication/records.json')
    assert {r['ID'] for r in inputs} == {r['ID'] for r in labels['records']}
    assert labels['model_predictions_seen'] is False
    overlap = read(HERE / 'ccd_public_overlap.json')
    assert not overlap['selected_source_group_collisions']
    assert not overlap['whole_video_sha256_duplicates'] and not overlap['sampled_exact_rgb_matches']
    paths = [HERE / n for n in ['protocol.json', 'evidence_protocol_review.json', 'evidence_protocol_review.md',
                               'ccd_public_overlap.json', 'ccd_public_overlap.md', 'ccd_prior_images_overlap.json',
                               'data_source_audit.json', 'run_confirmation.py', 'evaluate_confirmation.py', 'decompose_confirmation.py', 'freeze_confirmation.py',
                               'ccd_intake/selection.json', 'ccd_intake/annotation.txt', 'ccd_intake/inputs.json',
                               'ccd_intake/acquisition.json', 'ccd_intake/acquire_selected.py', 'ccd_intake/repo_readme.txt']]
    for folder in ['ccd_review_a', 'ccd_review_b', 'ccd_adjudication']:
        assert (HERE / folder / 'records.json').exists()
        paths.extend(p for p in (HERE / folder).rglob('*') if p.is_file() and '__pycache__' not in p.parts)
    for case in inputs:
        for key, expected in [('source_video', 'source_sha256'), ('pts_source', 'pts_sha256')]:
            p = ROOT / case[key]
            assert sha(p) == case[expected]
            paths.append(p)
        for image in case['images']:
            p = ROOT / image['path']
            assert sha(p) == image['sha256']
            paths.append(p)
        paths.extend((HERE / 'ccd_intake' / case['ID']).glob('sheet_*.jpg'))
    for p in paths:
        files[str(p.relative_to(ROOT))] = sha(p)
    excluded = labels.get('duplicate_excluded_ids', [])
    result = dict(status='locked_before_predictions', created_utc=datetime.now(timezone.utc).isoformat(),
                  files=dict(sorted(files.items())), excluded_duplicate_ids=excluded, run_cases=len(inputs) - len(excluded),
                  reviewed_cases=len(inputs), images=sum(len(c['images']) for c in inputs),
                  candidate_policy='same fixed jerk-only final selection; no new model/prompt/threshold search',
                  review_type='AI evidence annotations, not official or human ground truth')
    with (HERE / 'freeze.json').open('x') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
    print(json.dumps({k:v for k,v in result.items() if k != 'files'}))
