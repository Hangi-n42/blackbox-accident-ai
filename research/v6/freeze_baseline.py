"""Bind V6 development to verified V5 bytes; never write into prior artifacts."""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'research/v6/baseline_frozen.json'


def sha(path):
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def main():
    if OUT.exists():
        raise SystemExit('Existing baseline record will not be overwritten.')
    directory = ROOT / 'artifacts/submissions'
    manifest_path = directory / 'submit_v5.manifest.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8-sig'))
    selection_path = directory / 'v5_selection_frozen.json'
    selection = json.loads(selection_path.read_text(encoding='utf-8-sig'))
    expected = selection['expected_archive_sha256']
    assert {row['path']: row['sha256'] for row in manifest['files']} == expected
    archive = directory / 'submit_v5.zip'
    actual_sha = sha(archive)
    assert actual_sha == manifest['sha256']
    package = directory / 'verify_v5'
    rows = []
    for row in manifest['files']:
        path = package / row['path']
        value = sha(path)
        assert value == row['sha256'] and path.stat().st_size == row['bytes'], row['path']
        rows.append({'path': row['path'], 'sha256': value, 'bytes': row['bytes']})
    results = directory / 'verify_v5_results'
    audit = json.loads((results / 'package_audit.json').read_text(encoding='utf-8-sig'))
    assert audit['status'] == 'PASS' and audit['sha256'] == actual_sha
    assert set(audit['unchanged_outputs']) == {'1', '2', '3'}
    bound_evidence = {}
    for path in [manifest_path, selection_path, results / 'package_audit.json',
                 results / 'report.json', *[results / f'stage{i}.csv' for i in (1, 2, 3)]]:
        bound_evidence[path.relative_to(ROOT).as_posix()] = sha(path)
    result = {
        'schema_version': 1,
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'status': 'baseline_bytes_verified_no_new_accuracy_claim',
        'baseline_submission_id': 88887,
        'archive': archive.relative_to(ROOT).as_posix(),
        'archive_sha256': actual_sha,
        'archive_bytes': archive.stat().st_size,
        'package_root': package.relative_to(ROOT).as_posix(),
        'active_modules': {f'stage{i}': selection[f'stage{i}_module'] for i in (1, 2, 3)},
        'files': rows,
        'evidence_sha256': bound_evidence,
        'reported_scores': {'stage1': .5078837545, 'stage2': .2281306982, 'stage3': .5266169416},
        'score_source': 'User-provided completed V5 result image; prior receipt88887',
        'execution_scope': 'No model inference repeated; existing results bound after byte audit.',
        'new_accuracy_candidate': None,
        'holdout_assigned': False,
        'human_annotations_available': False,
        'training_or_candidate_tuning_started': False,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    print(json.dumps({'status': result['status'], 'files_checked': len(rows), 'archive_sha256': actual_sha}))


if __name__ == '__main__':
    main()
