"""Bind completed blind expert reviews, images, runtime and original sources before inference."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent


def read(path):
    return json.loads(path.read_text())


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


assert not (HERE / 'mac_run').exists(), 'Cannot freeze reviews after inference started'
assert not (HERE / 'review_lock.json').exists(), 'Do not overwrite prior lock'
files = dict(read(HERE.parent / 'candidate_runtime/freeze.json')['files'])
files.update(read(HERE / 'runtime_precheck.json')['model_files'])
for name, digest in files.items():
    assert sha(ROOT / name) == digest, name

for folder in ['reviewer_a', 'reviewer_b', 'adjudication']:
    review = read(HERE / folder / 'records.json')
    assert len(review['records']) == 6
    assert (HERE / folder / 'review.md').is_file()

selected = [path for folder in ['reviewer_a', 'reviewer_b', 'adjudication', 'blind_packet']
            for path in (HERE / folder).rglob('*') if path.is_file() and '__pycache__' not in path.parts]
selected += [HERE / name for name in ['protocol.json', 'inputs.json', 'input_checks.json', 'runtime_precheck.json',
                                     'prepare_inputs.py', 'extract.py', 'run_paired_mac.py', 'freeze_reviews.py']]
inputs = read(HERE / 'inputs.json')
assert len(inputs) == 6
for case in inputs:
    for path_key, hash_key in [('source_video', 'source_sha256'), ('pts_source', 'pts_sha256')]:
        path = ROOT / case[path_key]
        assert sha(path) == case[hash_key]
        selected.append(path)
for path in selected:
    files[str(path.relative_to(ROOT))] = sha(path)

lock = dict(status='locked_before_model_predictions', created_utc=datetime.now(timezone.utc).isoformat(),
            model_predictions_seen=False, human_review=False, review_type='two_independent_AI_experts_and_third_adjudicator',
            review_scope='Six new-source diagnostic videos; exact observations, conditional tracks and unknowns kept separate',
            input_image_binding='Every PNG SHA is bound in inputs.json and rechecked before each model worker',
            original_frames=sum(len(case['images']) for case in inputs),
            code_policy='The previously frozen final auxiliary-score removal; no parameter or prompt changes',
            files=dict(sorted(files.items())))
with (HERE / 'review_lock.json').open('x') as stream:
    json.dump(lock, stream, ensure_ascii=False, indent=2)
print(json.dumps(dict(status=lock['status'], files=len(files), input_frames=lock['original_frames'],
                      created_utc=lock['created_utc'])))
