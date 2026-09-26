"""One frozen CPU ablation: remove both auxiliary terms from final contact score."""
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
sys.path.insert(0, str(ROOT / 'artifacts/submissions/verify_v6/model/stage2/code'))
import numpy as np
from solution import stage2_uncapped_jerk_v6c as baseline


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def ahash(values):
    return hashlib.sha256(np.ascontiguousarray(values).tobytes()).hexdigest()


def main():
    protocol = read(HERE / 'protocol.json')
    for file, digest in protocol['files'].items():
        assert sha(ROOT / file) == digest, file
    core = ROOT / protocol['core_dir']
    frozen = read(core / 'freeze.json')
    for file, digest in frozen['files'].items():
        assert sha(ROOT / file) == digest, file
    cases = read(core / 'evaluation_manifest.json')['cases']
    rows = []
    for case in cases:
        source = ROOT / case['prediction_source']
        result = read(source)
        cache = np.load(source.parent / 'motion.npz')
        features, old_scores = cache['features'], cache['new_scores']
        recomputed_base, recomputed_new = baseline._scores_from_features(features)
        assert ahash(recomputed_base) == ahash(cache['base_scores'])
        assert ahash(recomputed_new) == ahash(old_scores)
        jerk = features[:, 0]
        median = np.median(jerk)
        scale = np.median(np.abs(jerk - median)) * 1.4826
        scores = np.maximum((jerk - median) / max(float(scale), 1e-3), 0)
        scores[0] = 0
        assert scores.shape == old_scores.shape and np.isfinite(scores).all()
        numbers = [image['frame'] for image in case['images']]
        times = {image['frame']: image['pts_seconds'] for image in case['images']}
        old = result['baseline_prediction']
        assert numbers[int(np.argmax(old_scores))] == old['collision_frame']
        new = dict(old, collision_frame=numbers[int(np.argmax(scores))])
        assert all(new[key] == old[key] for key in ['entry_frame', 'entry_side', 'evasion_space'])
        truth = case['labels']['contact']['pts_seconds']
        old_error = abs(times[old['collision_frame']] - truth)
        new_error = abs(times[new['collision_frame']] - truth)
        correct_old, correct_new = old_error <= .3 + 1e-12, new_error <= .3 + 1e-12
        rows.append(dict(ID=case['ID'], group=case['group'], baseline=old, candidate=new,
                         baseline_error_seconds=old_error, candidate_error_seconds=new_error,
                         baseline_correct=correct_old, candidate_correct=correct_new,
                         gained=not correct_old and correct_new, lost=correct_old and not correct_new,
                         auxiliary_terms_removed=True, baseline_score_sha256=ahash(old_scores),
                         candidate_score_sha256=ahash(scores), other_three_outputs_unchanged=True))
    summary = {}
    for group in ['public', 'human_dev']:
        members = [row for row in rows if row['group'] == group]
        summary[group] = dict(n=len(members), baseline_correct=sum(row['baseline_correct'] for row in members),
                              candidate_correct=sum(row['candidate_correct'] for row in members),
                              gained=sum(row['gained'] for row in members), lost=sum(row['lost'] for row in members),
                              baseline_mae_seconds=sum(row['baseline_error_seconds'] for row in members)/len(members),
                              candidate_mae_seconds=sum(row['candidate_error_seconds'] for row in members)/len(members))
    public, human = summary['public'], summary['human_dev']
    gate = public['lost'] == 0 and public['candidate_mae_seconds'] <= public['baseline_mae_seconds'] + 1e-12
    gate = gate and human['gained'] >= 1 and human['lost'] == 0 and human['candidate_mae_seconds'] <= human['baseline_mae_seconds'] + 1e-12
    output = dict(status='COMPLETE_DEVELOPMENT_ABLATION', protocol_sha256=sha(HERE / 'protocol.json'),
                  summary=summary, development_gate_passed=gate, submission_adopted=False,
                  new_inference_calls=0, official_S2=None, production_changed=False,
                  independent_confirmatory_truth_available=False, cases=rows)
    with (HERE / 'results.json').open('x', encoding='utf-8') as stream:
        json.dump(output, stream, ensure_ascii=False, indent=2, allow_nan=False)
    print(json.dumps({k: v for k, v in output.items() if k != 'cases'}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
