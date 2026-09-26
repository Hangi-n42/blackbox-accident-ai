"""Freeze existing Stage2 evidence and decompose cached decisions; never run a model."""
import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import platform
import sys
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / 'artifacts/mac_experiments/baseline_20260916'
PAIRED = ROOT / 'artifacts/stage2_goal_20260919/current_baseline/run'
PUBLIC = ROOT / 'artifacts/pipeline_diagnosis_20260917/stage2_public_metal'
sys.path.insert(0, str(ROOT / 'research/v7'))
from evaluate_stage2 import macro


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def save(path, data):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(data, stream, ensure_ascii=False, indent=2, allow_nan=False)


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def rel(path):
    return str(Path(path).resolve().relative_to(ROOT))


def close(time, truth):
    return abs(time - truth) <= .3 + 1e-12


def chain(times, truth, upper_frame, candidates, selected):
    """Separate absence in inputs, premature upper bound, sampling, and selection."""
    allowed = {frame for frame, time in times.items() if close(time, truth)}
    bounded = {frame for frame in allowed if frame <= upper_frame}
    offered = set(candidates) & bounded
    flags = [bool(allowed), bool(bounded), bool(offered), selected in allowed]
    labels = ['no_input_frame', 'premature_contact_bound', 'candidate_sampling', 'selection']
    failure = next((name for name, ok in zip(labels, flags) if not ok), None)
    return dict(A0=flags[0], A1=flags[1], A2=flags[2], P=flags[3],
                first_failure=failure, acceptable_frames=sorted(allowed),
                nearest_candidate_error_seconds=min(abs(times[f] - truth) for f in candidates))


def freeze(out):
    out.mkdir(parents=True, exist_ok=False)
    inputs = read(BASE / 'inputs.json')
    labels = {r['ID']: r for r in read(BASE / 'human_labels.json')}
    public_labels = {r['ID']: r for r in csv.DictReader((ROOT / 'Baseline/data/stage2/labels.csv').open())}
    bindings = {}

    def bind(path, expected=None):
        path = Path(path)
        key = rel(path)
        digest = bindings.get(key) or sha(path)
        if expected is not None and digest != expected:
            raise ValueError(f'Changed evidence: {key}')
        bindings[key] = digest
        return key

    for path in [Path(__file__), ROOT / 'research/v7/evaluate_stage2.py',
                 ROOT / '대회_통합_정보.md', BASE / 'inputs.json', BASE / 'human_labels.json',
                 BASE / 'freeze.json', PAIRED / 'stage2_report.json', PUBLIC / 'stage2_report.json',
                 ROOT / 'artifacts/stage2_goal_20260919/server_audit/readiness.json',
                 ROOT / 'Baseline/data/stage2/labels.csv']:
        bind(path)
    initial = read(BASE / 'freeze.json')
    bind(BASE / 'inputs.json', initial['inputs_sha256'])
    bind(BASE / 'human_labels.json', initial['human_labels_sha256'])
    bind(ROOT / 'Baseline/data/stage2/labels.csv', initial['files']['Baseline/data/stage2/labels.csv'])
    run_jobs = {}
    for run in [PAIRED, PUBLIC]:
        report = read(run / 'stage2_report.json')
        assert report['status'] == 'complete'
        assert report['configuration'] == dict(decode_mode='sync', compute_dtype='native', deepstack_fix=True, policy='baseline')
        assert all(w['exit_status'] == 0 for w in report['workers'])
        videos = {(v['group'], v['ID']): v for v in report['videos']}
        assert len(videos) == len(report['videos']) == len(report['workers'])
        jobs = list(sorted(run.glob('job_*.json')))
        assert len(jobs) == len(videos)
        for index, job_path in enumerate(jobs):
            bind(job_path)
            bind(run / f'worker_{index:03d}.log')
            job = read(job_path)
            assert set(job) == {'ID', 'group', 'paths'}
            key = (job['group'], job['ID'])
            assert key in videos and key not in run_jobs
            assert report['workers'][index]['ID'] == job['ID']
            run_jobs[key] = (job, videos[key])
    for path in (ROOT / 'scripts/mac').glob('*.py'):
        bind(path)
    for version in ['v6', 'v7']:
        base = ROOT / f'releases/{version}/source'
        bind(base / 'inference.py')
        for path in (base / 'model/stage2/code/solution').glob('*.py'):
            bind(path)
    # The Mac runner imports this preserved release, not releases/v7 by name.
    for path in (ROOT / 'artifacts/submissions/verify_v6/model/stage2/code/solution').glob('*.py'):
        bind(path)

    records = []
    for case in inputs:
        case_id, group = case['ID'], case['group']
        public = group == 'public'
        record = dict(ID=case_id, group=group, role='public_contact_regression' if public else 'exposed_human_draft_development',
                      independent_holdout=False, new_ground_truth=False, source_group_audit='unresolved', images=[])
        if public:
            pts_file = ROOT / f'research/v6_stage2/public_pts/{case_id}.json'
            bind(pts_file)
            pts = read(pts_file)
            times = {v['frame']: v['pts_seconds'] for v in pts['frame_pts']}
            record['time_mapping_source'] = rel(pts_file)
            record['source_sha256'] = pts['source_video_sha256']
            gt = int(public_labels[case_id]['t_collision'])
            record['labels'] = dict(contact=dict(status='official_public_partial', frame=gt, pts_seconds=times[gt]),
                                    entry=dict(status='unknown'), side=None, space=None, target=None)
            record['gt_kind'] = 'official_public_contact_only'
            result_file = PUBLIC / f'stage2/public/{case_id}/result.json'
        else:
            label = labels[case_id]
            bind(ROOT / label['path'], label['sha256'])
            draft = label['draft']
            assert draft == read(ROOT / label['path'])
            assert draft['source_video_sha256'] == case['source_sha256']
            record['source_sha256'] = case['source_sha256']
            record['review_source'] = label['path']
            record['labels'] = draft['review']
            record['gt_kind'] = 'single_human_draft_not_adjudicated'
            result_file = PAIRED / f'stage2/human_dev/{case_id}/result.json'
            times = {v['frame']: v['pts_seconds'] for v in case['images']}
            case_file = (ROOT / case['images'][0]['path']).parent / 'case.json'
            bind(case_file)
            source_case = read(case_file)
            assert source_case['video_sha256'] == case['source_sha256']
            assert [(v['frame'], v['pts_seconds'], v['sha256']) for v in source_case['frames']] == [(v['frame'], v['pts_seconds'], v['sha256']) for v in case['images']]
            record['time_mapping_source'] = rel(case_file)
            record['source_original_hash_rechecked_in_this_freeze'] = False
        for item in case['images']:
            bind(ROOT / item['path'], item['sha256'])
            record['images'].append(dict(item, pts_seconds=times[item['frame']]))
        frames = [v['frame'] for v in record['images']]
        assert frames == sorted(set(frames)) and all(math.isfinite(times[f]) for f in frames)
        assert all(times[a] < times[b] for a, b in zip(frames, frames[1:]))
        for field in ['contact', 'entry']:
            label = record['labels'].get(field, {})
            if label.get('status') in ('observed', 'official_public_partial'):
                assert label['frame'] in times and times[label['frame']] == label['pts_seconds']
        bind(result_file)
        bind(result_file.parent / 'calls.json')
        bind(result_file.parent / 'motion.npz')
        result = read(result_file)
        job, reported = run_jobs[(group, case_id)]
        assert result == reported and result['ID'] == case_id and result['group'] == group
        assert result['frames'] == len(case['images'])
        assert [str((ROOT / v['path']).resolve()) for v in case['images']] == job['paths']
        assert [int(Path(p).stem.split('_')[-1]) for p in job['paths']] == frames
        assert result['calls'] == read(result_file.parent / 'calls.json') and len(result['calls']) == 4
        assert result['baseline_prediction'] == result['prediction']
        assert all(c['decode_mode'] == 'sync' and c['compute_dtype'] == 'native' and c['deepstack_fix'] is True for c in result['calls'])
        for name in ['run_stage2.py', 'mlx_stage2.py', 'deepstack_fix.py', 'sync_decode.py']:
            bind(ROOT / 'scripts/mac' / name, result['source_sha256'][name])
        record['prediction_source'] = rel(result_file)
        record['cached_prediction_key'] = 'baseline_prediction'
        record['cache_scope'] = 'Four-call baseline only; actual job paths, report, logs and call configuration verified'
        records.append(record)
    manifest = dict(created_utc=datetime.now(timezone.utc).isoformat(), cases=records,
                    purpose='Frozen retrospective development diagnosis; not a new blind evaluation',
                    official_S2=None, tolerance_seconds=.3,
                    missing_labels_are_not_negative=True, server_precision='CUDA NF4',
                    mac_precision='MLX 4bit, sync native, deepstack_fix=True',
                    frame_times_for_scoring_only=True, inference_extra_metadata=False)
    save(out / 'evaluation_manifest.json', manifest)
    bind(out / 'evaluation_manifest.json')
    packages = {}
    for name in ['numpy', 'opencv-python-headless', 'pillow', 'torch', 'av']:
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    save(out / 'freeze.json', dict(files=bindings, platform=platform.platform(), python=sys.version,
                                 packages=packages, model_assets_audited_separately='server_audit/readiness.json',
                                 checked_input_images=sum(len(c['images']) for c in records)))
    print(json.dumps(dict(status='frozen', cases=len(records), files=len(bindings)), ensure_ascii=False))


def diagnose(out):
    import numpy as np
    frozen = read(out / 'freeze.json')
    for file, digest in frozen['files'].items():
        assert sha(ROOT / file) == digest, f'Changed frozen evidence: {file}'
    manifest = read(out / 'evaluation_manifest.json')
    rows = []
    metrics = {g: dict(contact=[], entry=[], side=[], space=[]) for g in ['public', 'human_dev']}
    for case in manifest['cases']:
        result = read(ROOT / case['prediction_source'])
        pred, d = result['baseline_prediction'], result['diagnostics']
        times = {r['frame']: r['pts_seconds'] for r in case['images']}
        numbers = list(times)
        assert set(pred) == {'collision_frame', 'entry_frame', 'entry_side', 'evasion_space'}
        assert type(pred['collision_frame']) is int and pred['collision_frame'] in times
        assert type(pred['entry_frame']) is int and pred['entry_frame'] in times
        assert pred['entry_side'] in ('LEFT', 'RIGHT') and type(pred['evasion_space']) is int and pred['evasion_space'] in (0, 1)
        motion = np.load((ROOT / case['prediction_source']).parent / 'motion.npz')
        for field in ['base_scores', 'new_scores']:
            values = motion[field]
            assert values.shape == (len(numbers),) and np.isfinite(values).all()
            digest = hashlib.sha256(np.ascontiguousarray(values).tobytes()).hexdigest()
            assert digest == d['uncapped_jerk'][field.replace('scores', 'score') + '_sha256']
        assert numbers[int(np.argmax(motion['new_scores']))] == pred['collision_frame']
        internal = d['collision_replacement']['base_collision_frame']
        offered = d['collision_candidates']
        assert internal in times and all(f in times for f in offered + d['entry_candidates'])
        assert all(f <= internal for f in d['entry_candidates'])
        assert pred['entry_frame'] in d['entry_candidates']
        row = dict(ID=case['ID'], group=case['group'], prediction=pred, internal_contact_frame=internal,
                   final_contact_frame=pred['collision_frame'],
                   contact_context_gap_seconds=times[pred['collision_frame']] - times[internal],
                   entry_after_final_contact=times[pred['entry_frame']] > times[pred['collision_frame']],
                   target_misidentification='not_established_by_numeric_trace',
                   contact_evidence_visibility='requires_visual_review_of_actual_model_input',
                   raw_first_four_answers=[c['text'] for c in result['calls'][:4]])
        for field, key in [('contact', 'collision_frame'), ('entry', 'entry_frame')]:
            label = case['labels'].get(field, {})
            if label.get('status') not in ('observed', 'official_public_partial'):
                row[field] = dict(scored=False, reason='unknown')
                continue
            truth = label['pts_seconds']
            error = times[pred[key]] - truth
            metrics[case['group']][field].append(abs(error))
            if field == 'entry':
                detail = chain(times, truth, internal, d['entry_candidates'], pred[key])
                detail['already_entered_at_start_draft'] = case['labels'].get('already_entered_at_start')
            else:
                acceptable = {f for f in numbers if close(times[f], truth)}
                covered = bool(acceptable & set(offered))
                detail = dict(candidate_coverage=covered, internal_choice_correct=internal in acceptable,
                              final_choice_correct=pred[key] in acceptable,
                              candidate_missing=not covered, candidate_present_but_internal_choice_wrong=covered and internal not in acceptable,
                              motion_recovered=internal not in acceptable and pred[key] in acceptable,
                              motion_lost=internal in acceptable and pred[key] not in acceptable)
            row[field] = dict(scored=True, truth_frame=label['frame'], truth_seconds=truth,
                              signed_error_seconds=error, correct=close(times[pred[key]], truth), **detail)
        for field, key, classes in [('side', 'entry_side', ['LEFT', 'RIGHT']), ('space', 'evasion_space', [0, 1])]:
            value = case['labels'].get(field)
            if field == 'space' and value in ('0', '1'):
                value = int(value)
            if value in classes:
                metrics[case['group']][field].append((value, pred[key]))
        rows.append(row)
    summaries = {}
    for group, values in metrics.items():
        summaries[group] = {}
        for field, errors in values.items():
            if field in ('contact', 'entry'):
                summaries[group][field] = dict(n=len(errors), correct=sum(e <= .3 + 1e-12 for e in errors),
                                               mae_seconds=sum(errors) / len(errors) if errors else None)
            else:
                summaries[group][field] = dict(n=len(errors), macro_f1=macro(errors, ['LEFT', 'RIGHT'] if field == 'side' else [0, 1]))
    entries = [r['entry'] for r in rows if r['entry']['scored']]
    contacts = {g: [r['contact'] for r in rows if r['group'] == g and r['contact']['scored']] for g in metrics}
    summary = dict(metrics=summaries,
                   entry_funnel={key: sum(r[key] for r in entries) for key in ['A0', 'A1', 'A2', 'P']},
                   entry_first_failure={key: sum(r['first_failure'] == key for r in entries) for key in ['no_input_frame', 'premature_contact_bound', 'candidate_sampling', 'selection']},
                   contact_decomposition={g: {k: sum(r[k] for r in records) for k in ['candidate_coverage', 'internal_choice_correct', 'final_choice_correct', 'motion_recovered', 'motion_lost']} for g, records in contacts.items()},
                   context_mismatch=sum(r['internal_contact_frame'] != r['final_contact_frame'] for r in rows),
                   entry_after_final_contact=sum(r['entry_after_final_contact'] for r in rows),
                   target_error_count=None, contact_visibility_error_count=None, official_S2=None,
                   limitation='Existing official partial labels and exposed single human drafts; no new accuracy or causal visual diagnosis')
    save(out / 'decomposition.json', dict(summary=summary, cases=rows))
    save(out / 'checks.json', dict(status='pass', input_hashes_checked=frozen['checked_input_images'],
                                 frozen_files_checked=len(frozen['files']), cache_motion_hash_and_argmax=True,
                                 native_time_mapping_preserved=True, output_contract_cases=len(rows),
                                 new_model_inference=False, production_changed=False))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def selfcheck():
    # A late ground-truth entry cannot be recovered by denser sampling before a wrong contact bound.
    times = {10: 0., 20: .5, 30: 1.0, 40: 1.3, 50: 2.0}
    assert chain(times, 2., 30, [10, 20, 30], 30)['first_failure'] == 'premature_contact_bound'
    assert chain(times, 1., 50, [10, 20, 50], 20)['first_failure'] == 'candidate_sampling'
    assert chain(times, 1., 50, [10, 30, 40], 10)['first_failure'] == 'selection'
    assert chain(times, 1., 50, [10, 30, 40], 40)['P']
    assert not chain(times, 1., 50, [10, 20, 50], 50)['P']
    print('funnel semantics and inclusive tolerance checks passed')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['freeze', 'diagnose', 'selfcheck'])
    parser.add_argument('--output', type=Path, default=ROOT / 'artifacts/stage2_goal_20260919/core')
    parser.add_argument('--human-run', type=Path, help='Fresh baseline report directory; default uses historical paired run')
    args = parser.parse_args()
    if args.human_run:
        PAIRED = args.human_run.resolve()
    if args.mode == 'selfcheck':
        selfcheck()
    else:
        (freeze if args.mode == 'freeze' else diagnose)(args.output.resolve())
