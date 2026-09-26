"""Independent stored-record CPU checks. No model instantiation or inference."""
import ast
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import sys

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OLD = ROOT / 'artifacts/stage2_goal_20260920'
MARKER = ROOT / 'artifacts/stage2_entry_marker_20260920'
PREFIX = 'The vehicle marked in yellow is the collision counterpart. '


def read(path):
    return json.loads(path.read_text())


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def rgb_sha(image):
    return hashlib.sha256(image.tobytes()).hexdigest()


def bounded(image):
    from PIL import Image
    scale = min(1., math.sqrt(1_200_000 / (image.width * image.height)))
    size = tuple(max(32, int(v * scale) // 32 * 32) for v in image.size)
    return image.convert('RGB').resize(size, Image.Resampling.BICUBIC)


# Reuse the independent exact-rational maths, not either experiment evaluator.
MATH_SOURCE = OLD / 'verify_runtime_score.py'
nodes = [n for n in ast.parse(MATH_SOURCE.read_text()).body
         if isinstance(n, ast.FunctionDef) and n.name in
         {'same', 'rational_grade', 'rational_paired_delta', 'independent_aggregate'}]
exec(compile(ast.Module(body=nodes, type_ignores=[]), str(MATH_SOURCE), 'exec'))


def main():
    import numpy as np
    from PIL import Image
    sys.path.insert(0, str(ROOT / 'artifacts/submissions/verify_v6/model/stage2/code'))
    from solution import stage2_v2 as v2
    from solution import stage2_uncapped_jerk_v6c as reference

    targets = [HERE / ('independent_verification' + suffix) for suffix in ('.json', '.md')]
    assert not any(path.exists() for path in targets)
    frozen, protocol, evaluation = [read(HERE / name) for name in ('freeze.json', 'protocol.json', 'evaluation.json')]
    assert frozen['status'] == 'locked_before_legend_and_state_predictions'
    ids = protocol['cases']
    assert len(ids) == len(set(ids)) == 5 and protocol['legend_sentence'] == PREFIX
    assert evaluation['status'] == 'complete' and evaluation['official_S2'] is None
    old_frozen = read(MARKER / 'freeze.json')['files']
    assert len(old_frozen) == 865
    assert all(frozen['files'].get(name) == digest for name, digest in old_frozen.items())
    for name, digest in frozen['files'].items():
        assert sha(ROOT / name) == digest, name
    freeze_time = datetime.fromisoformat(frozen['created_utc'])
    artifact_hashes = {str(MATH_SOURCE.relative_to(ROOT)): sha(MATH_SOURCE),
                       str(Path(__file__).relative_to(ROOT)): sha(Path(__file__)),
                       str((HERE / 'freeze.json').relative_to(ROOT)): sha(HERE / 'freeze.json')}
    sources = {r['ID']: r for r in read(OLD / 'ccd_intake/inputs.json')}
    originals = {r['ID']: r for r in read(OLD / 'mac_run/report.json')['videos']}
    labels = {r['ID']: r['entry'] for r in read(OLD / 'ccd_adjudication/records.json')['records']}
    resources = []

    def load_run(state=False):
        out = HERE / ('state_run' if state else 'run')
        report = read(out / 'report.json')
        arms = ['state'] if state else ['control', 'legend']
        expected = [(sid, arm) for sid in ids for arm in arms]
        assert report['status'] == 'complete' and report['freeze_sha256'] == sha(HERE / 'freeze.json')
        assert [(w['ID'], w['arm']) for w in report['workers']] == expected
        assert freeze_time < datetime.fromisoformat(report['started_utc']) < datetime.fromisoformat(report['ended_utc'])
        records = {}
        for entry in report['workers']:
            sid, arm = entry['ID'], entry['arm']
            assert entry['exit_status'] == 0
            folder = out / f'{sid}_{arm}'
            worker, result, calls = [read(folder / n) for n in ('worker_report.json', 'result.json', 'calls.json')]
            job = read(HERE / 'inputs' / f'{sid}_{arm}.job.json')
            assert job['ID'] == worker['ID'] == result['ID'] == sid and worker['arm'] == arm
            assert worker['status'] == 'complete' and worker['network_attempts'] == 0
            assert worker['model_calls'] == len(calls) == 1 and calls[0] == result['call']
            assert freeze_time < datetime.fromisoformat(worker['started_utc']) <= datetime.fromisoformat(worker['ended_utc'])
            call = calls[0]
            assert call['text'].strip() == result['raw'].strip()
            assert call['prompt'] == job['prompt'] and call['max_new_tokens'] == job['max_new_tokens'] == 40
            assert call['compute_dtype'] == 'native' and call['decode_mode'] == 'sync' and call['deepstack_fix'] is True
            assert call['processor_input_sha256'] and call['token_trace']
            if state:
                assert set(job) == {'ID', 'image', 'sha256', 'prompt', 'max_new_tokens'}
                image_path, digest = Path(job['image']), job['sha256']
                assert job['prompt'] == protocol['conditional_state_probe']['prompt']
            else:
                assert set(job) == {'ID', 'arm', 'sheet', 'sheet_sha256', 'paths', 'candidates', 'prompt', 'max_new_tokens'}
                assert result['arm'] == job['arm'] == arm
                image_path, digest = Path(job['sheet']), job['sheet_sha256']
            assert sha(image_path) == digest
            with Image.open(image_path) as image:
                actual_input = bounded(image)
                assert call['image_sha256'] == [rgb_sha(actual_input)] and call['image_sizes'] == [list(actual_input.size)]
            assert entry['wall_seconds'] >= worker['wall_seconds'] > 0 and call['peak_memory'] > 0
            resources.append(dict(ID=sid, arm=arm, parent_wall_seconds=entry['wall_seconds'], worker_wall_seconds=worker['wall_seconds'],
                                  mlx_peak_GB=call['peak_memory'], rss_bytes=worker['rss_bytes'], generation_seconds=call['seconds']))
            records[(sid, arm)] = (job, result, call)
            for name in ('worker_report.json', 'result.json', 'calls.json'):
                path = folder / name
                artifact_hashes[str(path.relative_to(ROOT))] = sha(path)
        artifact_hashes[str((out / 'report.json').relative_to(ROOT))] = sha(out / 'report.json')
        return records, report

    records, run_report = load_run()
    rows, replayed = [], 0
    for sid in ids:
        source, old, truth = sources[sid], originals[sid], labels[sid]
        paths = [ROOT / f['path'] for f in source['images']]
        numbers = [v2._frame_number(p) for p in paths]
        times = {f['frame']: Fraction(str(f['pts_seconds'])) for f in source['images']}
        assert len(paths) == 50 and numbers == list(range(50)) and truth['evaluation_eligible']
        pts = read(ROOT / source['pts_source'])
        assert sha(ROOT / source['source_video']) == source['source_sha256']
        assert sha(ROOT / source['pts_source']) == source['pts_sha256']
        for item, native, path in zip(source['images'], pts['mapping'], paths, strict=True):
            assert sha(path) == item['sha256'] and native['frame_id'] == item['frame']
            assert Fraction(native['native_pts']) * Fraction(native['time_base']) == times[item['frame']]
        control_job, control, control_call = records[(sid, 'control')]
        legend_job, legend, legend_call = records[(sid, 'legend')]
        prior_job = read(MARKER / 'inputs' / f'{sid}_marked.job.json')
        history = read(MARKER / 'run' / f'{sid}_marked/result.json')
        assert {k: v for k, v in control_job.items() if k != 'arm'} == {k: v for k, v in prior_job.items() if k != 'arm'}
        assert legend_job['prompt'] == PREFIX + control_job['prompt']
        assert {k: v for k, v in legend_job.items() if k not in ('arm', 'prompt')} == {k: v for k, v in control_job.items() if k not in ('arm', 'prompt')}
        assert control_call['processor_input_sha256'] == history['call']['processor_input_sha256']
        cp, lp = control_call['processor_input_sha256'], legend_call['processor_input_sha256']
        assert set(cp) == set(lp) and {'pixel_values', 'image_grid_thw', 'input_ids', 'attention_mask'} <= set(cp)
        assert {key for key in cp if cp[key] != lp[key]} == {'input_ids', 'attention_mask'}
        assert control_call['image_sha256'] == legend_call['image_sha256']
        assert control_call['prompt_sha256'] != legend_call['prompt_sha256']
        lo, hi = times[truth['lower_frame']], times[truth['upper_frame']]
        row = dict(ID=sid, interval_seconds=[float(lo), float(hi)], arms={},
                   historical_control_raw_match=control['raw'].strip() == history['raw'].strip(),
                   historical_control_frame_match=control['entry_frame'] == history['entry_frame'])
        with np.load(OLD / 'mac_run' / sid / 'motion.npz', allow_pickle=False) as motion:
            for arm in ('control', 'legend'):
                job, result, call = records[(sid, arm)]
                assert job['paths'] == [str(p) for p in paths] and job['candidates'] == old['diagnostics']['entry_candidates']
                offered = [numbers.index(n) for n in job['candidates']]
                parsed = v2._json_object(result['raw'])
                raw_integer = v2._integer(parsed.get('entry_frame'))
                chosen = numbers[v2._choice(parsed, 'entry_frame', paths, offered, 0)]
                same(result['parsed'], parsed)
                assert result['raw_integer'] == raw_integer and result['entry_frame'] == chosen
                assert result['valid_offered_integer'] == (raw_integer in job['candidates'])
                assert result['parser_changed'] == (raw_integer != chosen)
                class Replay:
                    count = 0
                    def ask(self, images, prompt, max_new_tokens):
                        i = self.count
                        self.count += 1
                        cached = old['calls'][i]
                        assert prompt == cached['prompt'] and max_new_tokens == cached['max_new_tokens']
                        assert [rgb_sha(bounded(im)) for im in images] == cached['image_sha256']
                        return result['raw'] if i == 2 else cached['text']
                replay = Replay()
                prediction, _ = reference._predict_file(paths, motion['base_scores'], motion['new_scores'], replay)
                assert replay.count == 4 and prediction['entry_frame'] == chosen
                assert all(prediction[key] == old['baseline_prediction'][key] for key in ('collision_frame', 'entry_side', 'evasion_space'))
                replayed += 3
                row['arms'][arm] = dict(prediction=prediction, raw=result['raw'], valid_offered=result['valid_offered_integer'],
                    parser_changed=result['parser_changed'], time_seconds=float(times[chosen]), **rational_grade(times[chosen], lo, hi))
        a, b = (row['arms'][arm] for arm in ('control', 'legend'))
        at, bt = (times[x['prediction']['entry_frame']] for x in (a, b))
        row['gain'] = a['result'] == 'wrong' and b['result'] == 'correct' and b['valid_offered'] and not b['parser_changed']
        row['loss'] = a['result'] == 'correct' and b['result'] == 'wrong'
        row['new_false_first'] = lo > 0 and bt == 0 and at != 0
        row['paired_accuracy_delta'] = rational_paired_delta(at, bt, lo, hi)
        deltas = [abs(bt - t) - abs(at - t) for t in (lo, hi)]
        row['paired_absolute_error_delta_seconds'] = [float(min(deltas)), float(max(deltas))]
        rows.append(row)
    same(evaluation['rows'], rows)
    metrics = {arm: independent_aggregate([r['arms'][arm] for r in rows]) for arm in ('control', 'legend')}
    accuracy_delta = [sum(r['paired_accuracy_delta'][i] for r in rows) / len(rows) for i in (0, 1)]
    mae_delta = [sum(r['paired_absolute_error_delta_seconds'][i] for r in rows) / len(rows) for i in (0, 1)]
    same(evaluation['metrics'], metrics)
    same(evaluation['paired_accuracy_delta_bounds'], accuracy_delta)
    same(evaluation['paired_MAE_delta_bounds_seconds'], mae_delta)
    gate = dict(gains=sum(r['gain'] for r in rows), losses=sum(r['loss'] for r in rows), new_false_first=sum(r['new_false_first'] for r in rows),
                paired_MAE_nonincrease=mae_delta[1] <= 1e-9,
                control_matches_history=all(r['historical_control_raw_match'] and r['historical_control_frame_match'] for r in rows))
    gate['pass'] = gate['gains'] >= 1 and gate['losses'] == 0 and gate['new_false_first'] == 0 and gate['paired_MAE_nonincrease'] and gate['control_matches_history']
    same(evaluation['gate'], gate)
    assert evaluation['actual_model_calls'] == len(records) == 10
    assert evaluation['cached_answers_replayed'] == replayed == 30 and evaluation['other_three_outputs_unchanged'] == 5
    state_summary = None
    if not gate['pass']:
        assert gate['control_matches_history'], 'Contract failure cannot authorize the conditional state probe'
        state_evaluation = read(HERE / 'state_evaluation.json')
        states, state_report = load_run(state=True)
        assert datetime.fromisoformat(run_report['ended_utc']) < datetime.fromisoformat(state_report['started_utc'])
        refs = []
        for who in ('a', 'b'):
            review = read(HERE / f'state_reference_{who}.json')
            assert review['human_review'] is False and review['model_predictions_seen'] is False
            refs.extend(review['cases'])
        assert len(refs) == 5 and {r['ID'] for r in refs} == set(ids)
        refs = {r['ID']: r for r in refs}
        state_rows = []
        for sid in sorted(ids):
            job, result, call = states[(sid, 'state')]
            with Image.open(job['image']) as tile, Image.open(MARKER / 'inputs' / f'{sid}_marked_model_input.png') as sheet:
                assert tile.size == (384, 256) and tile.tobytes() == sheet.crop((0, 0, 384, 256)).tobytes()
            try:
                parsed = json.loads(result['raw'])
            except (ValueError, TypeError):
                parsed = None
            value = parsed.get('lane_state') if isinstance(parsed, dict) else None
            valid = isinstance(value, str) and value in ('INSIDE', 'OUTSIDE', 'UNCERTAIN')
            prediction = value if valid else None
            assert result['valid_enum'] == valid and result['state'] == prediction
            target = refs[sid]['state']
            assert target in ('INSIDE', 'OUTSIDE', 'UNKNOWN')
            state_rows.append(dict(ID=sid, reference=target, eligible=target != 'UNKNOWN', prediction=prediction,
                                   valid_enum=valid, correct=(target == value) if target != 'UNKNOWN' else None, raw=result['raw']))
        same(state_evaluation['rows'], state_rows)
        eligible = [r for r in state_rows if r['eligible']]
        correct = sum(r['correct'] for r in eligible)
        columns = ['INSIDE', 'OUTSIDE', 'UNCERTAIN', None]
        confusion = dict(reference_rows=['INSIDE', 'OUTSIDE'], prediction_columns=columns,
                         matrix=[[sum(r['reference'] == target and r['prediction'] == pred for r in eligible) for pred in columns]
                                 for target in ('INSIDE', 'OUTSIDE')])
        state_summary = dict(actual_model_calls=len(states), eligible=len(eligible), correct=correct,
                             accuracy=correct / len(eligible) if eligible else None, confusion=confusion,
                             uncertain=sum(r['prediction'] == 'UNCERTAIN' for r in state_rows), invalid=sum(not r['valid_enum'] for r in state_rows))
        assert state_evaluation['status'] == 'complete' and state_evaluation['official_S2'] is None
        for key, value in state_summary.items():
            same(state_evaluation[key], value)
        state_summary['rows'] = state_rows
        assert len(states) == 5
    else:
        assert not (HERE / 'state_run').exists()
    for path in [HERE / 'evaluation.json'] + ([HERE / 'state_evaluation.json'] if state_summary else []):
        artifact_hashes[str(path.relative_to(ROOT))] = sha(path)
    for name, digest in frozen['files'].items():
        assert sha(ROOT / name) == digest, name
    for name, digest in artifact_hashes.items():
        assert sha(ROOT / name) == digest, name
    summary = dict(status='PASS', created_utc=datetime.now(timezone.utc).isoformat(), new_verifier_model_calls=0,
                   frozen_files_verified_pre_post=len(frozen['files']), previous_freeze_files_preserved=len(old_frozen),
                   source_png_and_native_pts_checked=250, main_actual_calls=10, cached_answers_replayed=replayed,
                   other_three_outputs_preserved=5, metrics=metrics, rows=rows, paired_accuracy_delta_bounds=accuracy_delta,
                   paired_MAE_delta_bounds_seconds=mae_delta, gate=gate, state_probe=state_summary,
                   resources=resources, artifact_sha256=artifact_hashes,
                   limits=['Exposed five-case expert-marker development diagnostic, not independent generalization.',
                           'AI state/timing references and marker identity correctness are not certified by this byte/math review.',
                           'State probe is a cropped single-frame task, not the same-input temporal ablation or an entry override.',
                           'Recorded processor hashes were checked; tensors were not independently regenerated.',
                           'Python socket checks do not establish OS-wide network isolation; Mac does not establish CUDA equivalence.',
                           'No official S2 was computed.'])
    targets[0].write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n')
    lines = ['# Legend / conditional state independent verification', '', '**PASS.** CPU-only stored-record verification; zero new model calls.', '',
             f"Frozen files before/after: {len(frozen['files'])}; previous freeze preserved: {len(old_frozen)}; source PNG/native PTS: 250.",
             'Ten fresh Q3 workers/calls; 30 cached Q1/Q2/Q4 answers replayed; other three final outputs unchanged in all five pairs.',
             'Marked images/pixel tensors/image grids are identical across arms. Exact prefix sentence alone changes the prompt; input_ids/attention hashes change.',
             'Fresh marker-only raw responses and frame predictions reproduce all five historical controls.', '',
             '| Case | Control | Legend | Control grade | Legend grade |', '|---|---:|---:|---|---|']
    for row in rows:
        a, b = (row['arms'][arm] for arm in ('control', 'legend'))
        lines.append(f"| {row['ID']} | {a['prediction']['entry_frame']} | {b['prediction']['entry_frame']} | {a['result']} | {b['result']} |")
    lines += ['', f'Shared-truth accuracy delta: {accuracy_delta}. Shared-truth MAE delta seconds: {mae_delta}.', f'Gate: {gate}.',
              'Grades use exact rational seconds and closed acceptance intervals; paired MAE extrema were independently calculated at interval endpoints.']
    if state_summary:
        lines += ['', f"Conditional probe: {state_summary['correct']}/{state_summary['eligible']} eligible AI state references; uncertain={state_summary['uncertain']}, invalid={state_summary['invalid']}.",
                  'All five state workers/calls completed after the valid failed main comparison. Each input is the unchanged 384x256 first-tile crop; strict JSON enum parsing has no fallback.',
                  'State response success does not establish first-wheel-contact timing, causal failure attribution, or an automatic frame-zero correction.']
    lines += ['', '## Limits', ''] + ['- ' + value for value in summary['limits']]
    lines += ['', 'No production, frozen source, reference, model, or parent evaluation file was modified. Measurements and artifact hashes are in independent_verification.json.']
    targets[1].write_text('\n'.join(lines) + '\n')
    print(json.dumps({k: summary[k] for k in ('status', 'frozen_files_verified_pre_post', 'main_actual_calls', 'paired_accuracy_delta_bounds', 'paired_MAE_delta_bounds_seconds', 'gate')}, ensure_ascii=False))


if __name__ == '__main__':
    main()
