"""Independent CPU audit of the new-source first-frame policy. Never runs a model."""
import ast
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import json
import math
import os
from pathlib import Path
import sys

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OLD = ROOT / 'artifacts/stage2_goal_20260920'
LEGEND = ROOT / 'artifacts/stage2_entry_legend_20260920'
PREV = ROOT / 'artifacts/stage2_state_resolution_20260920'

# Reuse only independently checked pure helpers, not prior evaluator mains.
for source, names in [(OLD / 'verify_runtime_score.py', {'same', 'rational_grade', 'rational_paired_delta', 'independent_aggregate'}),
                      (PREV / 'verify.py', {'read', 'sha', 'rgb_sha', 'bounded'})]:
    nodes = [n for n in ast.parse(source.read_text()).body if isinstance(n, ast.FunctionDef) and n.name in names]
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source), 'exec'))


def main():
    # No heavy verification work until all timed model workers have completed.
    report, evaluation = read(HERE / 'run/report.json'), read(HERE / 'evaluation.json')
    assert report['status'] == evaluation['status'] == 'complete'
    targets = [HERE / ('independent_verification' + x) for x in ('.json', '.md')]
    assert not any(p.exists() for p in targets)
    import numpy as np
    from PIL import Image, ImageDraw, ImageOps
    sys.path.insert(0, str(ROOT / 'artifacts/submissions/verify_v6/model/stage2/code'))
    from solution import stage2_v2 as v2
    from solution import stage2_uncapped_jerk_v6c as reference
    frozen = read(HERE / 'freeze.json')
    assert frozen['status'] == 'locked_before_all_new_predictions'
    old_files = read(PREV / 'freeze.json')['files']
    assert len(old_files) == 969 and all(frozen['files'].get(k) == v for k, v in old_files.items())
    for name, digest in frozen['files'].items():
        assert sha(ROOT / name) == digest, name
    freeze_time = datetime.fromisoformat(frozen['created_utc'])
    assert freeze_time < datetime.fromisoformat(report['started_utc']) < datetime.fromisoformat(report['ended_utc'])
    assert report['freeze_sha256'] == sha(HERE / 'freeze.json')
    sources, selection = read(HERE / 'intake/inputs.json'), read(HERE / 'intake/selection.json')
    refs = {r['ID']: r for r in read(HERE / 'references.json')['cases']}
    ids = [r['ID'] for r in sources]
    assert ids == [r['ID'] for r in selection['selected']] and len(ids) == len(set(ids)) == len(refs) == 12
    assert set(ids) == set(refs) and len({r['source_group'] for r in sources}) == 12
    assert not {r['source_group'] for r in sources} & set(selection['excluded_source_groups'])
    assert read(HERE / 'preflight_review.json')['status'] == read(HERE / 'input_peer_review.json')['status'] == 'PASS'
    for who in ('a', 'b'):
        review = read(HERE / f'review_{who}.json')
        assert review['model_predictions_seen'] is False and review['peer_review_seen'] is False
    overlap = read(HERE / 'intake/overlap.json')
    assert overlap['status'] == 'complete_no_overlap_found_in_checked_scope' and not overlap['source_group_overlap']
    assert not overlap['whole_video_sha256_duplicates'] and not overlap['sampled_exact_rgb_matches']
    expected_workers = [(sid, arm) for sid in ids for arm in ('baseline', 'state') if arm == 'baseline' or refs[sid]['mark_approved']]
    assert [(r['ID'], r['arm']) for r in report['workers']] == expected_workers
    assert all(r['exit_status'] == 0 for r in report['workers'])
    assert frozen['run_cases'] == 12 and frozen['state_cases'] == sum(refs[sid]['mark_approved'] for sid in ids)
    prompt = read(LEGEND / 'inputs/CCD_000688_state.job.json')['prompt']
    hashes = {str(Path(__file__).relative_to(ROOT)): sha(Path(__file__))}
    for path in (OLD / 'verify_runtime_score.py', PREV / 'verify.py', HERE / 'evaluation.json', HERE / 'run/report.json', HERE / 'freeze.json'):
        hashes[str(path.relative_to(ROOT))] = sha(path)
    rows, resources, state_inputs, total_calls = [], [], [], 0

    def worker_files(sid, arm, call_count):
        folder = HERE / 'run' / f'{sid}_{arm}'
        worker, result, calls = [read(folder / name) for name in ('worker_report.json', 'result.json', 'calls.json')]
        assert worker['ID'] == result['ID'] == sid and worker['status'] == 'complete'
        assert worker['model_calls'] == len(calls) == call_count and worker['network_attempts'] == 0
        assert freeze_time < datetime.fromisoformat(worker['started_utc']) <= datetime.fromisoformat(worker['ended_utc'])
        parent = next(r for r in report['workers'] if r['ID'] == sid and r['arm'] == arm)
        assert parent['wall_seconds'] >= worker['wall_seconds'] > 0
        for call in calls:
            assert call['deepstack_fix'] is True and call['compute_dtype'] == 'native' and call['decode_mode'] == 'sync'
            assert call['token_trace'] and call['processor_input_sha256'] and call['peak_memory'] > 0
        resources.append(dict(ID=sid, arm=arm, calls=len(calls), parent_wall_seconds=parent['wall_seconds'], worker_wall_seconds=worker['wall_seconds'],
                              mlx_peak_GB=max(r['peak_memory'] for r in calls), rss_bytes=worker.get('rss_bytes', worker.get('max_process_rss_bytes'))))
        for name in ('worker_report.json', 'result.json', 'calls.json'):
            hashes[str((folder / name).relative_to(ROOT))] = sha(folder / name)
        return folder, worker, result, calls

    for source in sources:
        sid, ref = source['ID'], refs[source['ID']]
        paths = [ROOT / r['path'] for r in source['images']]
        numbers = [v2._frame_number(p) for p in paths]
        times = {r['frame']: Fraction(str(r['pts_seconds'])) for r in source['images']}
        first = numbers[0]
        assert len(paths) == 50 and numbers == list(range(50))
        native_pts = read(ROOT / source['pts_source'])
        assert sha(ROOT / source['source_video']) == source['source_sha256'] and sha(ROOT / source['pts_source']) == source['pts_sha256']
        for im, path, pts in zip(source['images'], paths, native_pts['mapping'], strict=True):
            assert sha(path) == im['sha256'] and im['frame'] == pts['frame_id']
            assert Fraction(pts['native_pts']) * Fraction(pts['time_base']) == times[im['frame']] == Fraction(im['frame'], 10)
        base_job = read(HERE / 'inputs' / f'{sid}_baseline.job.json')
        assert set(base_job) == {'ID', 'paths', 'image_sha256'} and base_job['ID'] == sid
        assert base_job['paths'] == [str(p) for p in paths] and base_job['image_sha256'] == [r['sha256'] for r in source['images']]
        folder, worker, base, calls = worker_files(sid, 'baseline', 4)
        assert base['calls'] == calls
        baseline = base['baseline_prediction']
        class Replay:
            count = 0
            def ask(self, images, question, max_new_tokens):
                call = calls[self.count]
                self.count += 1
                assert question == call['prompt'] and max_new_tokens == call['max_new_tokens']
                assert [rgb_sha(bounded(im)) for im in images] == call['image_sha256']
                assert [list(bounded(im).size) for im in images] == call['image_sizes']
                return call['text']
        with np.load(folder / 'motion.npz', allow_pickle=False) as motion:
            assert all(np.isfinite(motion[k]).all() for k in ('features', 'base_scores', 'new_scores', 'candidate_scores'))
            replay = Replay()
            reproduced, diagnostics = reference._predict_file(paths, motion['base_scores'], motion['new_scores'], replay)
            assert replay.count == 4 and reproduced == baseline
        hashes[str((folder / 'motion.npz').relative_to(ROOT))] = sha(folder / 'motion.npz')
        total_calls += 4
        state = raw = call = None
        if ref['mark_approved']:
            job = read(HERE / 'inputs' / f'{sid}_state.job.json')
            assert set(job) == {'ID', 'image', 'sha256', 'prompt', 'max_new_tokens'}
            assert job['ID'] == sid and job['prompt'] == prompt and job['max_new_tokens'] == 40
            low = v2._sheet(paths, [0], columns=1)
            with Image.open(paths[0]) as original:
                width, height = original.size
                iw, ih = ImageOps.contain(original, (384, 228)).size
            x0, y0, x1, y1 = ref['marker_box_xyxy']
            assert all(type(v) is int for v in (x0, y0, x1, y1)) and 0 <= x0 < x1 < width and 0 <= y0 < y1 < height
            box = [(384 - iw) // 2 + round(x0 * iw / width), 28 + round(y0 * ih / height),
                   (384 - iw) // 2 + round(x1 * iw / width), 28 + round(y1 * ih / height)]
            assert box[0] < box[2] and box[1] < box[3]
            ImageDraw.Draw(low).rectangle(box, outline=(255, 255, 0), width=2)
            enlarged = low.resize((1152, 768), Image.Resampling.NEAREST)
            with Image.open(HERE / 'inputs' / f'{sid}_low.png') as saved:
                assert saved.convert('RGB').tobytes() == low.tobytes() and saved.size == low.size
            with Image.open(job['image']) as saved:
                assert saved.convert('RGB').tobytes() == enlarged.tobytes() and saved.size == enlarged.size
            assert sha(Path(job['image'])) == job['sha256']
            _, worker, result, sc = worker_files(sid, 'state', 1)
            assert worker['arm'] == 'state' and result['call'] == sc[0]
            call, raw = sc[0], result['raw']
            assert call['text'].strip() == raw.strip() and call['prompt'] == prompt and call['max_new_tokens'] == 40
            assert call['image_sha256'] == [rgb_sha(bounded(enlarged))] and call['image_sizes'] == [[1152, 768]]
            try:
                parsed = json.loads(raw)
            except (ValueError, TypeError):
                parsed = None
            value = parsed.get('lane_state') if isinstance(parsed, dict) else None
            valid = isinstance(value, str) and value in ('INSIDE', 'OUTSIDE', 'UNCERTAIN')
            state = value if valid else None
            assert result['state'] == state and result['valid_enum'] == valid
            state_inputs.append((sid, job, call))
            total_calls += 1
        else:
            assert not (HERE / 'inputs' / f'{sid}_state.job.json').exists() and not (HERE / 'run' / f'{sid}_state').exists()
        treatment = {**baseline, 'entry_frame': first if state == 'INSIDE' else baseline['entry_frame']}
        assert all(treatment[k] == baseline[k] for k in ('collision_frame', 'entry_side', 'evasion_space'))
        for pred in (baseline, treatment):
            assert set(pred) == {'collision_frame', 'entry_frame', 'entry_side', 'evasion_space'}
            assert all(type(pred[k]) is int and pred[k] in times for k in ('collision_frame', 'entry_frame'))
            assert pred['entry_side'] in ('LEFT', 'RIGHT') and type(pred['evasion_space']) is int and pred['evasion_space'] in (0, 1)
        row = dict(ID=sid, source_group=source['source_group'], baseline=baseline, treatment=treatment, mark_approved=ref['mark_approved'],
                   state=state, state_raw=raw, state_reference=ref['first_frame_state'],
                   state_reference_correct=(state == ref['first_frame_state']) if ref['mark_approved'] and ref['first_frame_state'] != 'UNKNOWN' else None,
                   state_prompt_tokens=call['prompt_tokens'] if call else None, overridden=state == 'INSIDE',
                   entry_changed=baseline['entry_frame'] != treatment['entry_frame'], other_three_unchanged=True, entry_reference=ref['entry'], timing=None,
                   known_outside_inside_response=ref['first_frame_state'] == 'OUTSIDE' and state == 'INSIDE',
                   known_outside_new_first=ref['first_frame_state'] == 'OUTSIDE' and state == 'INSIDE' and baseline['entry_frame'] != first)
        truth = ref['entry']
        if truth['evaluation_eligible']:
            assert truth['status'] in ('before_start', 'during_clip')
            lo, hi = times[truth['lower_frame']], times[truth['upper_frame']]
            assert lo <= hi
            if truth['status'] == 'before_start':
                assert truth['lower_frame'] == truth['upper_frame'] == first
            a, b = times[baseline['entry_frame']], times[treatment['entry_frame']]
            base_grade, new_grade = rational_grade(a, lo, hi), rational_grade(b, lo, hi)
            changes = [abs(b - t) - abs(a - t) for t in (lo, hi)]
            row['timing'] = dict(interval_seconds=[float(lo), float(hi)], baseline_seconds=float(a), treatment_seconds=float(b),
                                 baseline=base_grade, treatment=new_grade, paired_accuracy_delta=rational_paired_delta(a, b, lo, hi),
                                 new_false_first=truth['status'] == 'during_clip' and b == times[first] and a != times[first],
                                 definite_gain=base_grade['result'] == 'wrong' and new_grade['result'] == 'correct',
                                 definite_loss=base_grade['result'] == 'correct' and new_grade['result'] == 'wrong',
                                 paired_MAE_delta_seconds=[float(min(changes)), float(max(changes))])
        rows.append(row)
    same(evaluation['rows'], rows)
    eligible = [r for r in rows if r['timing'] is not None]
    n = len(eligible)
    metrics = {arm: independent_aggregate([r['timing'][arm] for r in eligible]) for arm in ('baseline', 'treatment')}
    strata = {kind: {arm: independent_aggregate([r['timing'][arm] for r in eligible if r['entry_reference']['status'] == kind])
                     for arm in ('baseline', 'treatment')} for kind in ('before_start', 'during_clip')}
    bounds = {name: [sum(r['timing'][field][i] for r in eligible) / n for i in (0, 1)] if n else None
              for name, field in (('accuracy', 'paired_accuracy_delta'), ('mae_seconds', 'paired_MAE_delta_seconds'))}
    gate = dict(both_strata=all(strata[k]['baseline']['n'] >= 1 for k in strata), gains=sum(r['timing']['definite_gain'] for r in eligible),
                losses=sum(r['timing']['definite_loss'] for r in eligible), possible_loss_cases=sum(r['timing']['paired_accuracy_delta'][0] < 0 for r in eligible),
                new_false_first=sum(r['timing']['new_false_first'] for r in eligible), known_outside_new_first=sum(r['known_outside_new_first'] for r in rows),
                paired_MAE_nonincrease=bounds['mae_seconds'] is not None and bounds['mae_seconds'][1] <= 1e-9)
    gate['pass'] = gate['both_strata'] and gate['gains'] >= 1 and gate['losses'] == gate['possible_loss_cases'] == gate['new_false_first'] == gate['known_outside_new_first'] == 0 and gate['paired_MAE_nonincrease']
    for key, value in (('metrics', metrics), ('strata', strata), ('paired_delta_bounds', bounds), ('gate', gate)):
        same(evaluation[key], value)
    counts = dict(actual_model_calls=total_calls, selected_cases=12, entry_eligible=n, state_queried=len(state_inputs),
                  state_reference_scored=sum(r['state_reference_correct'] is not None for r in rows),
                  state_reference_correct=sum(r['state_reference_correct'] is True for r in rows),
                  known_outside_queried=sum(r['mark_approved'] and r['state_reference'] == 'OUTSIDE' for r in rows),
                  known_outside_inside_responses=sum(r['known_outside_inside_response'] for r in rows), other_three_unchanged=12)
    for key, value in counts.items():
        same(evaluation[key], value)
    assert total_calls == 48 + len(state_inputs) and evaluation['official_S2'] is None

    tensor_rows = []
    if state_inputs:
        os.environ.update(HF_HUB_OFFLINE='1', HF_HUB_DISABLE_IMPLICIT_TOKEN='1', TOKENIZERS_PARALLELISM='false')
        import socket
        attempts = []
        def deny(*args, **kwargs):
            attempts.append(True)
            raise RuntimeError('Processor-only audit attempted network')
        socket.socket.connect = socket.socket.connect_ex = socket.create_connection = deny
        import mlx.core as mx
        mx.set_default_device(mx.cpu)
        import torch
        torch.set_num_threads(2)
        from mlx_vlm.utils import load_processor, load_image_processor, prepare_inputs
        model_path = ROOT / 'artifacts/mac_experiments/stage2_mlx/model'
        cfg = read(model_path / 'config.json')
        processor = load_processor(model_path, True, eos_token_ids=cfg.get('eos_token_id'), trust_remote_code=False, local_files_only=True)
        optional = load_image_processor(model_path, trust_remote_code=False, local_files_only=True)
        if optional is not None:
            processor.image_processor = optional
        for sid, job, call in state_inputs:
            with Image.open(job['image']) as image:
                image = bounded(image)
            text = processor.apply_chat_template([{'role': 'user', 'content': [{'type': 'image'}, {'type': 'text', 'text': prompt}]}], tokenize=False, add_generation_prompt=True)
            inputs = prepare_inputs(processor, images=[image], prompts=text, image_token_index=cfg.get('image_token_index') or cfg['image_token_id'], add_special_tokens=True)
            mx.eval(*[v for v in inputs.values() if isinstance(v, mx.array)])
            arrays = {k: np.asarray(v) for k, v in inputs.items() if isinstance(v, mx.array)}
            processor_hashes = {k: hashlib.sha256(v.tobytes()).hexdigest() for k, v in arrays.items()}
            assert processor_hashes == call['processor_input_sha256'] and hashlib.sha256(text.encode()).hexdigest() == call['prompt_sha256']
            visual_tokens = int(np.sum(arrays['input_ids'] == cfg['image_token_id']))
            grid = arrays['image_grid_thw'].tolist()
            assert visual_tokens == sum(math.prod(x) // processor.image_processor.merge_size ** 2 for x in grid)
            assert arrays['input_ids'].shape[-1] == call['prompt_tokens']
            tensor_rows.append(dict(ID=sid, image_grid_thw=grid, image_tokens=visual_tokens, input_tokens=call['prompt_tokens'],
                                   tensor_shapes={k: list(v.shape) for k, v in arrays.items()}, hashes_exact=True))
        assert not attempts and str(mx.default_device()) == 'Device(cpu, 0)'
    for name, digest in frozen['files'].items():
        assert sha(ROOT / name) == digest, name
    for name, digest in hashes.items():
        assert sha(ROOT / name) == digest, name
    summary = dict(status='PASS', created_utc=datetime.now(timezone.utc).isoformat(), frozen_files_verified_pre_post=len(frozen['files']),
                   previous_freeze_preserved=len(old_files), source_cases=12, source_png_pts_checked=600,
                   baseline_calls_replayed=48, state_processor_replayed=len(tensor_rows), verifier_model_loads=0, verifier_model_calls=0,
                   counts=counts, metrics=metrics, strata=strata, paired_delta_bounds=bounds, gate=gate, rows=rows,
                   state_processor_metadata=tensor_rows, resources=resources, artifact_sha256=hashes,
                   limits=['Source-group disjointness and a bounded overlap screen do not certify universal incident independence.',
                           'AI adjudicated interval/state references and privileged counterpart marks are not certified human or official GT.',
                           'Timing-unknown cases remain outside timing scores but within marker/state and known-OUTSIDE audits.',
                           'Baseline uses four shared real calls per case; treatment is a deterministic entry-only override.',
                           'CPU replay validates recorded execution and exact processor hashes, not visual ground truth or model reasoning.',
                           'Mac only; no CUDA equivalence, automatic target detection, production deployment or official S2 claim.'])
    targets[0].write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n')
    lines = ['# New-source first-frame policy independent verification', '', '**PASS.** No new model load or inference by this CPU verifier.', '',
             f"Frozen files: {len(frozen['files'])}; prior969 preserved;600 source PNG/native PTS checked;48 baseline calls replayed;{len(tensor_rows)} state processor inputs recreated exactly.",
             f'Counts: {counts}.', f'Gate: {gate}.', f'Shared-truth paired delta bounds: {bounds}.', '',
             '| Case | Baseline entry | Treatment entry | State | Timing reference |', '|---|---:|---:|---|---|']
    for row in rows:
        label = str(row['entry_reference'])
        lines.append(f"| {row['ID']} | {row['baseline']['entry_frame']} | {row['treatment']['entry_frame']} | {row['state']} | {label} |")
    lines += ['', 'All policy branches, unchanged contact/side/space outputs, fixed eligible denominators, before/during strata, interval grades and known-OUTSIDE audits were independently recomputed.',
              'Timing grades use rational seconds; paired accuracy uses closed-interval set differences and paired MAE extrema use shared-target endpoints.', '', '## Limits', '']
    lines += ['- ' + text for text in summary['limits']]
    lines += ['', 'Frozen files and parent results were not changed. Full per-case scores, processor metadata, resources and hashes are in independent_verification.json.']
    targets[1].write_text('\n'.join(lines) + '\n')
    print(json.dumps({k: summary[k] for k in ('status', 'frozen_files_verified_pre_post', 'counts', 'metrics', 'paired_delta_bounds', 'gate')}, ensure_ascii=False))


if __name__ == '__main__':
    main()
