"""Independent CPU verification of frozen resolution inputs, records and processor tensors."""
import ast
from datetime import datetime, timezone
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
MARKER = ROOT / 'artifacts/stage2_entry_marker_20260920'
ARMS = ('low', 'upsampled', 'high')


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


source = OLD / 'verify_runtime_score.py'
nodes = [n for n in ast.parse(source.read_text()).body if isinstance(n, ast.FunctionDef) and n.name == 'same']
exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source), 'exec'))


def main():
    # Parent run completion is required before any heavy hashes or processor work.
    report, evaluation = [read(HERE / name) for name in ('run/report.json', 'evaluation.json')]
    assert report['status'] == evaluation['status'] == 'complete'
    targets = [HERE / ('independent_verification' + suffix) for suffix in ('.json', '.md')]
    assert not any(path.exists() for path in targets)
    import numpy as np
    from PIL import Image, ImageDraw
    frozen, protocol = read(HERE / 'freeze.json'), read(HERE / 'protocol.json')
    assert frozen['status'] == 'locked_before_resolution_predictions'
    ids = protocol['cases']
    assert len(ids) == len(set(ids)) == 5 and protocol['arms'] == list(ARMS)
    prior_freeze = read(LEGEND / 'freeze.json')['files']
    assert len(prior_freeze) == 909 and all(frozen['files'].get(k) == v for k, v in prior_freeze.items())
    for name, digest in frozen['files'].items():
        assert sha(ROOT / name) == digest, name
    freeze_time = datetime.fromisoformat(frozen['created_utc'])
    assert freeze_time < datetime.fromisoformat(report['started_utc']) < datetime.fromisoformat(report['ended_utc'])
    assert report['freeze_sha256'] == sha(HERE / 'freeze.json')
    assert [(r['ID'], r['arm']) for r in report['workers']] == [(sid, arm) for sid in ids for arm in ARMS]
    assert all(r['exit_status'] == 0 for r in report['workers'])
    artifact_hashes = {str(Path(__file__).relative_to(ROOT)): sha(Path(__file__)), str(source.relative_to(ROOT)): sha(source)}
    for path in (HERE / 'freeze.json', HERE / 'run/report.json', HERE / 'evaluation.json'):
        artifact_hashes[str(path.relative_to(ROOT))] = sha(path)
    sources = {r['ID']: r for r in read(OLD / 'ccd_intake/inputs.json')}
    original_marks = {r['ID']: r for r in read(MARKER / 'input_checks.json')['cases']}
    input_checks = {r['ID']: r for r in read(HERE / 'input_checks.json')['cases']}
    references = []
    for who in ('a', 'b'):
        review = read(LEGEND / f'state_reference_{who}.json')
        assert review['human_review'] is False and review['model_predictions_seen'] is False
        references += review['cases']
        visual = read(HERE / f'visual_review_{who}.json')
        assert visual['status'] == 'PASS' and visual['new_predictions_seen'] is False
    assert len(references) == 5 and {r['ID'] for r in references} == set(ids)
    references = {r['ID']: r for r in references}
    records, rows, input_rows, resources = {}, [], [], []
    for sid in ids:
        meta, source_case = input_checks[sid], sources[sid]
        item = source_case['images'][0]
        assert item['frame'] == meta['source_frame']['frame'] == 0
        original = ROOT / item['path']
        assert Path(meta['source_frame']['path']).resolve() == original.resolve()
        assert sha(original) == item['sha256'] == meta['source_frame']['sha256']
        old_job = read(LEGEND / 'inputs' / f'{sid}_state.job.json')
        with Image.open(old_job['image']) as image:
            low = image.convert('RGB')
        with Image.open(original) as image:
            native = image.convert('RGB')
        assert low.size == (384, 256) and native.size == (1280, 720)
        prior_box = original_marks[sid]['boxes'][0]
        assert prior_box['frame'] == 0 and prior_box['box'] == meta['marker_low_box']
        mask = Image.new('1', low.size)
        ImageDraw.Draw(mask).rectangle(prior_box['box'], outline=1, width=2)
        mask = mask.resize((1152, 768), Image.Resampling.NEAREST)
        up = low.resize((1152, 768), Image.Resampling.NEAREST)
        high = up.copy()
        high.paste(native.resize((1152, 648), Image.Resampling.BICUBIC), (0, 84))
        high.paste((255, 255, 0), (0, 0, 1152, 768), mask)
        with Image.open(HERE / 'inputs' / f'{sid}_marker_mask.png') as saved:
            assert np.array_equal(np.asarray(saved), np.asarray(mask))
        difference = np.any(np.asarray(up) != np.asarray(high), axis=2)
        m = np.asarray(mask, dtype=bool)
        scene = np.zeros(difference.shape, dtype=bool)
        scene[84:732, :] = True
        assert difference.any() and not np.any(difference & (~scene | m))
        assert np.array_equal(np.asarray(up)[m], np.asarray(high)[m])
        assert np.all(np.asarray(high)[m] == [255, 255, 0])
        assert int(difference.sum()) == meta['source_detail_difference_pixels']
        target = references[sid]['state']
        assert target in ('INSIDE', 'OUTSIDE', 'UNKNOWN')
        row = dict(ID=sid, reference=target, eligible=target != 'UNKNOWN', arms={})
        for arm, expected in zip(ARMS, (low, up, high), strict=True):
            folder = HERE / 'run' / f'{sid}_{arm}'
            job = read(HERE / 'inputs' / f'{sid}_{arm}.job.json')
            worker, result, calls = [read(folder / name) for name in ('worker_report.json', 'result.json', 'calls.json')]
            assert set(job) == {'ID', 'image', 'sha256', 'prompt', 'max_new_tokens'}
            assert job['ID'] == worker['ID'] == result['ID'] == sid and worker['arm'] == 'state'
            assert job['prompt'] == old_job['prompt'] and job['max_new_tokens'] == old_job['max_new_tokens'] == 40
            assert sha(Path(job['image'])) == job['sha256']
            with Image.open(job['image']) as image:
                image = image.convert('RGB')
                assert image.size == expected.size and image.tobytes() == expected.tobytes()
                assert bounded(image).tobytes() == image.tobytes()
            assert rgb_sha(expected) == meta[arm + '_rgb_sha256']
            assert worker['status'] == 'complete' and worker['network_attempts'] == 0
            assert worker['model_calls'] == len(calls) == 1 and calls[0] == result['call']
            assert freeze_time < datetime.fromisoformat(worker['started_utc']) <= datetime.fromisoformat(worker['ended_utc'])
            call = calls[0]
            assert call['text'].strip() == result['raw'].strip() and call['prompt'] == job['prompt']
            assert call['max_new_tokens'] == 40 and call['deepstack_fix'] is True
            assert call['compute_dtype'] == 'native' and call['decode_mode'] == 'sync'
            assert call['image_sha256'] == [rgb_sha(expected)] and call['image_sizes'] == [list(expected.size)]
            assert call['token_trace'] and call['processor_input_sha256']
            try:
                parsed = json.loads(result['raw'])
            except (ValueError, TypeError):
                parsed = None
            value = parsed.get('lane_state') if isinstance(parsed, dict) else None
            valid = isinstance(value, str) and value in ('INSIDE', 'OUTSIDE', 'UNCERTAIN')
            prediction = value if valid else None
            assert result['state'] == prediction and result['valid_enum'] == valid
            row['arms'][arm] = dict(prediction=prediction, valid_enum=valid, raw=result['raw'],
                                   correct=(valid and value == target) if target != 'UNKNOWN' else None,
                                   prompt_tokens=call['prompt_tokens'], image_sizes=call['image_sizes'])
            records[(sid, arm)] = (job, call)
            parent_worker = next(w for w in report['workers'] if w['ID'] == sid and w['arm'] == arm)
            assert parent_worker['wall_seconds'] >= worker['wall_seconds'] > 0 and call['peak_memory'] > 0
            resources.append(dict(ID=sid, arm=arm, parent_wall_seconds=parent_worker['wall_seconds'], worker_wall_seconds=worker['wall_seconds'],
                                  generation_seconds=call['seconds'], mlx_peak_GB=call['peak_memory'], rss_bytes=worker['rss_bytes']))
            for name in ('worker_report.json', 'calls.json', 'result.json'):
                artifact_hashes[str((folder / name).relative_to(ROOT))] = sha(folder / name)
        history = read(LEGEND / 'state_run' / f'{sid}_state/result.json')
        assert records[(sid, 'low')][1]['processor_input_sha256'] == history['call']['processor_input_sha256']
        row['historical_low_raw_and_enum_match'] = row['arms']['low']['raw'].strip() == history['raw'].strip() and row['arms']['low']['prediction'] == history['state']
        hp, up_hash = (records[(sid, arm)][1]['processor_input_sha256'] for arm in ('high', 'upsampled'))
        assert set(hp) == set(up_hash) and {k for k in hp if hp[k] != up_hash[k]} == {'pixel_values'}
        row['high_vs_upsampled_same_text_and_grid'] = True
        rows.append(row)
        input_rows.append(dict(ID=sid, source_detail_difference_pixels=int(difference.sum()), source_frame_sha256=sha(original),
                               low_size=list(low.size), high_size=list(high.size), header_padding_marker_preserved=True))
    same(evaluation['rows'], rows)
    eligible = [r for r in rows if r['eligible']]
    columns = ['INSIDE', 'OUTSIDE', 'UNCERTAIN', None]
    metrics = {}
    for arm in ARMS:
        correct = sum(r['arms'][arm]['correct'] for r in eligible)
        matrix = [[sum(r['reference'] == target and r['arms'][arm]['prediction'] == pred for r in eligible) for pred in columns]
                  for target in ('INSIDE', 'OUTSIDE')]
        metrics[arm] = dict(n=len(eligible), correct=correct, accuracy=correct / len(eligible) if eligible else None,
                            confusion=dict(reference_rows=['INSIDE', 'OUTSIDE'], prediction_columns=columns, matrix=matrix),
                            uncertain=sum(r['arms'][arm]['prediction'] == 'UNCERTAIN' for r in eligible),
                            invalid=sum(not r['arms'][arm]['valid_enum'] for r in eligible))
    same(evaluation['metrics'], metrics)
    comparisons = {}
    for control, treatment in (('low', 'high'), ('low', 'upsampled'), ('upsampled', 'high')):
        gains = [r['ID'] for r in eligible if not r['arms'][control]['correct'] and r['arms'][treatment]['correct']]
        losses = [r['ID'] for r in eligible if r['arms'][control]['correct'] and not r['arms'][treatment]['correct']]
        comparisons[control + '_to_' + treatment] = dict(gained_ids=gains, lost_ids=losses, accuracy_delta=(len(gains) - len(losses)) / len(eligible) if eligible else None)
    same(evaluation['comparisons'], comparisons)
    primary = comparisons['low_to_high']
    gate = dict(gains=len(primary['gained_ids']), losses=len(primary['lost_ids']),
                inside_gains=sum(r['reference'] == 'INSIDE' and r['ID'] in primary['gained_ids'] for r in rows),
                historical_control_match=all(r['historical_low_raw_and_enum_match'] for r in rows))
    gate['pass'] = gate['gains'] >= 1 and gate['inside_gains'] >= 1 and gate['losses'] == 0 and gate['historical_control_match']
    same(evaluation['gate'], gate)
    assert evaluation['actual_model_calls'] == len(records) == 15 and evaluation['official_S2'] is None

    # Recreate the exact preprocessing path on CPU, without mlx_vlm.load/any model.
    os.environ.update(HF_HUB_OFFLINE='1', HF_HUB_DISABLE_IMPLICIT_TOKEN='1', TOKENIZERS_PARALLELISM='false')
    import socket
    network_attempts = []
    def deny(*args, **kwargs):
        network_attempts.append(True)
        raise RuntimeError('Processor-only verification attempted network')
    socket.socket.connect = socket.socket.connect_ex = socket.create_connection = deny
    import mlx.core as mx
    mx.set_default_device(mx.cpu)
    import torch
    torch.set_num_threads(2)
    from mlx_vlm.utils import load_processor, load_image_processor, prepare_inputs
    model_path = ROOT / 'artifacts/mac_experiments/stage2_mlx/model'
    config = read(model_path / 'config.json')
    processor = load_processor(model_path, True, eos_token_ids=config.get('eos_token_id'), trust_remote_code=False, local_files_only=True)
    optional_image_processor = load_image_processor(model_path, trust_remote_code=False, local_files_only=True)
    if optional_image_processor is not None:
        processor.image_processor = optional_image_processor
    tensor_rows = []
    for sid in ids:
        for arm in ARMS:
            job, call = records[(sid, arm)]
            with Image.open(job['image']) as image:
                image = bounded(image)
            text = processor.apply_chat_template([{'role': 'user', 'content': [{'type': 'image'}, {'type': 'text', 'text': job['prompt']}]}],
                                                 tokenize=False, add_generation_prompt=True)
            inputs = prepare_inputs(processor, images=[image], prompts=text,
                                    image_token_index=config.get('image_token_index') or config['image_token_id'], add_special_tokens=True)
            mx.eval(*[v for v in inputs.values() if isinstance(v, mx.array)])
            arrays = {k: np.asarray(v) for k, v in inputs.items() if isinstance(v, mx.array)}
            hashes = {k: hashlib.sha256(v.tobytes()).hexdigest() for k, v in arrays.items()}
            assert hashes == call['processor_input_sha256'], (sid, arm)
            assert hashlib.sha256(text.encode()).hexdigest() == call['prompt_sha256']
            grid = arrays['image_grid_thw'].tolist()
            tokens = int(np.sum(arrays['input_ids'] == config['image_token_id']))
            merge = processor.image_processor.merge_size
            assert tokens == sum(math.prod(g) // (merge * merge) for g in grid)
            assert arrays['input_ids'].shape == arrays['attention_mask'].shape
            assert arrays['input_ids'].shape[-1] == call['prompt_tokens']
            tensor_rows.append(dict(ID=sid, arm=arm, image_grid_thw=grid, image_token_count=tokens,
                                    total_input_tokens=arrays['input_ids'].shape[-1], nonimage_tokens=int(arrays['input_ids'].size) - tokens,
                                    tensor_shapes={k: list(v.shape) for k, v in arrays.items()},
                                    tensor_dtypes={k: str(v.dtype) for k, v in arrays.items()},
                                    all_recorded_hashes_match=True, processor_input_sha256=hashes))
    assert str(mx.default_device()) == 'Device(cpu, 0)' and not network_attempts
    for sid in ids:
        low_t, up_t, high_t = [next(r for r in tensor_rows if r['ID'] == sid and r['arm'] == arm) for arm in ARMS]
        assert up_t['image_grid_thw'] == high_t['image_grid_thw']
        assert up_t['image_token_count'] == high_t['image_token_count'] and up_t['total_input_tokens'] == high_t['total_input_tokens']
        assert low_t['nonimage_tokens'] == up_t['nonimage_tokens'] == high_t['nonimage_tokens']
    for name, digest in frozen['files'].items():
        assert sha(ROOT / name) == digest, name
    for name, digest in artifact_hashes.items():
        assert sha(ROOT / name) == digest, name
    per_arm_resources = {arm: dict(worker_wall_sum_seconds=sum(r['parent_wall_seconds'] for r in resources if r['arm'] == arm),
                                   mlx_peak_max_GB=max(r['mlx_peak_GB'] for r in resources if r['arm'] == arm),
                                   rss_max_bytes=max(r['rss_bytes'] for r in resources if r['arm'] == arm)) for arm in ARMS}
    constant_baselines = {label: sum(r['reference'] == label for r in eligible) / len(eligible) if eligible else None for label in ('INSIDE', 'OUTSIDE')}
    summary = dict(status='PASS', created_utc=datetime.now(timezone.utc).isoformat(), frozen_files_verified_pre_post=len(frozen['files']),
                   prior_freeze_preserved=len(prior_freeze), actual_model_calls_verified=15, verifier_model_loads=0, verifier_model_calls=0,
                   verifier_device=str(mx.default_device()), verifier_network_attempts=len(network_attempts), input_reconstruction=input_rows,
                   rows=rows, metrics=metrics, comparisons=comparisons, gate=gate, constant_baseline_accuracy=constant_baselines,
                   processor_replay=dict(processor_class=type(processor).__name__, image_processor_class=type(processor.image_processor).__name__,
                                         cases=tensor_rows, all_fifteen_hashes_exact=True),
                   resources=resources, per_arm_resources=per_arm_resources, artifact_sha256=artifact_hashes,
                   limits=['Five exposed AI references with privileged counterpart marking; semantic label validity is not certified.',
                           'High uses 90 percent native width; this is not untouched native-resolution input.',
                           'Low-to-high changes source rendering, grid and token count jointly; high-to-upsampled holds grid/tokens fixed.',
                           'Source detail, blur and aliasing are not causally separated by the equal-size control.',
                           'Categorical first-frame state does not validate first-wheel-contact timing or automatic frame-zero correction.',
                           'Processor tensors were independently regenerated on CPU; model inference itself was not repeated.',
                           'Python socket checks are narrower than OS networking; Mac does not establish CUDA equivalence; official S2 was not computed.'])
    targets[0].write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n')
    lines = ['# State resolution independent verification', '', '**PASS.** Frozen inputs, actual calls, categorical scoring and CPU processor reconstruction agree.', '',
             f"Frozen files before/after: {len(frozen['files'])}; prior freeze preserved: {len(prior_freeze)}; actual workers/calls: 15/15.",
             'All three input recipes were reconstructed; original frame0 and frozen marker-box lineage match. High changes scene pixels outside the marker only.',
             'Fresh low raw/enum controls match all five historical state results. No model was loaded or called by this verifier.', '',
             '| Case | AI reference | Low | Upsampled | High |', '|---|---|---|---|---|']
    for row in rows:
        predictions = [str(row['arms'][arm]['prediction']) for arm in ARMS]
        lines.append('| ' + ' | '.join([row['ID'], row['reference']] + predictions) + ' |')
    lines += ['', '| Arm | Correct / eligible | Grid t,h,w | Image tokens | Total input tokens |', '|---|---:|---|---:|---:|']
    for arm in ARMS:
        meta = next(r for r in tensor_rows if r['arm'] == arm)
        lines.append(f"| {arm} | {metrics[arm]['correct']}/{metrics[arm]['n']} | {meta['image_grid_thw']} | {meta['image_token_count']} | {meta['total_input_tokens']} |")
    lines += ['', f'Gate: {gate}.', f'Constant-label accuracies: {constant_baselines}.',
              'All 15 recreated processor tensor hashes and templated prompt hashes exactly match actual calls. MLX CPU was selected; no model weights were loaded and no new model inference occurred.',
              'High and upsampled have identical input_ids, attention masks and image grids; their pixel tensors differ. Invalid/UNCERTAIN answers remain incorrect for eligible fixed references.', '', '## Limits', '']
    lines += ['- ' + item for item in summary['limits']]
    lines += ['', 'Source, frozen reference, model, worker and parent evaluation files were not changed. Exact per-call metadata, resource measurements and hashes are in independent_verification.json.']
    targets[1].write_text('\n'.join(lines) + '\n')
    print(json.dumps({key: summary[key] for key in ('status', 'frozen_files_verified_pre_post', 'actual_model_calls_verified', 'metrics', 'gate', 'per_arm_resources')}, ensure_ascii=False))


if __name__ == '__main__':
    main()
