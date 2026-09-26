"""Independent frozen-input, recorded-call and CPU processor audit; no model inference."""
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
PREV = ROOT / 'artifacts/stage2_first_frame_policy_20260920'
HELPERS = ROOT / 'artifacts/stage2_state_resolution_20260920/verify.py'
ARMS = ('low', 'upsampled')
nodes = [n for n in ast.parse(HELPERS.read_text()).body
         if isinstance(n, ast.FunctionDef) and n.name in {'read', 'sha', 'rgb_sha', 'bounded'}]
exec(compile(ast.Module(body=nodes, type_ignores=[]), str(HELPERS), 'exec'))


def audit_inputs():
    from PIL import Image, ImageDraw, ImageOps
    sys.path.insert(0, str(ROOT / 'artifacts/submissions/verify_v6/model/stage2/code'))
    from solution import stage2_v2 as v2
    protocol, reference = read(HERE / 'protocol.json'), read(HERE / 'references.json')
    refs = reference['cases']
    assert protocol['case'] == 'CCD_000540' and protocol['frames'] == [31, 35, 39]
    assert protocol['arms'] == list(ARMS) and protocol['max_new_tokens'] == 40
    assert [r['frame'] for r in refs] == protocol['frames'] and reference['new_predictions_seen'] is False
    for name, digest in reference['independent_review_sha256'].items():
        assert sha(HERE / name) == digest, name
    reviews = [read(HERE / f'review_{who}.json') for who in ('a', 'b')]
    for review in reviews:
        assert not any(review[k] for k in ('human_review', 'new_predictions_seen', 'peer_review_seen'))
        assert review['prior_interval_known'] is True
        assert [r['frame'] for r in review['cases']] == protocol['frames']
    source = next(r for r in read(ROOT / protocol['source_manifest']) if r['ID'] == protocol['case'])
    paths = [ROOT / r['path'] for r in source['images']]
    native_pts = read(ROOT / source['pts_source'])
    assert sha(ROOT / source['pts_source']) == source['pts_sha256']
    checks = {r['frame']: r for r in read(HERE / 'input_checks.json')['cases']}
    prepared = {}
    for ref, a, b in zip(refs, reviews[0]['cases'], reviews[1]['cases'], strict=True):
        frame = ref['frame']
        expected_state = a['state'] if ref['counterpart_agreement'] and a['state'] == b['state'] else 'UNKNOWN'
        assert ref['state'] == expected_state and ref['state'] in ('INSIDE', 'OUTSIDE', 'UNKNOWN')
        assert ref['mark_approved'] == (ref['counterpart_agreement'] and a['counterpart_identified'] and b['counterpart_identified'] and a['marker']['status'] == b['marker']['status'] == 'visible')
        index = next(i for i, r in enumerate(source['images']) if r['frame'] == frame)
        meta, pts = source['images'][index], native_pts['mapping'][index]
        assert sha(paths[index]) == meta['sha256'] and checks[frame]['source'] == meta
        assert pts['frame_id'] == frame and Fraction(pts['native_pts']) * Fraction(pts['time_base']) == Fraction(str(meta['pts_seconds'])) == Fraction(frame, 10)
        if not ref['mark_approved']:
            assert not any((HERE / 'inputs' / f'f{frame}_{arm}.job.json').exists() for arm in ARMS)
            continue
        assert ref['marker_box_xyxy'] == a['marker']['box_xyxy']
        clean = v2._sheet(paths, [index], columns=1)
        low = clean.copy()
        with Image.open(paths[index]) as native:
            w, h = native.size
            iw, ih = ImageOps.contain(native, (384, 228)).size
        x0, y0, x1, y1 = ref['marker_box_xyxy']
        assert all(type(v) is int for v in (x0, y0, x1, y1)) and 0 <= x0 < x1 < w and 0 <= y0 < y1 < h
        box = [(384-iw)//2+round(x0*iw/w), 28+round(y0*ih/h), (384-iw)//2+round(x1*iw/w), 28+round(y1*ih/h)]
        assert checks[frame]['marker_source_box'] == ref['marker_box_xyxy'] and checks[frame]['marker_low_box'] == box
        ImageDraw.Draw(low).rectangle(box, outline=(255, 255, 0), width=2)
        enlarged = low.resize((1152, 768), Image.Resampling.NEAREST)
        with Image.open(HERE / 'inputs' / f'f{frame}_unmarked.png') as saved:
            assert saved.size == clean.size and saved.convert('RGB').tobytes() == clean.tobytes()
        for arm, expected in zip(ARMS, (low, enlarged), strict=True):
            job = read(HERE / 'inputs' / f'f{frame}_{arm}.job.json')
            path = HERE / 'inputs' / f'f{frame}_{arm}.png'
            assert job == dict(ID=f'CCD_000540_f{frame}', image=str(path), sha256=sha(path), prompt=protocol['prompt'], max_new_tokens=40)
            with Image.open(path) as saved:
                assert saved.size == expected.size and saved.convert('RGB').tobytes() == expected.tobytes()
            assert checks[frame][arm+'_rgb_sha256'] == rgb_sha(expected)
            assert bounded(expected).tobytes() == expected.tobytes()
            prepared[(frame, arm)] = (job, expected)
    return protocol, refs, prepared


def main():
    report, evaluation = read(HERE / 'run/report.json'), read(HERE / 'evaluation.json')
    assert report['status'] == evaluation['status'] == 'complete'
    outputs = [HERE / ('independent_verification'+suffix) for suffix in ('.json', '.md')]
    assert not any(p.exists() for p in outputs)
    frozen = read(HERE / 'freeze.json')
    previous = read(PREV / 'freeze.json')['files']
    assert len(previous) == 1793 and all(frozen['files'].get(k) == v for k, v in previous.items())
    assert frozen['status'] == 'locked_before_new_diagnostic_calls'
    for name, digest in frozen['files'].items():
        assert sha(ROOT / name) == digest, name
    stamp = datetime.fromisoformat(frozen['created_utc'])
    assert stamp < datetime.fromisoformat(report['started_utc']) < datetime.fromisoformat(report['ended_utc'])
    assert report['freeze_sha256'] == sha(HERE / 'freeze.json')
    assert read(HERE / 'preflight_review.json')['status'] == read(HERE / 'input_peer_review.json')['status'] == 'PASS'
    protocol, refs, prepared = audit_inputs()
    expected_workers = [(r['frame'], arm) for r in refs if r['mark_approved'] for arm in ARMS]
    assert [(w['frame'], w['arm']) for w in report['workers']] == expected_workers
    assert frozen['model_calls_planned'] == len(expected_workers) and all(w['exit_status'] == 0 for w in report['workers'])
    hashes = {str(p.relative_to(ROOT)): sha(p) for p in [Path(__file__), HELPERS, HERE/'freeze.json', HERE/'run/report.json', HERE/'evaluation.json']}
    rows, records, resources = [], {}, []
    for ref in refs:
        frame = ref['frame']
        row = dict(frame=frame, reference=ref['state'], mark_approved=ref['mark_approved'], eligible=ref['mark_approved'] and ref['state'] != 'UNKNOWN', arms={})
        if ref['mark_approved']:
            for arm in ARMS:
                job, image = prepared[(frame, arm)]
                folder = HERE / 'run' / f'f{frame}_{arm}'
                worker, result, calls = [read(folder / n) for n in ('worker_report.json', 'result.json', 'calls.json')]
                assert worker['ID'] == result['ID'] == job['ID'] and worker['arm'] == 'state' and worker['status'] == 'complete'
                assert worker['model_calls'] == len(calls) == 1 and worker['network_attempts'] == 0
                assert stamp < datetime.fromisoformat(worker['started_utc']) <= datetime.fromisoformat(worker['ended_utc'])
                call = calls[0]
                assert result['call'] == call and call['text'].strip() == result['raw'].strip()
                assert call['prompt'] == protocol['prompt'] and call['max_new_tokens'] == 40
                assert call['deepstack_fix'] is True and call['compute_dtype'] == 'native' and call['decode_mode'] == 'sync'
                assert call['image_sha256'] == [rgb_sha(image)] and call['image_sizes'] == [list(image.size)]
                assert call['token_trace'] and call['processor_input_sha256'] and call['peak_memory'] > 0
                try:
                    obj = json.loads(result['raw'])
                except (ValueError, TypeError):
                    obj = None
                value = obj.get('lane_state') if isinstance(obj, dict) else None
                valid = isinstance(value, str) and value in ('INSIDE', 'OUTSIDE', 'UNCERTAIN')
                state = value if valid else None
                assert result['state'] == state and result['valid_enum'] == valid
                row['arms'][arm] = dict(prediction=state, raw=result['raw'], valid_enum=valid, correct=(state == ref['state']) if row['eligible'] else None, prompt_tokens=call['prompt_tokens'])
                records[(frame, arm)] = (job, call)
                parent = next(w for w in report['workers'] if (w['frame'], w['arm']) == (frame, arm))
                assert parent['wall_seconds'] >= worker['wall_seconds'] > 0
                resources.append(dict(frame=frame, arm=arm, parent_wall_seconds=parent['wall_seconds'], worker_wall_seconds=worker['wall_seconds'], mlx_peak_GB=call['peak_memory'], rss_bytes=worker['rss_bytes']))
                for name in ('worker_report.json', 'result.json', 'calls.json'):
                    hashes[str((folder/name).relative_to(ROOT))] = sha(folder/name)
        rows.append(row)
    assert evaluation['rows'] == rows
    eligible = [r for r in rows if r['eligible']]
    metrics = {}
    for arm in ARMS:
        n, correct = len(eligible), sum(r['arms'][arm]['correct'] for r in eligible)
        sequence = [dict(frame=r['frame'], prediction=r['arms'][arm]['prediction']) for r in rows if r['mark_approved']]
        outputs_enum, targets = ['OUTSIDE', 'INSIDE', 'UNCERTAIN', None], ['OUTSIDE', 'INSIDE']
        metrics[arm] = dict(n=n, correct=correct, accuracy=correct/n if n else None,
            confusion=dict(reference_rows=targets, prediction_columns=outputs_enum, matrix=[[sum(r['reference']==t and r['arms'][arm]['prediction']==v for r in eligible) for v in outputs_enum] for t in targets]),
            distinguish_definite_outside_inside=(correct==n) if {r['reference'] for r in eligible}==set(targets) else None,
            chronological_responses=sequence,
            inside_to_outside_adjacent_pairs=[[sequence[i]['frame'], sequence[i+1]['frame']] for i in range(len(sequence)-1) if (sequence[i]['prediction'],sequence[i+1]['prediction'])==('INSIDE','OUTSIDE')],
            uncertain_all_queried=sum(r['prediction']=='UNCERTAIN' for r in sequence), invalid_all_queried=sum(r['prediction'] is None for r in sequence))
    gains = [r['frame'] for r in eligible if r['arms']['upsampled']['correct'] and not r['arms']['low']['correct']]
    losses = [r['frame'] for r in eligible if r['arms']['low']['correct'] and not r['arms']['upsampled']['correct']]
    enlargement = dict(gained_frames=gains, lost_frames=losses, accuracy_delta=(len(gains)-len(losses))/len(eligible) if eligible else None)
    assert evaluation['metrics'] == metrics and evaluation['enlargement'] == enlargement
    assert evaluation['actual_model_calls'] == len(records) == frozen['model_calls_planned']
    assert evaluation['entry_output_modified'] is False and evaluation['official_S2'] is None

    # Only processor/tokenizer construction; never instantiate model weights or generate.
    import numpy as np
    from PIL import Image
    os.environ.update(HF_HUB_OFFLINE='1', HF_HUB_DISABLE_IMPLICIT_TOKEN='1', TOKENIZERS_PARALLELISM='false')
    import socket
    network_attempts = []
    def deny(*args, **kwargs):
        network_attempts.append(True)
        raise RuntimeError('Processor audit attempted network')
    socket.socket.connect = socket.socket.connect_ex = socket.create_connection = deny
    import mlx.core as mx
    mx.set_default_device(mx.cpu)
    import torch
    torch.set_num_threads(2)
    from mlx_vlm.utils import load_processor, load_image_processor, prepare_inputs
    model = ROOT / 'artifacts/mac_experiments/stage2_mlx/model'
    cfg = read(model / 'config.json')
    processor = load_processor(model, True, eos_token_ids=cfg.get('eos_token_id'), trust_remote_code=False, local_files_only=True)
    optional = load_image_processor(model, trust_remote_code=False, local_files_only=True)
    if optional is not None:
        processor.image_processor = optional
    tensors = []
    for (frame, arm), (job, call) in records.items():
        with Image.open(job['image']) as image:
            image = bounded(image)
        text = processor.apply_chat_template([{'role':'user', 'content':[{'type':'image'}, {'type':'text','text':job['prompt']}]}], tokenize=False, add_generation_prompt=True)
        inputs = prepare_inputs(processor, images=[image], prompts=text, image_token_index=cfg.get('image_token_index') or cfg['image_token_id'], add_special_tokens=True)
        mx.eval(*[v for v in inputs.values() if isinstance(v, mx.array)])
        arrays = {k:np.asarray(v) for k,v in inputs.items() if isinstance(v,mx.array)}
        assert {k:hashlib.sha256(v.tobytes()).hexdigest() for k,v in arrays.items()} == call['processor_input_sha256']
        assert hashlib.sha256(text.encode()).hexdigest() == call['prompt_sha256']
        grid = arrays['image_grid_thw'].tolist()
        visual = int(np.sum(arrays['input_ids'] == cfg['image_token_id']))
        assert visual == sum(math.prod(g)//processor.image_processor.merge_size**2 for g in grid)
        assert arrays['input_ids'].shape == arrays['attention_mask'].shape and arrays['input_ids'].shape[-1] == call['prompt_tokens']
        tensors.append(dict(frame=frame, arm=arm, image_grid_thw=grid, image_tokens=visual, input_tokens=call['prompt_tokens'], nonimage_tokens=int(arrays['input_ids'].size)-visual, tensor_shapes={k:list(v.shape) for k,v in arrays.items()}, tensor_dtypes={k:str(v.dtype) for k,v in arrays.items()}, hashes_exact=True))
    assert str(mx.default_device()) == 'Device(cpu, 0)' and not network_attempts
    for ref in refs:
        if ref['mark_approved']:
            pair = [r for r in tensors if r['frame'] == ref['frame']]
            assert len(pair) == 2 and pair[0]['nonimage_tokens'] == pair[1]['nonimage_tokens']
            assert pair[0]['image_grid_thw'] == [[1,16,24]] and pair[1]['image_grid_thw'] == [[1,48,72]]
    for name, digest in {**frozen['files'], **hashes}.items():
        assert sha(ROOT/name) == digest, name
    summary = dict(status='PASS', created_utc=datetime.now(timezone.utc).isoformat(), frozen_files_verified_pre_post=len(frozen['files']), previous_freeze_preserved=len(previous), actual_model_calls_verified=len(records), verifier_model_loads=0, verifier_model_calls=0, verifier_device=str(mx.default_device()), verifier_network_attempts=len(network_attempts), rows=rows, metrics=metrics, enlargement=enlargement, processor_metadata=tensors, resources=resources, artifact_sha256=hashes,
        limits=['One previously exposed incident and two definite AI state references; not independent performance validation.', 'f35 remains UNKNOWN and unscored regardless of its output; no point state is derived from the old entry interval.', 'Nearest enlargement jointly changes raster size and image tokens, without adding native source detail.', 'Privileged counterpart marks and AI references are not human or official ground truth.', 'Chronological response reversals are descriptive only; labels cannot identify an internal model cause.', 'No entry timing output changes or official S2; Mac does not establish CUDA equivalence.'])
    outputs[0].write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
    lines = ['# Wheel-state diagnostic independent verification', '', '**PASS.** No model was loaded or called by this CPU verifier.', '', f"Frozen files pre/post: {len(frozen['files'])}; previous1793 preserved. Actual calls: {len(records)}. Source frames/native PTS, approved A markers and both input rasters were reconstructed; every recorded processor hash matches.", '', '| Frame | AI reference | Low | Upsampled |', '|---|---|---|---|']
    lines += [f"| {r['frame']} | {r['reference']} | {r['arms'].get('low',{}).get('prediction')} | {r['arms'].get('upsampled',{}).get('prediction')} |" for r in rows]
    lines += ['', f'Metrics: {metrics}', '', f'Enlargement: {enlargement}', '', '## Limits', ''] + ['- '+x for x in summary['limits']]
    outputs[1].write_text('\n'.join(lines)+'\n')
    print(json.dumps({k:summary[k] for k in ('status','frozen_files_verified_pre_post','actual_model_calls_verified','metrics','enlargement')},ensure_ascii=False))


if __name__ == '__main__':
    main()
