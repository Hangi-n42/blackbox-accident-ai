"""Independent detail-control audit. CPU preprocessing only; never model inference."""
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sys

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PREV = ROOT / 'artifacts/stage2_wheel_state_540_20260920'
ARMS = ('upsampled', 'source_detail')
spec = importlib.util.spec_from_file_location('prior_independent_wheel_audit', PREV / 'verify.py')
prior = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prior)
read, sha, rgb_sha, bounded = prior.read, prior.sha, prior.rgb_sha, prior.bounded


def audit_inputs():
    import numpy as np
    from PIL import Image, ImageDraw
    old_protocol, old_refs, old_inputs = prior.audit_inputs()
    protocol = read(HERE / 'protocol.json')
    assert (HERE / 'references.json').read_bytes() == (PREV / 'references.json').read_bytes()
    assert protocol['frames'] == [31, 35, 39] and protocol['arms'] == list(ARMS)
    assert protocol['case'] == old_protocol['case'] == 'CCD_000540'
    assert protocol['prompt'] == old_protocol['prompt'] and protocol['max_new_tokens'] == 40
    source = next(r for r in read(ROOT / protocol['source_manifest']) if r['ID'] == protocol['case'])
    old_checks = {r['frame']: r for r in read(PREV / 'input_checks.json')['cases']}
    current_checks = {r['frame']: r for r in read(HERE / 'input_checks.json')['cases']}
    prepared, audit = {}, []
    for ref in old_refs:
        frame = ref['frame']
        assert ref['mark_approved']
        old_job, upsampled = old_inputs[(frame, 'upsampled')]
        mask = Image.new('1', (384, 256))
        ImageDraw.Draw(mask).rectangle(old_checks[frame]['marker_low_box'], outline=1, width=2)
        mask = mask.resize((1152, 768), Image.Resampling.NEAREST)
        item = next(r for r in source['images'] if r['frame'] == frame)
        path = ROOT / item['path']
        assert sha(path) == item['sha256']
        with Image.open(path) as native:
            assert native.size == (1280, 720)
            scene = native.convert('RGB').resize((1152, 648), Image.Resampling.BICUBIC)
        detail = upsampled.copy()
        detail.paste(scene, (0, 84))
        detail.paste((255, 255, 0), (0, 0, 1152, 768), mask)
        delta = np.any(np.asarray(upsampled) != np.asarray(detail), axis=2)
        allowed = np.zeros(delta.shape, dtype=bool)
        allowed[84:732, :] = True
        marked = np.asarray(mask, dtype=bool)
        assert delta.any() and not np.any(delta & (~allowed | marked))
        assert np.array_equal(np.asarray(upsampled)[marked], np.asarray(detail)[marked])
        with Image.open(HERE / 'inputs' / f'f{frame}_marker_mask.png') as saved:
            assert np.array_equal(np.asarray(saved), np.asarray(mask))
        check = current_checks[frame]
        assert check['source'] == item and check['source_detail_changed_pixels'] == int(delta.sum()) and check['marker_pixels'] == int(marked.sum())
        assert check['marker_low_box'] == old_checks[frame]['marker_low_box'] and check['control_file_bytes_exact'] is True
        for arm, expected in zip(ARMS, (upsampled, detail), strict=True):
            job = read(HERE / 'inputs' / f'f{frame}_{arm}.job.json')
            assert set(job) == {'ID', 'image', 'sha256', 'prompt', 'max_new_tokens'}
            assert job['ID'] == f'CCD_000540_f{frame}' and job['prompt'] == protocol['prompt'] and job['max_new_tokens'] == 40
            assert sha(Path(job['image'])) == job['sha256']
            if arm == 'upsampled':
                assert Path(job['image']).read_bytes() == Path(old_job['image']).read_bytes()
            with Image.open(job['image']) as image:
                assert image.size == expected.size == (1152, 768) and image.convert('RGB').tobytes() == expected.tobytes()
            assert bounded(expected).tobytes() == expected.tobytes()
            assert check[arm+'_rgb_sha256'] == rgb_sha(expected)
            prepared[(frame, arm)] = (job, expected)
        audit.append(dict(frame=frame, original_sha256=sha(path), original_size=[1280,720], canvas_size=[1152,768], scene_box=[0,84,1152,732], changed_pixels=int(delta.sum()), header_padding_marker_exact=True, control_file_bytes_exact=True))
    return protocol, old_refs, prepared, audit


def processor_replay(records):
    import numpy as np
    from PIL import Image
    os.environ.update(HF_HUB_OFFLINE='1', HF_HUB_DISABLE_IMPLICIT_TOKEN='1', TOKENIZERS_PARALLELISM='false')
    import socket
    attempts = []
    def deny(*args, **kwargs):
        attempts.append(True)
        raise RuntimeError('CPU processor audit attempted network')
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
    rows = []
    for (frame, arm), (job, call) in records.items():
        with Image.open(job['image']) as image:
            image = bounded(image)
        text = processor.apply_chat_template([{'role':'user','content':[{'type':'image'},{'type':'text','text':job['prompt']}]}], tokenize=False, add_generation_prompt=True)
        inputs = prepare_inputs(processor, images=[image], prompts=text, image_token_index=cfg.get('image_token_index') or cfg['image_token_id'], add_special_tokens=True)
        mx.eval(*[v for v in inputs.values() if isinstance(v,mx.array)])
        arrays = {k:np.asarray(v) for k,v in inputs.items() if isinstance(v,mx.array)}
        assert {k:hashlib.sha256(v.tobytes()).hexdigest() for k,v in arrays.items()} == call['processor_input_sha256']
        assert hashlib.sha256(text.encode()).hexdigest() == call['prompt_sha256']
        grid = arrays['image_grid_thw'].tolist()
        visual = int(np.sum(arrays['input_ids'] == cfg['image_token_id']))
        assert grid == [[1,48,72]] and visual == 864 == sum(math.prod(g)//processor.image_processor.merge_size**2 for g in grid)
        assert arrays['input_ids'].shape == arrays['attention_mask'].shape and arrays['input_ids'].shape[-1] == call['prompt_tokens'] == 985
        rows.append(dict(frame=frame, arm=arm, image_grid_thw=grid, image_tokens=visual, input_tokens=call['prompt_tokens'], nonimage_tokens=int(arrays['input_ids'].size)-visual, tensor_shapes={k:list(v.shape) for k,v in arrays.items()}, tensor_dtypes={k:str(v.dtype) for k,v in arrays.items()}, hashes_exact=True))
    assert str(mx.default_device()) == 'Device(cpu, 0)' and not attempts
    return rows


def main():
    report, evaluation = read(HERE / 'run/report.json'), read(HERE / 'evaluation.json')
    assert report['status'] == evaluation['status'] == 'complete'
    outputs = [HERE / ('independent_verification'+suffix) for suffix in ('.json','.md')]
    assert not any(p.exists() for p in outputs)
    frozen, previous = read(HERE / 'freeze.json'), read(PREV / 'freeze.json')['files']
    assert frozen['status'] == 'locked_before_equal_size_detail_predictions' and frozen['model_calls_planned'] == 6
    assert len(previous) == 1829 and all(frozen['files'].get(k) == v for k,v in previous.items())
    for name, digest in frozen['files'].items():
        assert sha(ROOT/name) == digest, name
    stamp = datetime.fromisoformat(frozen['created_utc'])
    assert stamp < datetime.fromisoformat(report['started_utc']) < datetime.fromisoformat(report['ended_utc'])
    assert report['freeze_sha256'] == sha(HERE / 'freeze.json')
    assert read(HERE / 'preflight_review.json')['status'] == read(HERE / 'input_peer_review.json')['status'] == 'PASS'
    protocol, refs, prepared, input_audit = audit_inputs()
    assert [(w['frame'],w['arm']) for w in report['workers']] == [(f,a) for f in protocol['frames'] for a in ARMS]
    assert all(w['exit_status'] == 0 for w in report['workers'])
    hashes = {str(p.relative_to(ROOT)):sha(p) for p in [Path(__file__),PREV/'verify.py',HERE/'freeze.json',HERE/'run/report.json',HERE/'evaluation.json']}
    rows, records, resources = [], {}, []
    for ref in refs:
        frame = ref['frame']
        row = dict(frame=frame, reference=ref['state'], eligible=ref['state']!='UNKNOWN', arms={})
        for arm in ARMS:
            job, image = prepared[(frame,arm)]
            folder = HERE / 'run' / f'f{frame}_{arm}'
            worker, result, calls = [read(folder/n) for n in ('worker_report.json','result.json','calls.json')]
            assert worker['ID'] == result['ID'] == job['ID'] and worker['arm'] == 'state' and worker['status'] == 'complete'
            assert worker['model_calls'] == len(calls) == 1 and worker['network_attempts'] == 0
            assert stamp < datetime.fromisoformat(worker['started_utc']) <= datetime.fromisoformat(worker['ended_utc'])
            call = calls[0]
            assert call == result['call'] and call['text'].strip() == result['raw'].strip()
            assert call['prompt'] == protocol['prompt'] and call['max_new_tokens'] == 40
            assert call['deepstack_fix'] is True and call['compute_dtype'] == 'native' and call['decode_mode'] == 'sync'
            assert call['image_sha256'] == [rgb_sha(image)] and call['image_sizes'] == [[1152,768]]
            assert call['token_trace'] and call['processor_input_sha256'] and call['peak_memory'] > 0
            try:
                obj = json.loads(result['raw'])
            except (ValueError,TypeError):
                obj = None
            value = obj.get('lane_state') if isinstance(obj,dict) else None
            valid = isinstance(value,str) and value in ('OUTSIDE','INSIDE','UNCERTAIN')
            state = value if valid else None
            assert result['state'] == state and result['valid_enum'] == valid
            row['arms'][arm] = dict(prediction=state, raw=result['raw'], valid_enum=valid, correct=(state==ref['state']) if row['eligible'] else None, prompt_tokens=call['prompt_tokens'])
            records[(frame,arm)] = (job,call)
            parent = next(w for w in report['workers'] if (w['frame'],w['arm']) == (frame,arm))
            assert parent['wall_seconds'] >= worker['wall_seconds'] > 0
            resources.append(dict(frame=frame,arm=arm,parent_wall_seconds=parent['wall_seconds'],worker_wall_seconds=worker['wall_seconds'],mlx_peak_GB=call['peak_memory'],rss_bytes=worker['rss_bytes']))
            for name in ('worker_report.json','result.json','calls.json'):
                hashes[str((folder/name).relative_to(ROOT))] = sha(folder/name)
        history = read(PREV / 'run' / f'f{frame}_upsampled/result.json')
        control, detail = records[(frame,'upsampled')][1], records[(frame,'source_detail')][1]
        row['historical_control_raw_enum_match'] = row['arms']['upsampled']['raw'].strip()==history['raw'].strip() and row['arms']['upsampled']['prediction']==history['state']
        assert control['processor_input_sha256'] == history['call']['processor_input_sha256']
        row['historical_control_processor_exact'] = True
        a, b = control['processor_input_sha256'], detail['processor_input_sha256']
        assert set(a)==set(b) and {k for k in a if a[k]!=b[k]}=={'pixel_values'}
        assert control['prompt_tokens']==detail['prompt_tokens'] and control['prompt_sha256']==detail['prompt_sha256']
        row['same_text_grid_tokens_different_pixels'] = True
        rows.append(row)
    assert evaluation['rows'] == rows
    eligible = [r for r in rows if r['eligible']]
    n, metrics = len(eligible), {}
    for arm in ARMS:
        correct = sum(r['arms'][arm]['correct'] for r in eligible)
        metrics[arm] = dict(n=n,correct=correct,accuracy=correct/n if n else None,
            distinguish_definite_outside_inside=(correct==n) if {r['reference'] for r in eligible}=={'OUTSIDE','INSIDE'} else None,
            chronological_responses=[dict(frame=r['frame'],prediction=r['arms'][arm]['prediction']) for r in rows],
            uncertain=sum(r['arms'][arm]['prediction']=='UNCERTAIN' for r in rows),invalid=sum(r['arms'][arm]['prediction'] is None for r in rows))
    gains = [r['frame'] for r in eligible if r['arms']['source_detail']['correct'] and not r['arms']['upsampled']['correct']]
    losses = [r['frame'] for r in eligible if r['arms']['upsampled']['correct'] and not r['arms']['source_detail']['correct']]
    comparison = dict(gained_frames=gains,lost_frames=losses,accuracy_delta=(len(gains)-len(losses))/n if n else None)
    gate = dict(gains=len(gains),losses=len(losses),historical_control_match=all(r['historical_control_raw_enum_match'] for r in rows),equal_text_grid_tokens=all(r['same_text_grid_tokens_different_pixels'] for r in rows))
    gate['pass'] = gate['gains']>=1 and gate['losses']==0 and gate['historical_control_match'] and gate['equal_text_grid_tokens']
    assert evaluation['metrics']==metrics and evaluation['source_detail_comparison']==comparison and evaluation['gate']==gate
    assert evaluation['actual_model_calls']==len(records)==6 and evaluation['entry_output_modified'] is False and evaluation['official_S2'] is None
    tensors = processor_replay(records)
    for name,digest in {**frozen['files'],**hashes}.items():
        assert sha(ROOT/name)==digest,name
    summary = dict(status='PASS',created_utc=datetime.now(timezone.utc).isoformat(),frozen_files_verified_pre_post=len(frozen['files']),previous_freeze_preserved=len(previous),actual_model_calls_verified=6,verifier_model_loads=0,verifier_model_calls=0,verifier_device='Device(cpu, 0)',verifier_network_attempts=0,input_reconstruction=input_audit,rows=rows,metrics=metrics,source_detail_comparison=comparison,gate=gate,processor_metadata=tensors,resources=resources,artifact_sha256=hashes,
        limits=['One exposed incident with two definite AI references and privileged counterpart marks; no independent generalization or official GT claim.', 'f35 stays UNKNOWN and unscored; no temporal entry output is produced or changed.', 'Source detail, blur and aliasing change together, while canvas, marker, header, text/grid/tokens are held equal.', 'Source scenes are90percent native width, not untouched native resolution.', 'Separate state replies do not reveal the internal cause of the old Q3 time-selection error.', 'CPU replay confirms recorded preprocessing and scoring, not model reasoning or CUDA equivalence.'])
    outputs[0].write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
    lines = ['# Equal-size source-detail independent verification','','**PASS.** No new model load or inference by the CPU verifier.','',f"Frozen files pre/post: {len(frozen['files'])}; prior1,829 preserved. Six actual calls and six CPU processor reconstructions match; only pixel_values differ between each equal-size pair.",'','| Frame | AI reference | Upsampled | Source detail | Historical control |','|---|---|---|---|---|']
    lines += [f"| {r['frame']} | {r['reference']} | {r['arms']['upsampled']['prediction']} | {r['arms']['source_detail']['prediction']} | {r['historical_control_raw_enum_match']} |" for r in rows]
    lines += ['',f'Metrics: {metrics}',f'Source-detail comparison: {comparison}',f'Diagnostic gate: {gate}','','## Limits','']+['- '+s for s in summary['limits']]
    outputs[1].write_text('\n'.join(lines)+'\n')
    print(json.dumps({k:summary[k] for k in ('status','frozen_files_verified_pre_post','actual_model_calls_verified','metrics','gate')},ensure_ascii=False))


if __name__ == '__main__':
    main()
