"""Independent start-gate audit. CPU processor/cached-policy replay only; no model load."""
import argparse
from datetime import datetime, timezone
from fractions import Fraction as F
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
PREVIOUS = ROOT / 'artifacts/stage2_spatial_grounding_540_20260920'
CODE = ROOT / 'artifacts/submissions/verify_v6/model/stage2/code'
sys.path.insert(0, str(CODE))
from solution import stage2_v2 as v2
from solution import stage2_uncapped_jerk_v6c as policy

read = lambda p: json.loads(p.read_text())
TOL = F(3, 10)


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def rgb_sha(image):
    return hashlib.sha256(image.convert('RGB').tobytes()).hexdigest()


def bounded(image):
    from PIL import Image
    scale = min(1.0, math.sqrt(1200000 / (image.width * image.height)))
    return image.convert('RGB').resize(tuple(max(32, int(v * scale) // 32 * 32) for v in image.size), Image.Resampling.BICUBIC)


def strict_state(raw):
    def reject(value):
        raise ValueError(value)
    def unique(pairs):
        obj = {}
        for key, value in pairs:
            if key in obj:
                raise ValueError('duplicate key')
            obj[key] = value
        return obj
    try:
        obj = json.loads(raw, parse_constant=reject, object_pairs_hook=unique)
    except (ValueError, TypeError):
        return None
    return obj['lane_state'] if isinstance(obj, dict) and set(obj) == {'lane_state'} and isinstance(obj['lane_state'], str) and obj['lane_state'] in ('INSIDE', 'OUTSIDE', 'UNCERTAIN') else None


def grade(prediction, low, high):
    p, lo, hi = map(lambda x: F(str(x)), (prediction, low, high))
    best, worst = max(lo - p, F(0), p - hi), max(abs(p - lo), abs(p - hi))
    return dict(result='correct' if worst <= TOL else 'wrong' if best > TOL else 'indeterminate', minimum_error_seconds=float(best), maximum_error_seconds=float(worst))


def deltas(baseline, candidate, low, high):
    p, q, lo, hi = map(lambda x: F(str(x)), (baseline, candidate, low, high))
    points = sorted({lo, hi, *[v for v in (p - TOL, p + TOL, q - TOL, q + TOL) if lo <= v <= hi]})
    probes = points + [(a + b) / 2 for a, b in zip(points, points[1:])]
    acc = [int(abs(q - t) <= TOL) - int(abs(p - t) <= TOL) for t in probes]
    mae = [abs(q - t) - abs(p - t) for t in {lo, hi, *[v for v in (p, q) if lo <= v <= hi]}]
    return [min(acc), max(acc)], [float(min(mae)), float(max(mae))]


def coverage(frames, times, low, high):
    lo, hi = F(str(low)), F(str(high))
    centers = {f: F(str(times[str(f)])) for f in frames}
    cuts = sorted({lo, hi, *[x for t in centers.values() for x in (t - TOL, t + TOL) if lo <= x <= hi]})
    probes = cuts + [(a + b) / 2 for a, b in zip(cuts, cuts[1:])]
    return dict(all_truths_have_some_candidate=all(any(abs(t - p) <= TOL for p in centers.values()) for t in probes),
                any_truth_has_candidate=any(max(lo - t, F(0), t - hi) <= TOL for t in centers.values()),
                single_candidate_covers_entire_reference=[f for f, t in centers.items() if max(abs(t - lo), abs(t - hi)) <= TOL],
                possibly_correct_candidates=[f for f, t in centers.items() if max(lo - t, F(0), t - hi) <= TOL])


def audit_inputs(replay=False):
    from PIL import Image
    import numpy as np
    inventory = read(HERE / 'inventory.json')['cases']
    traces = read(HERE / 'path_replay.json')
    assert traces['status'] == 'PASS' and traces['model_calls'] == 0 and traces['cached_calls_replayed'] == 96
    assert len(inventory) == len(traces['rows']) == 24
    assert len({r['source']['source_group'] for r in inventory}) == 24
    trace = {r['ID']: r for r in traces['rows']}
    sources = {r['ID']: r for r in inventory}
    references = read(HERE / 'references.json')
    selected = [r['ID'] for r in inventory if r['review_selected']]
    assert len(selected) == 8 and set(selected) == {r['ID'] for r in references['cases']}
    reviews = {}
    for who in ('a', 'b'):
        p = HERE / f'review_{who}.json'
        assert sha(p) == references['review_sha256'][who]
        review = read(p)
        assert review['new_predictions_seen'] is False and review['human_review'] is False and review['current_peer_review_seen'] is False
        reviews[who] = {r['ID']: r for r in review['cases']}
    decomp = read(HERE / 'decomposition.json')
    assert decomp['status'] == 'complete'
    decomposition = {r['ID']: r for r in decomp['rows']}
    counts = dict(reviewed=8, eligible=0, unknown=0, stage_full_coverage={k: 0 for k in ('original', 'precontact', 'candidates')}, definite_correct_candidate_cases=0,
                  final={k: 0 for k in ('correct', 'wrong', 'indeterminate')}, before_start=0, during_clip=0)
    for ref in references['cases']:
        sid = ref['ID']; a, b = [reviews[w][sid] for w in ('a', 'b')]
        assert a['same_counterpart_verified'] and b['same_counterpart_verified'] and ref['same_counterpart_verified'] and ref['eligible']
        assert ref['expert_intervals'] == dict(a=a['entry'], b=b['entry'])
        assert a['entry']['status'] == b['entry']['status'] == ref['entry']['status']
        lo, hi = min(a['entry']['lower_frame'], b['entry']['lower_frame']), max(a['entry']['upper_frame'], b['entry']['upper_frame'])
        assert max(a['entry']['lower_frame'], b['entry']['lower_frame']) <= min(a['entry']['upper_frame'], b['entry']['upper_frame'])
        assert ref['entry'] == dict(status=a['entry']['status'], lower_frame=lo, upper_frame=hi)
        times = trace[sid]['original_times']; row = decomposition[sid]
        assert row['reference_seconds'] == [times[str(lo)], times[str(hi)]]
        counts['eligible'] += 1; counts[ref['entry']['status']] += 1
        for stage, key in (('original', 'all_frames'), ('precontact', 'precontact_frames'), ('candidates', 'entry_candidates')):
            answer = coverage(trace[sid][key], times, times[str(lo)], times[str(hi)])
            assert row['stages'][stage] == answer
            counts['stage_full_coverage'][stage] += answer['all_truths_have_some_candidate']
        counts['definite_correct_candidate_cases'] += bool(row['stages']['candidates']['single_candidate_covers_entire_reference'])
        final = grade(times[str(trace[sid]['final_entry'])], times[str(lo)], times[str(hi)])
        assert row['final'] == final
        counts['final'][final['result']] += 1
    assert counts == decomp['summary']
    assert counts['final'] == dict(correct=0, wrong=7, indeterminate=1) and counts['definite_correct_candidate_cases'] == 6
    audit = read(HERE / 'path_contract_review.json')
    for record in audit['code_comparison']:
        for name, digest in record['sha256_by_path'].items():
            assert sha(ROOT / name) == digest, name
    protocol = read(HERE / 'protocol.json')
    assert protocol['model_calls_planned'] == 16 and protocol['max_new_tokens'] == 40 and protocol['arms'] == ['control', 'gate']
    peer = read(HERE / 'input_peer_review.json')
    assert peer['status'] == 'PASS' and peer['new_predictions_seen'] is False and peer['model_calls'] == 0
    assert {r['ID'] for r in peer['cases']} == set(selected) and all(r['status'] == 'PASS' for r in peer['cases'])
    assert [r['ID'] for r in peer['cases'] if r['counterpart_visibility'] == 'not_identifiable_in_context'] == ['CCD_000052']
    assert protocol['preflight_identity_limitation']['ID'] == 'CCD_000052' and protocol['preflight_identity_limitation']['recorded_before_model_calls'] is True
    jobs, cached = {}, 0
    checks = {r['ID']: r for r in read(HERE / 'input_checks.json')['cases']}
    for src in inventory:
        sid = src['ID']; source = src['source']; record = trace[sid]
        assert sha(ROOT / source['source_video']) == source['source_sha256']
        assert sha(ROOT / source['pts_source']) == source['pts_sha256']
        pts = read(ROOT / source['pts_source'])
        assert pts['source_sha256'] == source['source_sha256']
        paths = [ROOT / x['path'] for x in source['images']]
        numbers = [v2._frame_number(p) for p in paths]
        assert len(numbers) == 50 and numbers == record['all_frames'] == sorted(set(numbers))
        native = {r['frame_id']: r for r in pts['mapping']}
        for image, path in zip(source['images'], paths):
            assert sha(path) == image['sha256']
            time = F(native[image['frame']]['native_pts']) * F(native[image['frame']]['time_base'])
            assert time == F(str(image['pts_seconds'])) == F(str(record['original_times'][str(image['frame'])]))
        folder = ROOT / src['baseline_folder']
        for name, digest in src['baseline_sha256'].items():
            assert sha(folder / name) == digest
        result, calls = read(folder / 'result.json'), read(folder / 'calls.json')
        assert len(calls) == 4 and result['calls'] == calls and result['baseline_prediction'] == record['baseline']
        if replay:
            class Replay:
                count = 0
                def ask(self, images, prompt, max_new_tokens):
                    call = calls[self.count]; self.count += 1
                    assert prompt == call['prompt'] and max_new_tokens == call['max_new_tokens']
                    assert [rgb_sha(bounded(im)) for im in images] == call['image_sha256']
                    assert [list(bounded(im).size) for im in images] == call['image_sizes']
                    return call['text']
            replay_vlm = Replay()
            with np.load(folder / 'motion.npz', allow_pickle=False) as motion:
                assert motion['base_scores'].dtype == motion['new_scores'].dtype == np.dtype('float32')
                prediction, diag = policy._predict_file(paths, motion['base_scores'], motion['new_scores'], replay_vlm)
            assert prediction == result['baseline_prediction'] and replay_vlm.count == 4
            assert diag['entry_candidates'] == record['entry_candidates']
            cached += replay_vlm.count
        diag = result['diagnostics']
        anchor = v2._choice(diag['collision'], 'collision_frame', paths, [numbers.index(f) for f in diag['collision_candidates']], numbers.index(diag['motion_proposal']))
        assert numbers[anchor] == record['vlm_contact_frame'] and numbers[:anchor + 1] == record['precontact_frames']
        candidates = v2._uniform_indices(0, anchor, 12)
        assert [numbers[i] for i in candidates] == record['entry_candidates']
        if not src['review_selected']:
            continue
        assert record['raw_entry_is_shown_integer'] and not record['choice_fallback_or_mapping']
        context = sorted({max(0, anchor - 2), anchor, min(len(paths) - 1, anchor + 2)})
        low = v2._sheet(paths, [0], columns=1)
        first = low.resize((1152, 768), Image.Resampling.NEAREST)
        gate = Image.new('RGB', (1536, 768), '#171717')
        gate.paste(first, (0, 0)); gate.paste(v2._sheet(paths, context, columns=1), (1152, 0))
        control = v2._sheet(paths, candidates, columns=4)
        assert rgb_sha(first) == checks[sid]['first_view_rgb_sha256'] and rgb_sha(gate) == checks[sid]['gate_rgb_sha256']
        assert checks[sid]['context_frames'] == [numbers[i] for i in context]
        for arm, expected, prompt in (('control', control, calls[2]['prompt']), ('gate', gate, protocol['prompt'])):
            job = read(HERE / 'inputs' / f'{sid}_{arm}.job.json')
            assert set(job) == {'ID', 'image', 'sha256', 'prompt', 'max_new_tokens'}
            assert job['ID'] == sid and job['prompt'] == prompt and job['max_new_tokens'] == 40
            assert sha(Path(job['image'])) == job['sha256']
            with Image.open(job['image']) as image:
                assert image.size == (1536, 768) and image.convert('RGB').tobytes() == expected.tobytes()
                assert bounded(image).tobytes() == expected.tobytes()
            if arm == 'control':
                assert [rgb_sha(expected)] == calls[2]['image_sha256'] and calls[2]['image_sizes'] == [[1536, 768]]
            jobs[(sid, arm)] = job
    assert len(jobs) == 16 and (not replay or cached == 96)
    return dict(protocol=protocol, references=references, sources=sources, traces=trace, jobs=jobs, decomposition=counts, cached_calls_replayed=cached)


def audit_run(data):
    from PIL import Image
    frozen, report = read(HERE / 'freeze.json'), read(HERE / 'run/report.json')
    previous = read(PREVIOUS / 'freeze.json')['files']
    assert all(frozen['files'].get(k) == v for k, v in previous.items())
    assert frozen['status'] == 'locked_before_automatic_start_gate' and frozen['model_calls_planned'] == 16
    for name, digest in frozen['files'].items():
        assert sha(ROOT / name) == digest, name
    stamp = datetime.fromisoformat(frozen['created_utc'])
    assert stamp < datetime.fromisoformat(report['started_utc']) < datetime.fromisoformat(report['ended_utc'])
    assert report['status'] == 'complete' and report['freeze_sha256'] == sha(HERE / 'freeze.json')
    assert read(HERE / 'preflight_review.json')['status'] == read(HERE / 'input_peer_review.json')['status'] == 'PASS'
    expected = [(ref['ID'], arm) for ref in data['references']['cases'] for arm in ('control', 'gate')]
    assert [(w['ID'], w['arm']) for w in report['workers']] == expected
    assert all(w['exit_status'] == 0 for w in report['workers'])
    records, hashes, resources = {}, {}, []
    for parent in report['workers']:
        sid, arm = parent['ID'], parent['arm']; job = data['jobs'][(sid, arm)]
        folder = HERE / 'run' / f'{sid}_{arm}'
        worker, result, calls = [read(folder / n) for n in ('worker_report.json', 'result.json', 'calls.json')]
        assert worker['ID'] == result['ID'] == job['ID'] and worker['arm'] == 'state' and worker['status'] == 'complete'
        assert worker['model_calls'] == len(calls) == 1 and worker['network_attempts'] == 0
        assert stamp < datetime.fromisoformat(worker['started_utc']) <= datetime.fromisoformat(worker['ended_utc'])
        call = calls[0]
        assert call == result['call'] and call['text'].strip() == result['raw'].strip()
        assert call['prompt'] == job['prompt'] and call['max_new_tokens'] == 40 and call['image_sizes'] == [[1536, 768]]
        assert call['deepstack_fix'] is True and call['compute_dtype'] == 'native' and call['decode_mode'] == 'sync'
        with Image.open(job['image']) as im:
            assert call['image_sha256'] == [rgb_sha(bounded(im))]
        assert 0 < len(call['token_trace']) == call['generation_tokens'] <= 40
        # Preserve the legacy worker's loose enum record; the policy uses strict_state separately.
        try:
            obj = json.loads(result['raw'])
        except (ValueError, TypeError):
            obj = None
        state = obj.get('lane_state') if isinstance(obj, dict) else None
        valid = isinstance(state, str) and state in ('INSIDE', 'OUTSIDE', 'UNCERTAIN')
        assert result['valid_enum'] == valid and result['state'] == (state if valid else None)
        if arm == 'control':
            old = read(ROOT / data['sources'][sid]['baseline_folder'] / 'calls.json')[2]
            for key in ('processor_input_sha256', 'prompt_sha256', 'prompt_tokens', 'image_sha256', 'image_sizes', 'prompt', 'max_new_tokens'):
                assert call[key] == old[key], (sid, key)
            assert result['raw'].strip() == old['text'].strip()
        assert parent['wall_seconds'] >= worker['wall_seconds'] > 0 and call['peak_memory'] > 0 and worker['rss_bytes'] > 0
        resources.append(dict(ID=sid, arm=arm, parent_wall_seconds=parent['wall_seconds'], worker_wall_seconds=worker['wall_seconds'], mlx_peak_GB=call['peak_memory'], rss_bytes=worker['rss_bytes']))
        records[(sid, arm)] = result
        for name in ('worker_report.json', 'result.json', 'calls.json'):
            hashes[str((folder / name).relative_to(ROOT))] = sha(folder / name)
    return frozen, records, hashes, resources


def audit_processors(data, records):
    import numpy as np
    from PIL import Image
    os.environ.update(HF_HUB_OFFLINE='1', HF_HUB_DISABLE_IMPLICIT_TOKEN='1', TOKENIZERS_PARALLELISM='false')
    import socket
    attempts = []
    def deny(*args, **kwargs):
        attempts.append(True)
        raise RuntimeError('Processor audit attempted network')
    socket.socket.connect = socket.socket.connect_ex = socket.create_connection = deny
    import mlx.core as mx
    mx.set_default_device(mx.cpu)
    import torch
    torch.set_num_threads(2)
    from mlx_vlm.utils import load_processor, load_image_processor, prepare_inputs
    model = ROOT / 'artifacts/mac_experiments/stage2_mlx/model'; cfg = read(model / 'config.json')
    processor = load_processor(model, True, eos_token_ids=cfg.get('eos_token_id'), trust_remote_code=False, local_files_only=True)
    optional = load_image_processor(model, trust_remote_code=False, local_files_only=True)
    if optional is not None:
        processor.image_processor = optional
    rows = []
    for (sid, arm), result in records.items():
        job = data['jobs'][(sid, arm)]; call = result['call']
        with Image.open(job['image']) as image:
            image = bounded(image)
        text = processor.apply_chat_template([{'role': 'user', 'content': [{'type': 'image'}, {'type': 'text', 'text': job['prompt']}]}], tokenize=False, add_generation_prompt=True)
        inputs = prepare_inputs(processor, images=[image], prompts=text, image_token_index=cfg.get('image_token_index') or cfg['image_token_id'], add_special_tokens=True)
        mx.eval(*[v for v in inputs.values() if isinstance(v, mx.array)])
        arrays = {k: np.asarray(v) for k, v in inputs.items() if isinstance(v, mx.array)}
        assert {k: hashlib.sha256(v.tobytes()).hexdigest() for k, v in arrays.items()} == call['processor_input_sha256']
        assert hashlib.sha256(text.encode()).hexdigest() == call['prompt_sha256']
        grid = arrays['image_grid_thw'].tolist(); visual = int(np.sum(arrays['input_ids'] == cfg['image_token_id']))
        assert grid == [[1, 48, 96]] and visual == 1152
        assert arrays['input_ids'].shape == arrays['attention_mask'].shape and arrays['input_ids'].shape[-1] == call['prompt_tokens']
        eos = cfg.get('eos_token_id') or cfg.get('text_config', {}).get('eos_token_id') or []
        eos = [eos] if type(eos) is int else eos
        rows.append(dict(ID=sid, arm=arm, image_grid_thw=grid, image_tokens=visual, input_tokens=call['prompt_tokens'], nonimage_tokens=call['prompt_tokens'] - visual,
                         generation_tokens=call['generation_tokens'], at_token_limit=call['generation_tokens'] == 40, last_token_is_configured_eos=call['token_trace'][-1]['token'] in eos, processor_hashes_exact=True))
    assert str(mx.default_device()) == 'Device(cpu, 0)' and not attempts
    return rows


def audit_scores(data, records):
    evaluated = read(HERE / 'evaluation.json')
    assert evaluated['status'] == 'complete' and evaluated['actual_model_calls'] == 16 and evaluated['official_S2'] is None
    expected_rows = []
    for ref in data['references']['cases']:
        sid = ref['ID']; trace = data['traces'][sid]; src = data['sources'][sid]['source']
        control, gate_result = [records[(sid, arm)] for arm in ('control', 'gate')]
        history = read(ROOT / data['sources'][sid]['baseline_folder'] / 'calls.json')[2]
        paths = [ROOT / r['path'] for r in src['images']]; numbers = [v2._frame_number(p) for p in paths]
        value = v2._json_object(control['raw'])
        selected = v2._choice(value, 'entry_frame', paths, [numbers.index(f) for f in trace['entry_candidates']], 0)
        assert type(value.get('entry_frame')) is int and value['entry_frame'] in trace['entry_candidates']
        base = {**trace['baseline'], 'entry_frame': numbers[selected]}
        assert base == trace['baseline']
        state = strict_state(gate_result['raw'])
        candidate = {**base, 'entry_frame': numbers[0] if state == 'INSIDE' else base['entry_frame']}
        row = dict(ID=sid, reference=ref['entry'], eligible=ref['eligible'], baseline=base, candidate=candidate, state=state, state_raw=gate_result['raw'],
                   control_raw_matches_history=control['raw'].strip() == history['text'].strip(), control_entry_matches_history=base['entry_frame'] == trace['final_entry'],
                   control_processor_exact=control['call']['processor_input_sha256'] == history['processor_input_sha256'],
                   other_three_unchanged=all(candidate[k] == base[k] == trace['baseline'][k] for k in ('collision_frame', 'entry_side', 'evasion_space')),
                   entry_changed=base['entry_frame'] != candidate['entry_frame'])
        assert ref['eligible']  # This experiment froze eight known intervals; no retrospective exclusion.
        times = trace['original_times']; lo, hi = [times[str(ref['entry'][k])] for k in ('lower_frame', 'upper_frame')]
        p, q = times[str(base['entry_frame'])], times[str(candidate['entry_frame'])]
        acc, mae = deltas(p, q, lo, hi)
        row['timing'] = dict(baseline=grade(p, lo, hi), candidate=grade(q, lo, hi), accuracy_delta=acc, mae_delta_seconds=mae, reference_seconds=[lo, hi])
        row['new_false_first'] = ref['entry']['status'] == 'during_clip' and candidate['entry_frame'] == numbers[0] and base['entry_frame'] != numbers[0]
        expected_rows.append(row)
    n = len(expected_rows)
    assert n == 8
    def aggregate(rows, arm):
        count = {key: sum(r['timing'][arm]['result'] == key for r in rows) for key in ('correct', 'wrong', 'indeterminate')}
        size = len(rows)
        return dict(n=size, **count, accuracy_bounds=[count['correct'] / size, (count['correct'] + count['indeterminate']) / size] if size else None,
                    mae_bounds_seconds=[sum(r['timing'][arm][key] for r in rows) / size for key in ('minimum_error_seconds', 'maximum_error_seconds')] if size else None)
    metrics = {arm: aggregate(expected_rows, arm) for arm in ('baseline', 'candidate')}
    paired = {key: [sum(r['timing'][key][i] for r in expected_rows) / n for i in (0, 1)] for key in ('accuracy_delta', 'mae_delta_seconds')}
    acceptance = dict(gain_ids=[r['ID'] for r in expected_rows if r['timing']['baseline']['result'] == 'wrong' and r['timing']['candidate']['result'] == 'correct'],
                      definite_loss_ids=[r['ID'] for r in expected_rows if r['timing']['baseline']['result'] == 'correct' and r['timing']['candidate']['result'] == 'wrong'],
                      possible_loss_ids=[r['ID'] for r in expected_rows if r['timing']['accuracy_delta'][0] < 0],
                      new_false_first_ids=[r['ID'] for r in expected_rows if r['new_false_first']],
                      unsupported_identity_commit_ids=[r['ID'] for r in expected_rows if r['ID'] == data['protocol']['preflight_identity_limitation']['ID'] and r['state'] == 'INSIDE'],
                      controls_reproduced=all(r['control_raw_matches_history'] and r['control_entry_matches_history'] for r in expected_rows),
                      other_three_preserved=all(r['other_three_unchanged'] for r in expected_rows),
                      both_strata={r['reference']['status'] for r in expected_rows} == {'before_start', 'during_clip'})
    acceptance['pass'] = bool(acceptance['gain_ids']) and not any(acceptance[k] for k in ('definite_loss_ids', 'possible_loss_ids', 'new_false_first_ids', 'unsupported_identity_commit_ids')) and acceptance['controls_reproduced'] and acceptance['other_three_preserved'] and acceptance['both_strata'] and paired['mae_delta_seconds'][1] <= 0
    strata = {s: {arm: aggregate([r for r in expected_rows if r['reference']['status'] == s], arm) for arm in ('baseline', 'candidate')} for s in ('before_start', 'during_clip')}
    def equal(actual, expected, location='root'):
        if isinstance(expected, dict):
            assert isinstance(actual, dict) and set(actual) == set(expected), location
            for key in expected:
                equal(actual[key], expected[key], location + '.' + key)
        elif isinstance(expected, list):
            assert isinstance(actual, list) and len(actual) == len(expected), location
            for i, (a, b) in enumerate(zip(actual, expected)):
                equal(a, b, location + f'[{i}]')
        elif type(expected) is float:
            assert type(actual) in (int, float) and math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-12), location
        else:
            assert actual == expected and type(actual) is type(expected), location
    for key, expected in [('rows', expected_rows), ('metrics', metrics), ('paired_delta_bounds', paired), ('gate', acceptance), ('strata', strata)]:
        equal(evaluated[key], expected, key)
    return dict(metrics=metrics, paired_delta_bounds=paired, gate=acceptance, strata=strata,
                states={str(state): sum(r['state'] == state for r in expected_rows) for state in ('INSIDE', 'OUTSIDE', 'UNCERTAIN', None)},
                all_rows_mathematically_recomputed=True, references_fixed_denominator=8, unsupported_identity_rule_registered_before_predictions=True)


def main():
    out = [HERE / ('independent_verification' + suffix) for suffix in ('.json', '.md')]
    assert not any(p.exists() for p in out)
    data = audit_inputs(replay=True)
    frozen, records, hashes, resources = audit_run(data)
    scores = audit_scores(data, records)
    tensors = audit_processors(data, records)
    for p in [Path(__file__), HERE / 'freeze.json', HERE / 'evaluation.json', HERE / 'run/report.json']:
        hashes[str(p.relative_to(ROOT))] = sha(p)
    for name, digest in {**frozen['files'], **hashes}.items():
        assert sha(ROOT / name) == digest, name
    summary = dict(status='PASS',created_utc=datetime.now(timezone.utc).isoformat(),frozen_files_verified_pre_post=len(frozen['files']),previous_freeze_preserved=len(read(PREVIOUS / 'freeze.json')['files']),
                   actual_model_calls_verified=len(records), cached_calls_replayed=data['cached_calls_replayed'], verifier_model_loads=0, verifier_model_calls=0, verifier_device='Device(cpu, 0)', verifier_network_attempts=0,
                   decomposition_verified=data['decomposition'], scores=scores, processor_metadata=tensors, resources=resources,
                   resource_summary=dict(parent_worker_wall_sum_seconds=sum(r['parent_wall_seconds'] for r in resources),worker_wall_sum_seconds=sum(r['worker_wall_seconds'] for r in resources),max_mlx_peak_GB=max(r['mlx_peak_GB'] for r in resources),max_rss_bytes=max(r['rss_bytes'] for r in resources)),
                   artifact_sha256=hashes, scope='Eight exposed cases: fresh original Q3 plus one automatic unmarked start gate per case. Other three outputs are cached, not freshly inferred. No CUDA/official score claim.')
    out[0].write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n')
    out[1].write_text('# 독립 실행·점수 검증\n\n**PASS.** 동결 파일, 기존 24개·96응답 CPU 정책 재생, 신규 16 worker 및 CPU processor 텐서, 엄격 gate·시점 평가를 대조했다.\n\n' + summary['scope'] + '\n\n상세 수치와 원시 파일 해시는 independent_verification.json에 보존했다.\n')
    print(json.dumps({k: summary[k] for k in ('status', 'frozen_files_verified_pre_post', 'actual_model_calls_verified', 'cached_calls_replayed', 'resource_summary', 'scores', 'processor_metadata')}, ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--inputs-only', action='store_true'); args = parser.parse_args()
    if args.inputs_only:
        data = audit_inputs()
        print(json.dumps(dict(status='INPUT_AUDIT_PASS', jobs=len(data['jobs']), decomposition=data['decomposition'], new_model_calls=0)))
    else:
        main()
