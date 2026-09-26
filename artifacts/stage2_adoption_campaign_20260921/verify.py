"""Independent saved-call/CPU-processor audit; never loads model weights or generates."""
from collections import Counter
from datetime import datetime, timezone
from fractions import Fraction as F
import hashlib
import json
import math
import os
from pathlib import Path
import sys

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
TAU = F(3, 10)
read = lambda p: json.loads(p.read_text())


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def rgb(image):
    return hashlib.sha256(image.convert('RGB').tobytes()).hexdigest()


def bounded(images):
    from PIL import Image
    limit = max(1024, 1200000 // len(images))
    result = []
    for image in images:
        scale = min(1., math.sqrt(limit / (image.width * image.height)))
        size = tuple(max(32, int(v * scale) // 32 * 32) for v in image.size)
        result.append(image.convert('RGB').resize(size, Image.Resampling.BICUBIC))
    return result


def grade(prediction, lo, hi):
    best = max(lo - prediction, prediction - hi, F(0))
    worst = max(abs(prediction - lo), abs(prediction - hi))
    return dict(accuracy_lower=int(worst <= TAU), accuracy_upper=int(best <= TAU),
                error_seconds_lower=float(best), error_seconds_upper=float(worst))


def paired(old, new, lo, hi):
    edges = sorted({lo, hi} | {t for p in (old, new) for t in (p-TAU, p+TAU) if lo <= t <= hi})
    probes = edges + [(a+b)/2 for a, b in zip(edges, edges[1:])]
    accuracy = [int(abs(new-t) <= TAU)-int(abs(old-t) <= TAU) for t in probes]
    errors = [abs(new-t)-abs(old-t) for t in {lo, hi, *[p for p in (old, new) if lo <= p <= hi]}]
    return [min(accuracy), max(accuracy)], [min(errors), max(errors)]


def coverage(predictions, lo, hi):
    edges = sorted({lo, hi} | {p+d for p in predictions for d in (-TAU, TAU) if lo <= p+d <= hi})
    probes = edges + [(a+b)/2 for a, b in zip(edges, edges[1:])]
    return dict(any_possible_candidate=any(max(lo-p, p-hi, F(0)) <= TAU for p in predictions),
                all_reference_times_covered=all(any(abs(p-t) <= TAU for p in predictions) for t in probes),
                one_candidate_correct_for_all_times=any(max(abs(p-lo), abs(p-hi)) <= TAU for p in predictions))


def same(actual, expected):
    if isinstance(expected, dict):
        for key, value in expected.items():
            assert key in actual, key
            same(actual[key], value)
    elif isinstance(expected, list):
        assert len(actual) == len(expected)
        for a, e in zip(actual, expected):
            same(a, e)
    elif isinstance(expected, float):
        assert math.isfinite(actual) and abs(actual-expected) <= 1e-10, (actual, expected)
    else:
        assert actual == expected, (actual, expected)


def verify():
    import numpy as np
    from PIL import Image
    sys.path.insert(0, str(ROOT/'artifacts/submissions/verify_v6/model/stage2/code'))
    from solution import stage2_v2 as v2
    from solution import stage2_uncapped_jerk_v6c as policy
    frozen = read(HERE/'freeze.json')['files']
    for path, digest in frozen.items():
        assert sha(ROOT/path) == digest, path
    model_sources = read(HERE/'model_source_manifest.json')['files']
    assert len(model_sources) == 45
    for path, digest in model_sources.items():
        assert sha(ROOT/path) == digest, path
    sources = read(HERE/'intake/inputs.json')
    refs = read(HERE/'references.json')
    reviews = {who: read(HERE/f'review_{who}.json') for who in ('a', 'b')}
    source_map = {s['ID']: s for s in sources}
    ref_map = {r['ID']: r for r in refs['cases']}
    assert len(sources) == len(source_map) == len(ref_map) == 12
    assert set(source_map) == set(ref_map)
    for who, review in reviews.items():
        assert review['human_review'] is False and review['model_predictions_seen'] is False and review['peer_review_seen'] is False
        assert len(review['cases']) == 12
        for row in review['cases']:
            assert row == ref_map[row['ID']][f'review_{who}']
    for ref in refs['cases']:
        a, b = ref['review_a'], ref['review_b']
        if ref['eligible']:
            assert a['eligible'] and b['eligible'] and a['same_counterpart_verified'] and b['same_counterpart_verified']
            assert a['entry']['status'] == b['entry']['status'] == ref['entry']['status']
            assert ref['entry']['lower_frame'] == min(a['entry']['lower_frame'], b['entry']['lower_frame'])
            assert ref['entry']['upper_frame'] == max(a['entry']['upper_frame'], b['entry']['upper_frame'])
        assert ref['eligible'] == (ref['primary'] or ref['secondary'])
        assert not (ref['primary'] and ref['secondary'])
        if ref['primary']:
            assert a['strict_boundary_evidence'] and b['strict_boundary_evidence']
    overlap = read(HERE/'intake/overlap.json')
    assert overlap['status'] == 'PASS_WITH_SCOPE_LIMITS' and not overlap['whole_video_sha256_duplicates'] and not overlap['sampled_exact_rgb_matches']
    assert len({s['source_group'] for s in sources}) == 12
    assert not {s['source_group'] for s in sources} & set(read(HERE/'intake/selection.json')['excluded_source_groups'])
    baseline_execution = read(HERE/'baseline/execution.json')
    paired_execution = read(HERE/'paired/execution.json')
    assert baseline_execution['status'] == paired_execution['status'] == 'complete'
    expected_jobs = [(s['ID'], 'baseline') for s in sources]
    expected_pairs = [(s['ID'], a) for s in sources for a in ('control', 'candidate')]
    for report, expected in ((baseline_execution, expected_jobs), (paired_execution, expected_pairs)):
        assert [(r['ID'], r['arm']) for r in report['workers']] == expected
        assert all(r['exit_code'] == 0 for r in report['workers'])
    evaluation = read(HERE/'evaluation.json')
    assert evaluation['status'] == 'paired_complete' and evaluation['official_stage2_score'] is None
    evaluated = {r['ID']: r for r in evaluation['cases']}
    assert set(evaluated) == set(source_map)
    jobs, records, summary_rows, output_bindings = [], [], [], {}
    for src in sources:
        sid = src['ID']; ref = ref_map[sid]; folder = HERE/'baseline'/sid
        assert src == read(HERE/'intake'/sid/'input_manifest.json')
        native = read(ROOT/src['pts_source'])['mapping']
        times = {r['frame_id']: F(r['native_pts'])*F(r['time_base']) for r in native}
        paths = [ROOT/r['path'] for r in src['images']]
        numbers = [v2._frame_number(p) for p in paths]
        assert numbers == sorted(times) == list(range(50))
        assert sha(ROOT/src['source_video']) == src['source_sha256'] and sha(ROOT/src['pts_source']) == src['pts_sha256']
        for p, item in zip(paths, src['images']):
            assert sha(p) == item['sha256'] and times[item['frame']] == F(str(item['pts_seconds']))
        base, calls = read(folder/'result.json'), read(folder/'calls.json')
        report = read(folder/'worker_report.json')
        assert base['ID'] == report['ID'] == sid and report['status'] == 'complete'
        assert report['model_calls'] == len(calls) == 4 and report['network_attempts'] == 0
        assert calls == base['calls']
        class Replay:
            count = 0
            def ask(self, images, prompt, max_new_tokens):
                call = calls[self.count]; self.count += 1
                assert prompt == call['prompt'] and max_new_tokens == call['max_new_tokens']
                ims = bounded(images)
                assert [rgb(im) for im in ims] == call['image_sha256']
                assert [list(im.size) for im in ims] == call['image_sizes']
                return call['text']
        replay = Replay()
        with np.load(folder/'motion.npz', allow_pickle=False) as motion:
            assert motion['base_scores'].dtype == motion['new_scores'].dtype == np.dtype('float32')
            prediction, diag = policy._predict_file(paths, motion['base_scores'], motion['new_scores'], replay)
        assert replay.count == 4 and prediction == base['baseline_prediction']
        frames = diag['entry_candidates']; assert frames == base['diagnostics']['entry_candidates']
        assert 1 <= len(frames) <= 12 and frames == sorted(set(frames))
        indices = [numbers.index(f) for f in frames]
        anchor = v2._choice(diag['collision'], 'collision_frame', paths, [numbers.index(f) for f in diag['collision_candidates']], numbers.index(diag['motion_proposal']))
        assert indices == v2._uniform_indices(0, anchor, 12)
        sheet = v2._sheet(paths, indices, columns=4)
        source_prompt = calls[2]['prompt']; phrase = 'These frames are chronological, left to right then top to bottom. '
        assert source_prompt.count(phrase) == 1
        arm_results = {}
        for arm in ('control', 'candidate'):
            out = HERE/'run'/f'{sid}_{arm}'
            result, fresh_calls = read(out/'result.json'), read(out/'calls.json')
            assert result['ID'] == sid and result['arm'] == arm and len(fresh_calls) == 1 and result['calls'] == fresh_calls
            call = fresh_calls[0]
            assert result['raw'].strip() == call['text'].strip()
            parsed = v2._json_object(result['raw'])
            assert parsed == result['parsed'] and result['candidate_frames'] == frames
            chosen = v2._choice(parsed, 'entry_frame', paths, indices, 0)
            expected = dict(prediction, entry_frame=numbers[chosen])
            assert result['prediction'] == expected
            assert result['valid_raw_choice'] == (type(parsed.get('entry_frame')) is int and parsed.get('entry_frame') in frames)
            assert call['max_new_tokens'] == 40 and call['compute_dtype'] == 'native' and call['decode_mode'] == 'sync' and call['deepstack_fix']
            images, prompt = [sheet], source_prompt
            if arm == 'candidate':
                cols = min(4, len(frames))
                images = [sheet.crop(((i%cols)*384, (i//cols)*256, (i%cols+1)*384, (i//cols+1)*256)) for i in range(len(frames))]
                prompt = source_prompt.replace(phrase, 'These images are chronological in the order provided. ', 1)
            assert call['prompt'] == prompt
            ims = bounded(images)
            assert [rgb(im) for im in ims] == call['image_sha256'] and [list(im.size) for im in ims] == call['image_sizes']
            assert all(im.size == original.size and rgb(im) == rgb(original) for im, original in zip(ims, images))
            if arm == 'control':
                assert expected == prediction
                for key in ('text', 'token_trace', 'processor_input_sha256', 'image_sizes', 'image_sha256', 'prompt_sha256'):
                    assert call[key] == calls[2][key], (sid, key)
            jobs.append((sid, arm, ims, prompt, call)); arm_results[arm] = result
            records.append(dict(ID=sid, arm=arm, frames=frames, raw=result['raw'], valid_raw_choice=result['valid_raw_choice'], prediction=expected,
                                wall_seconds=result['wall_seconds'], model_load_seconds=result['model_load_seconds'], inference_seconds=call['seconds'],
                                peak_memory_GB=call['peak_memory'], process_peak_rss_bytes=result['process_peak_rss_bytes']))
            output_bindings[str((out/'result.json').relative_to(ROOT))] = sha(out/'result.json')
            output_bindings[str((out/'calls.json').relative_to(ROOT))] = sha(out/'calls.json')
        candidate = arm_results['candidate']['prediction']
        row = dict(ID=sid, primary=ref['primary'], secondary=ref['secondary'], entry=ref['entry'], diagnostic_only=not ref['eligible'], failure=False,
                   baseline=prediction, candidate=candidate, entry_candidates=frames, q2_anchor=frames[-1],
                   baseline_valid_raw=arm_results['control']['valid_raw_choice'], candidate_valid_raw=arm_results['candidate']['valid_raw_choice'],
                   unknown_new_first_frame=ref['entry']['status']=='unknown' and candidate['entry_frame']==0 and prediction['entry_frame']!=0,
                   unscored_new_first_frame=not ref['eligible'] and candidate['entry_frame']==0 and prediction['entry_frame']!=0,
                   other_three_equal=all(prediction[k] == candidate[k] for k in ('collision_frame', 'entry_side', 'evasion_space')), control_reproduced=True)
        if ref['entry']['status'] != 'unknown':
            lo, hi = [times[ref['entry'][k]] for k in ('lower_frame', 'upper_frame')]
            old, new = times[prediction['entry_frame']], times[candidate['entry_frame']]
            acc, mae = paired(old, new, lo, hi)
            row.update(reference_seconds=[float(lo), float(hi)], baseline_metric=grade(old, lo, hi), candidate_metric=grade(new, lo, hi),
                       coverage={name: coverage([times[f] for f in fs], lo, hi) for name, fs in [('full_input', numbers), ('q2_window', numbers[:anchor+1]), ('candidates', frames)]},
                       paired_accuracy_delta=acc, paired_mae_delta_seconds=list(map(float, mae)), definite_recovery=acc[0]==1, possible_regression=acc[0]<0,
                       new_false_first_frame=candidate['entry_frame']==0 and prediction['entry_frame']!=0 and lo>0)
        same(evaluated[sid], row); summary_rows.append(row)
        for name in ('result.json', 'calls.json', 'motion.npz', 'worker_report.json'):
            output_bindings[str((folder/name).relative_to(ROOT))] = sha(folder/name)
    groups = {}
    for name, key in (('strict', 'primary'), ('conditional', 'secondary')):
        rows = [r for r in summary_rows if r[key]]; n = len(rows)
        g = dict(n=n, IDs=[r['ID'] for r in rows], before_start=sum(r['entry']['status']=='before_start' for r in rows), during_clip=sum(r['entry']['status']=='during_clip' for r in rows))
        for arm in ('baseline', 'candidate'):
            g[arm] = {k:sum(r[f'{arm}_metric'][k] for r in rows)/n for k in ('accuracy_lower', 'accuracy_upper', 'error_seconds_lower', 'error_seconds_upper')}
        g.update(paired_accuracy_delta=[sum(r['paired_accuracy_delta'][i] for r in rows)/n for i in (0, 1)],
                 paired_mae_delta_seconds=[float(sum(F(str(r['paired_mae_delta_seconds'][i])) for r in rows)/n) for i in (0, 1)],
                 definite_recoveries=sum(r['definite_recovery'] for r in rows), possible_regressions=sum(r['possible_regression'] for r in rows), new_false_first_frames=sum(r['new_false_first_frame'] for r in rows))
        same(evaluation['groups'][name], g); groups[name] = g
    totals = dict(screened=12, baseline_completed=12, paired_completed=12, primary_scored=sum(r['primary'] for r in summary_rows), conditional_scored=sum(r['secondary'] for r in summary_rows),
                  unscored=sum(not r['primary'] and not r['secondary'] for r in summary_rows), control_reproduced=12, other_three_equal=12,
                  baseline_invalid_raw=sum(not r['baseline_valid_raw'] for r in summary_rows), candidate_invalid_raw=sum(not r['candidate_valid_raw'] for r in summary_rows))
    same(evaluation['totals'], totals)
    processor_rows = audit_processors(jobs)
    for path, digest in frozen.items():
        assert sha(ROOT/path) == digest, path
    for path, digest in model_sources.items():
        assert sha(ROOT/path) == digest, path
    for path, digest in output_bindings.items():
        assert sha(ROOT/path) == digest, path
    return dict(status='PASS', created_utc=datetime.now(timezone.utc).isoformat(), verifier_sha256=sha(Path(__file__)), freeze_sha256=sha(HERE/'freeze.json'), frozen_files_preserved=len(frozen),
                actual_model_calls=72, baseline_calls_cpu_replayed=48, fresh_q3_calls=24, processor_replays=len(processor_rows), additional_model_calls=0, model_weights_loaded=False, original_model_and_source_files_preserved=len(model_sources),
                evaluation_sha256=sha(HERE/'evaluation.json'), output_sha256=output_bindings, groups=groups, totals=totals, cases=summary_rows, processors=processor_rows, runtime_records=records,
                unscored_new_first_ids=[r['ID'] for r in summary_rows if not (r['primary'] or r['secondary']) and r['candidate']['entry_frame']==0 and r['baseline']['entry_frame']!=0],
                execution=dict(baseline_parent_wall_seconds=(datetime.fromisoformat(baseline_execution['ended_utc'])-datetime.fromisoformat(baseline_execution['started_utc'])).total_seconds(),
                               paired_parent_wall_seconds=(datetime.fromisoformat(paired_execution['ended_utc'])-datetime.fromisoformat(paired_execution['started_utc'])).total_seconds(),
                               baseline_worker_seconds=sum(r['wall_seconds'] for r in baseline_execution['workers']), paired_worker_seconds=sum(r['wall_seconds'] for r in paired_execution['workers'])),
                limits=['Dual AI references only; strict and conditional have separate fixed denominators, and nine cases remain unscored.',
                        'All eligible references concern before-start encoding. No eligible during-clip case tests false-first-frame risk.',
                        'Distinct provider compilation groups plus sampled duplicate screening do not certify original-incident independence.',
                        'Packaging changes visual boundaries/positions and text layout together; this is not a pure semantic or pixel-count causal test.',
                        'Other three outputs are copied from baseline, not newly inferred. Mac MLX validation is not submitted CUDA/NF4 or official total Stage2 performance.'])


def audit_processors(jobs):
    import numpy as np
    import socket
    os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', HF_HUB_DISABLE_IMPLICIT_TOKEN='1', TOKENIZERS_PARALLELISM='false')
    def deny(*args, **kwargs):
        raise AssertionError('Offline processor requested network')
    socket.socket.connect = socket.socket.connect_ex = socket.create_connection = deny
    import mlx.core as mx
    mx.set_default_device(mx.cpu)
    import torch
    torch.set_num_threads(2)
    from mlx_vlm.utils import load_processor, load_image_processor, prepare_inputs
    model = ROOT/'artifacts/mac_experiments/stage2_mlx/model'; cfg = read(model/'config.json')
    processor = load_processor(model, True, eos_token_ids=cfg.get('eos_token_id'), trust_remote_code=False, local_files_only=True)
    image_processor = load_image_processor(model, trust_remote_code=False, local_files_only=True)
    if image_processor is not None:
        processor.image_processor = image_processor
    rows = []
    for sid, arm, images, prompt, call in jobs:
        content = [{'type':'image'} for _ in images] + [{'type':'text', 'text':prompt}]
        text = processor.apply_chat_template([{'role':'user', 'content':content}], tokenize=False, add_generation_prompt=True)
        inputs = prepare_inputs(processor, images=images, prompts=text, image_token_index=cfg.get('image_token_index') or cfg['image_token_id'], add_special_tokens=True)
        mx.eval(*[v for v in inputs.values() if isinstance(v, mx.array)])
        arrays = {k:np.asarray(v) for k, v in inputs.items() if isinstance(v, mx.array)}
        hashes = {k:hashlib.sha256(v.tobytes()).hexdigest() for k, v in arrays.items()}
        assert hashes == call['processor_input_sha256'] and hashlib.sha256(text.encode()).hexdigest() == call['prompt_sha256']
        assert arrays['input_ids'].shape[-1] == call['prompt_tokens']
        grid = arrays['image_grid_thw'].tolist()
        expected_grid = [[1, im.height//16, im.width//16] for im in images]
        assert grid == expected_grid
        tokens = int(np.sum(arrays['input_ids'] == cfg['image_token_id']))
        assert tokens == sum(t*h*w//4 for t, h, w in grid)
        rows.append(dict(ID=sid, arm=arm, image_count=len(images), image_grid_thw=grid, image_tokens=tokens, prompt_tokens=call['prompt_tokens'],
                         nonimage_tokens=call['prompt_tokens']-tokens, generation_tokens=call['generation_tokens'], max_new_tokens=call['max_new_tokens'],
                         generation_at_cap=call['generation_tokens']>=call['max_new_tokens'], processor_hashes_exact=True))
    assert str(mx.default_device()) == 'Device(cpu, 0)'
    return rows


if __name__ == '__main__':
    out = HERE/'independent_verification.json'
    assert not out.exists()
    result = verify()
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    lines = ['# 독립 실행·평가 검증: PASS', '',
             f"동결 {result['frozen_files_preserved']}개 해시 전후 보존. baseline 48개 저장 응답의 동일 코드·질문·RGB·출력 재생, fresh Q3 24질의와 CPU processor 24회 텐서 해시 대조를 완료했다. 새 모델 호출·모델 가중치 로드0.", '',
             '|분모|건수|기존 정확도 하한/상한|후보 정확도 하한/상한|같은 정답시각 MAE 변화(초)|', '|---|---:|---|---|---|']
    for name, g in result['groups'].items():
        lines.append(f"|{name}|{g['n']}|{g['baseline']['accuracy_lower']}/{g['baseline']['accuracy_upper']}|{g['candidate']['accuracy_lower']}/{g['candidate']['accuracy_upper']}|{g['paired_mae_delta_seconds']}|")
    lines += ['', f"실제 72 model calls, 전체 36 workers exit0. 동일 control 12/12, 다른 세 출력 12/12 보존. 형식상 미유효 응답 baseline {result['totals']['baseline_invalid_raw']}, candidate {result['totals']['candidate_invalid_raw']}.", '', *result['limits']]
    (HERE/'independent_verification.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({k:result[k] for k in ('status', 'frozen_files_preserved', 'actual_model_calls', 'groups', 'totals', 'execution')}, ensure_ascii=False))
