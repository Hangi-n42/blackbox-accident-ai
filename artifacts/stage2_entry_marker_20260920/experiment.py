"""Fixed five-case Q3 target-marking diagnostic. Reuses the original sheet, parser and Mac backend."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OLD = ROOT / 'artifacts/stage2_goal_20260920'
sys.path.insert(0, str(ROOT / 'scripts/mac'))
sys.path.insert(0, str(ROOT / 'artifacts/submissions/verify_v6/model/stage2/code'))
sys.path.insert(0, str(OLD))
from run_stage2 import ENV
from evaluate_confirmation import grade, aggregate, paired_accuracy_delta


def read(path):
    return json.loads(path.read_text())


def write(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def bounded(image):
    import math
    from PIL import Image
    scale = min(1.0, math.sqrt(1_200_000 / (image.width * image.height)))
    size = tuple(max(32, int(v * scale) // 32 * 32) for v in image.size)
    return image.convert('RGB').resize(size, Image.Resampling.BICUBIC)


def rgb_sha(image):
    return hashlib.sha256(image.tobytes()).hexdigest()


def prepare():
    from PIL import Image, ImageDraw, ImageOps
    import numpy as np
    from solution import stage2_v2 as v2
    out = HERE / 'inputs'
    out.mkdir(exist_ok=False)
    marks = {}
    for who in ['a', 'b']:
        review = read(HERE / f'reviewer_{who}_marks.json')
        assert review['model_predictions_seen'] is False
        for case in review['cases']:
            assert case['ID'] not in marks
            marks[case['ID']] = case
    sources = {x['ID']: x for x in read(OLD / 'ccd_intake/inputs.json')}
    runs = {x['ID']: x for x in read(OLD / 'mac_run/report.json')['videos']}
    cases = []
    for task in read(HERE / 'annotation_tasks.json'):
        sid = task['ID']
        source, record = sources[sid], runs[sid]
        paths = [ROOT / x['path'] for x in source['images']]
        nums = [v2._frame_number(p) for p in paths]
        candidates = record['diagnostics']['entry_candidates']
        indices = [nums.index(n) for n in candidates]
        for item in source['images']:
            assert sha(ROOT / item['path']) == item['sha256']
        base = v2._sheet(paths, indices, columns=4)
        marked = base.copy()
        mask = Image.new('1', base.size)
        drawer, mask_drawer = ImageDraw.Draw(marked), ImageDraw.Draw(mask)
        annotations = {x['frame']: x for x in marks[sid]['frames']}
        assert set(annotations) == set(candidates)
        boxes = []
        for slot, number in enumerate(candidates):
            item = annotations[number]
            path = paths[nums.index(number)]
            with Image.open(path) as im:
                w, h = im.size
                iw, ih = ImageOps.contain(im, (384, 228)).size
            assert (item['width'], item['height']) == (w, h)
            box = item['box_xyxy']
            if box is None:
                assert item['status'] in ['occluded', 'absent', 'uncertain']
                boxes.append(dict(frame=number, box=None, status=item['status']))
                continue
            assert item['status'] == 'visible'
            x0, y0, x1, y1 = box
            assert all(type(v) is int for v in box)
            assert 0 <= x0 < x1 < w and 0 <= y0 < y1 < h
            ox, oy = (slot % 4) * 384 + (384 - iw) // 2, (slot // 4) * 256 + 28
            rendered = [ox + round(x0 * iw / w), oy + round(y0 * ih / h),
                        ox + round(x1 * iw / w), oy + round(y1 * ih / h)]
            assert rendered[2] - rendered[0] >= 1 and rendered[3] - rendered[1] >= 1
            drawer.rectangle(rendered, outline=(255, 255, 0), width=2)
            mask_drawer.rectangle(rendered, outline=1, width=2)
            boxes.append(dict(frame=number, box=rendered, source_box=box, status='visible'))
        diff = np.any(np.asarray(base) != np.asarray(marked), axis=2)
        allowed = np.asarray(mask, dtype=bool)
        assert diff.any() and not np.any(diff & ~allowed)
        assert np.all(np.asarray(marked)[diff] == [255, 255, 0])
        call = record['calls'][2]
        assert call['max_new_tokens'] == 40
        assert list(bounded(base).size) == call['image_sizes'][0]
        assert rgb_sha(bounded(base)) == call['image_sha256'][0]
        for arm, image in [('baseline', base), ('marked', marked)]:
            target = out / f'{sid}_{arm}.png'
            image.save(target)
            bounded(image).save(out / f'{sid}_{arm}_model_input.png')
            job = dict(ID=sid, arm=arm, sheet=str(target), sheet_sha256=sha(target),
                       paths=[str(p) for p in paths], candidates=candidates,
                       prompt=call['prompt'], max_new_tokens=call['max_new_tokens'])
            write(out / f'{sid}_{arm}.job.json', job)
        cases.append(dict(ID=sid, candidates=candidates, boxes=boxes, canvas=list(base.size),
                          bounded_size=list(bounded(base).size), baseline_bounded_rgb_sha256=rgb_sha(bounded(base)),
                          marked_bounded_rgb_sha256=rgb_sha(bounded(marked)),
                          changed_pixels=int(diff.sum()), allowed_pixels=int(allowed.sum()),
                          changed_fraction=float(diff.mean()), old_Q3_image_exact=True))
    write(HERE / 'input_checks.json', dict(status='PASS_pending_visual_peer_review', cases=cases))


def freeze():
    assert not (HERE / 'freeze.json').exists() and not (HERE / 'run').exists()
    old = read(OLD / 'freeze.json')['files']
    for name, digest in old.items():
        assert sha(ROOT / name) == digest, name
    files = dict(old)
    for who in ['a', 'b']:
        approval = read(HERE / f'peer_review_{who}.json')
        assert approval['status'] == 'PASS' and approval['model_predictions_seen'] is False
    for path in HERE.rglob('*'):
        if path.is_file() and path.suffix in ['.py', '.json', '.png', '.md'] and path.name not in ['freeze.json', 'STATUS.json']:
            files[str(path.relative_to(ROOT))] = sha(path)
    # Source references used by evaluation/replay are bound explicitly too.
    for name in ['mac_run/report.json', 'ccd_intake/inputs.json', 'ccd_adjudication/records.json', 'evaluate_confirmation.py']:
        path = OLD / name
        files[str(path.relative_to(ROOT))] = sha(path)
    for sid in read(HERE / 'protocol.json')['cases']:
        path = OLD / 'mac_run' / sid / 'motion.npz'
        files[str(path.relative_to(ROOT))] = sha(path)
    for name, digest in files.items():
        assert sha(ROOT / name) == digest, name
    write(HERE / 'freeze.json', dict(status='locked_before_marker_predictions',
          created_utc=datetime.now(timezone.utc).isoformat(), files=files))


def worker(job_path, out):
    import socket
    import resource
    import numpy as np
    import torch
    from PIL import Image
    from solution import stage2_v2 as v2
    from mlx_stage2 import MLXVLM
    job = read(job_path)
    assert set(job) == {'ID', 'arm', 'sheet', 'sheet_sha256', 'paths', 'candidates', 'prompt', 'max_new_tokens'}
    assert sha(Path(job['sheet'])) == job['sheet_sha256']
    out.mkdir(exist_ok=False)
    start = time.perf_counter()
    report = dict(status='running', ID=job['ID'], arm=job['arm'], network_attempts=0,
                  started_utc=datetime.now(timezone.utc).isoformat())
    def deny(*args, **kwargs):
        report['network_attempts'] += 1
        raise RuntimeError('Offline worker attempted Python socket connection')
    socket.socket.connect = socket.socket.connect_ex = socket.create_connection = deny
    vlm = None
    try:
        torch.set_num_threads(2)
        v2._motion_scan.__globals__['cv2'].setNumThreads(2)
        torch.manual_seed(0)
        np.random.seed(0)
        vlm = MLXVLM(ROOT / 'artifacts/mac_experiments/stage2_mlx/model', out)
        with Image.open(job['sheet']) as im:
            raw = vlm.ask([im.convert('RGB')], job['prompt'], max_new_tokens=job['max_new_tokens'])
        parsed = v2._json_object(raw)
        paths = [Path(p) for p in job['paths']]
        numbers = [v2._frame_number(p) for p in paths]
        offered = [numbers.index(n) for n in job['candidates']]
        chosen = v2._choice(parsed, 'entry_frame', paths, offered, 0)
        raw_value = v2._integer(parsed.get('entry_frame'))
        write(out / 'result.json', dict(ID=job['ID'], arm=job['arm'], raw=raw, parsed=parsed,
              raw_integer=raw_value, valid_offered_integer=raw_value in job['candidates'],
              entry_frame=numbers[chosen], parser_changed=(raw_value != numbers[chosen]),
              call=vlm.calls[0], model_load_seconds=vlm.load_seconds))
        assert len(vlm.calls) == 1 and report['network_attempts'] == 0
        report['status'] = 'complete'
    finally:
        report.update(model_calls=0 if vlm is None else len(vlm.calls), wall_seconds=time.perf_counter()-start,
                      rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                      ended_utc=datetime.now(timezone.utc).isoformat())
        write(out / 'worker_report.json', report)


def run():
    assert platform.system() == 'Darwin' and platform.machine() == 'arm64'
    frozen = read(HERE / 'freeze.json')
    for name, digest in frozen['files'].items():
        assert sha(ROOT / name) == digest, name
    out = HERE / 'run'
    out.mkdir(exist_ok=False)
    env = dict(os.environ, **ENV, BLACKBOX_DEEPSTACK_FIX='1', BLACKBOX_COMPUTE_DTYPE='native', BLACKBOX_DECODE_MODE='sync')
    report = dict(status='running', started_utc=datetime.now(timezone.utc).isoformat(),
                  platform=platform.platform(), freeze_sha256=sha(HERE / 'freeze.json'), workers=[])
    write(out / 'report.json', report)
    for sid in read(HERE / 'protocol.json')['cases']:
        for arm in ['baseline', 'marked']:
            name = f'{sid}_{arm}'
            start = time.perf_counter()
            with (out / f'{name}.log').open('x') as log:
                result = subprocess.run([sys.executable, str(Path(__file__)), 'worker', '--job',
                    str(HERE / 'inputs' / f'{name}.job.json'), '--out', str(out / name)],
                    env=env, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
            report['workers'].append(dict(ID=sid, arm=arm, exit_status=result.returncode, wall_seconds=time.perf_counter()-start))
            write(out / 'report.json', report)
            print(json.dumps(report['workers'][-1]), flush=True)
            if result.returncode:
                report['status'] = 'failed'
                write(out / 'report.json', report)
                raise SystemExit(result.returncode)
    report.update(status='complete', ended_utc=datetime.now(timezone.utc).isoformat())
    write(out / 'report.json', report)


def evaluate():
    import numpy as np
    from solution import stage2_uncapped_jerk_v6c as reference
    sources = {x['ID']: x for x in read(OLD / 'ccd_intake/inputs.json')}
    labels = {x['ID']: x for x in read(OLD / 'ccd_adjudication/records.json')['records']}
    old_runs = {x['ID']: x for x in read(OLD / 'mac_run/report.json')['videos']}
    run_report = read(HERE / 'run/report.json')
    ids = read(HERE / 'protocol.json')['cases']
    assert run_report['status'] == 'complete'
    assert len(run_report['workers']) == 2 * len(ids)
    assert {(w['ID'], w['arm']) for w in run_report['workers']} == {(sid, arm) for sid in ids for arm in ['baseline', 'marked']}
    assert all(w['exit_status'] == 0 for w in run_report['workers'])
    for name, digest in read(HERE / 'freeze.json')['files'].items():
        assert sha(ROOT / name) == digest, name
    actual_calls = cached_replays = 0
    unchanged = set()
    rows = []
    for sid in ids:
        source, label, old = sources[sid], labels[sid]['entry'], old_runs[sid]
        paths = [ROOT / x['path'] for x in source['images']]
        times = {x['frame']: x['pts_seconds'] for x in source['images']}
        lower, upper = times[label['lower_frame']], times[label['upper_frame']]
        row = dict(ID=sid, interval_seconds=[lower, upper], already_inside=(lower == upper == 0), arms={})
        motion = np.load(OLD / 'mac_run' / sid / 'motion.npz')
        for arm in ['baseline', 'marked']:
            destination = HERE / 'run' / f'{sid}_{arm}'
            wr = read(destination / 'worker_report.json')
            calls = read(destination / 'calls.json')
            assert wr['status'] == 'complete' and wr['model_calls'] == len(calls) == 1
            assert wr['network_attempts'] == 0
            actual_calls += len(calls)
            result = read(destination / 'result.json')
            assert result['call'] == calls[0]
            job = read(HERE / 'inputs' / f'{sid}_{arm}.job.json')
            from PIL import Image
            with Image.open(job['sheet']) as image:
                assert result['call']['image_sha256'] == [rgb_sha(bounded(image))]
                assert result['call']['image_sizes'] == [list(bounded(image).size)]
            assert result['call']['prompt'] == job['prompt'] and result['call']['max_new_tokens'] == job['max_new_tokens']
            class Replay:
                count = 0
                def ask(self, images, prompt, max_new_tokens):
                    i = self.count
                    self.count += 1
                    historical = old['calls'][i]
                    assert prompt == historical['prompt'] and max_new_tokens == historical['max_new_tokens']
                    assert [rgb_sha(bounded(im)) for im in images] == historical['image_sha256']
                    return result['raw'] if i == 2 else historical['text']
            replay = Replay()
            prediction, diagnostic = reference._predict_file(paths, motion['base_scores'], motion['new_scores'], replay)
            assert replay.count == 4 and prediction['entry_frame'] == result['entry_frame']
            cached_replays += replay.count - 1
            assert all(prediction[k] == old['baseline_prediction'][k] for k in ['collision_frame', 'entry_side', 'evasion_space'])
            unchanged.add((sid, arm))
            scored = grade(times[result['entry_frame']], lower, upper)
            row['arms'][arm] = dict(prediction=prediction, raw=result['raw'], valid_offered_integer=result['valid_offered_integer'],
                                   parser_changed=result['parser_changed'], time_seconds=times[result['entry_frame']], **scored)
            if arm == 'baseline':
                assert result['call']['processor_input_sha256'] == old['calls'][2]['processor_input_sha256']
                row['historical_baseline_raw_match'] = result['raw'].strip() == old['calls'][2]['text'].strip()
                row['historical_baseline_prediction_match'] = prediction == old['baseline_prediction']
        a, b = (row['arms'][x] for x in ['baseline', 'marked'])
        row['definite_gain'] = a['result'] == 'wrong' and b['result'] == 'correct' and b['valid_offered_integer'] and not b['parser_changed']
        row['definite_loss'] = a['result'] == 'correct' and b['result'] == 'wrong'
        row['paired_accuracy_delta_bounds'] = paired_accuracy_delta(a['time_seconds'], b['time_seconds'], lower, upper)
        cuts = [lower, upper] + [p for p in [a['time_seconds'], b['time_seconds']] if lower <= p <= upper]
        delta = [abs(b['time_seconds']-t)-abs(a['time_seconds']-t) for t in cuts]
        row['paired_absolute_error_delta_bounds_seconds'] = [min(delta), max(delta)]
        row['false_first_on_late_entry'] = not row['already_inside'] and b['prediction']['entry_frame'] == source['images'][0]['frame'] and a['prediction']['entry_frame'] != source['images'][0]['frame']
        rows.append(row)
    n = len(rows)
    mae_delta = [sum(r['paired_absolute_error_delta_bounds_seconds'][i] for r in rows)/n for i in [0, 1]]
    gate = dict(gains=sum(r['definite_gain'] for r in rows), losses=sum(r['definite_loss'] for r in rows),
                no_new_false_first=not any(r['false_first_on_late_entry'] for r in rows), paired_MAE_nonincrease=mae_delta[1] <= 1e-9,
                baseline_matches_history=all(r['historical_baseline_prediction_match'] for r in rows))
    gate['pass'] = gate['gains'] >= 1 and gate['losses'] == 0 and gate['no_new_false_first'] and gate['paired_MAE_nonincrease'] and gate['baseline_matches_history']
    write(HERE / 'evaluation.json', dict(status='complete', official_S2=None, automatic_policy=False,
          actual_model_calls=actual_calls, cached_answer_replays=cached_replays, rows=rows,
          metrics={a: aggregate([r['arms'][a] for r in rows]) for a in ['baseline', 'marked']},
          paired_accuracy_delta_bounds=[sum(r['paired_accuracy_delta_bounds'][i] for r in rows)/n for i in [0,1]],
          paired_MAE_delta_bounds_seconds=mae_delta, gate=gate, other_three_outputs_unchanged=sum(all((sid, arm) in unchanged for arm in ['baseline', 'marked']) for sid in ids)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['prepare', 'freeze', 'run', 'worker', 'evaluate'])
    parser.add_argument('--job', type=Path)
    parser.add_argument('--out', type=Path)
    args = parser.parse_args()
    if args.action == 'worker':
        worker(args.job, args.out)
    else:
        globals()[args.action]()
