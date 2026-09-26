"""One legend sentence on frozen marker images; conditional first-frame state probe."""
import argparse
from datetime import datetime, timezone
import importlib.util
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
MARKER = ROOT / 'artifacts/stage2_entry_marker_20260920'
spec = importlib.util.spec_from_file_location('marker_experiment', MARKER / 'experiment.py')
prior = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prior)
read, write, sha = prior.read, prior.write, prior.sha


def check_frozen():
    frozen = read(HERE / 'freeze.json')
    for name, digest in frozen['files'].items():
        assert sha(ROOT / name) == digest, name
    return frozen


def freeze():
    assert not (HERE / 'freeze.json').exists() and not (HERE / 'run').exists()
    files = dict(read(MARKER / 'freeze.json')['files'])
    for name, digest in files.items():
        assert sha(ROOT / name) == digest, name
    protocol = read(HERE / 'protocol.json')
    references = []
    for who in ['a', 'b']:
        ref = read(HERE / f'state_reference_{who}.json')
        assert ref['model_predictions_seen'] is False
        references += ref['cases']
    assert len(references) == len(protocol['cases'])
    assert {r['ID'] for r in references} == set(protocol['cases'])
    assert all(r['state'] in ['INSIDE', 'OUTSIDE', 'UNKNOWN'] for r in references)
    from PIL import Image
    for sid in protocol['cases']:
        old_job = read(MARKER / 'inputs' / f'{sid}_marked.job.json')
        control = read(HERE / 'inputs' / f'{sid}_control.job.json')
        legend = read(HERE / 'inputs' / f'{sid}_legend.job.json')
        assert {k:v for k,v in control.items() if k != 'arm'} == {k:v for k,v in old_job.items() if k != 'arm'}
        assert legend['prompt'] == protocol['legend_sentence'] + control['prompt']
        assert {k:v for k,v in control.items() if k not in ['arm','prompt']} == {k:v for k,v in legend.items() if k not in ['arm','prompt']}
        probe = read(HERE / 'inputs' / f'{sid}_state.job.json')
        assert probe['prompt'] == protocol['conditional_state_probe']['prompt'] and probe['max_new_tokens'] == 40
        with Image.open(MARKER / 'inputs' / f'{sid}_marked_model_input.png') as sheet, Image.open(probe['image']) as tile:
            assert tile.size == (384,256) and prior.rgb_sha(tile) == prior.rgb_sha(sheet.crop((0,0,384,256)))
        for name in ['result.json','calls.json','worker_report.json']:
            path = MARKER / 'run' / f'{sid}_marked' / name
            files[str(path.relative_to(ROOT))] = sha(path)
    for path in [MARKER / 'freeze.json', MARKER / 'evaluation.json', MARKER / 'run/report.json']:
        files[str(path.relative_to(ROOT))] = sha(path)
    for path in HERE.rglob('*'):
        if path.is_file() and path.name != 'STATUS.json':
            files[str(path.relative_to(ROOT))] = sha(path)
    write(HERE / 'freeze.json', dict(status='locked_before_legend_and_state_predictions',
          created_utc=datetime.now(timezone.utc).isoformat(), files=files))


def run(state=False):
    assert platform.system() == 'Darwin' and platform.machine() == 'arm64'
    check_frozen()
    if state:
        result = read(HERE / 'evaluation.json')
        assert result['status'] == 'complete' and result['gate']['pass'] is False
        assert result['gate']['control_matches_history'], 'Investigate contract failure before causal follow-up'
    out = HERE / ('state_run' if state else 'run')
    out.mkdir(exist_ok=False)
    env = dict(os.environ, **prior.ENV, BLACKBOX_DEEPSTACK_FIX='1', BLACKBOX_COMPUTE_DTYPE='native', BLACKBOX_DECODE_MODE='sync')
    report = dict(status='running', started_utc=datetime.now(timezone.utc).isoformat(), platform=platform.platform(),
                  freeze_sha256=sha(HERE / 'freeze.json'), workers=[])
    write(out / 'report.json', report)
    for sid in read(HERE / 'protocol.json')['cases']:
        for arm in (['state'] if state else ['control','legend']):
            job = HERE / 'inputs' / f'{sid}_{arm}.job.json'
            script, action = (Path(__file__), 'state_worker') if state else (MARKER / 'experiment.py', 'worker')
            started = time.perf_counter()
            with (out / f'{sid}_{arm}.log').open('x') as log:
                proc = subprocess.run([sys.executable, str(script), action, '--job', str(job), '--out', str(out / f'{sid}_{arm}')],
                                      env=env, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
            report['workers'].append(dict(ID=sid, arm=arm, exit_status=proc.returncode, wall_seconds=time.perf_counter()-started))
            if proc.returncode:
                report['status'] = 'failed'
                write(out / 'report.json', report)
                raise SystemExit(proc.returncode)
            write(out / 'report.json', report)
    report.update(status='complete', ended_utc=datetime.now(timezone.utc).isoformat())
    write(out / 'report.json', report)


def state_worker(job_path, out):
    import socket
    import resource
    import numpy as np
    import torch
    from PIL import Image
    from mlx_stage2 import MLXVLM
    job = read(job_path)
    assert set(job) == {'ID','image','sha256','prompt','max_new_tokens'}
    assert sha(Path(job['image'])) == job['sha256']
    out.mkdir(exist_ok=False)
    started = time.perf_counter()
    report = dict(status='running', ID=job['ID'], arm='state', network_attempts=0,
                  started_utc=datetime.now(timezone.utc).isoformat())
    def deny(*args, **kwargs):
        report['network_attempts'] += 1
        raise RuntimeError('Unexpected Python socket connection')
    socket.socket.connect = socket.socket.connect_ex = socket.create_connection = deny
    vlm = None
    try:
        torch.set_num_threads(2)
        torch.manual_seed(0)
        np.random.seed(0)
        vlm = MLXVLM(ROOT / 'artifacts/mac_experiments/stage2_mlx/model', out)
        with Image.open(job['image']) as image:
            raw = vlm.ask([image.convert('RGB')], job['prompt'], max_new_tokens=job['max_new_tokens'])
        try:
            obj = json.loads(raw)
        except (ValueError, TypeError):
            obj = None
        value = obj.get('lane_state') if isinstance(obj, dict) else None
        valid = isinstance(value, str) and value in ['INSIDE','OUTSIDE','UNCERTAIN']
        write(out / 'result.json', dict(ID=job['ID'], raw=raw, state=value if valid else None,
              valid_enum=valid, call=vlm.calls[0]))
        assert len(vlm.calls) == 1 and report['network_attempts'] == 0
        report['status'] = 'complete'
    finally:
        report.update(model_calls=0 if vlm is None else len(vlm.calls), wall_seconds=time.perf_counter()-started,
                      rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                      ended_utc=datetime.now(timezone.utc).isoformat())
        write(out / 'worker_report.json', report)


def results(state=False):
    from PIL import Image
    check_frozen()
    out = HERE / ('state_run' if state else 'run')
    report = read(out / 'report.json')
    ids = read(HERE / 'protocol.json')['cases']
    arms = ['state'] if state else ['control','legend']
    expected = {(sid, arm) for sid in ids for arm in arms}
    assert report['status'] == 'complete' and len(report['workers']) == len(expected)
    assert {(w['ID'],w['arm']) for w in report['workers']} == expected
    assert all(w['exit_status'] == 0 for w in report['workers'])
    records = {}
    for sid, arm in sorted(expected):
        folder = out / f'{sid}_{arm}'
        wr, calls, result = (read(folder / name) for name in ['worker_report.json','calls.json','result.json'])
        assert wr['status'] == 'complete' and wr['model_calls'] == len(calls) == 1 and wr['network_attempts'] == 0
        assert result['call'] == calls[0]
        assert result['ID'] == wr['ID'] == sid and wr['arm'] == arm
        if not state:
            assert result['arm'] == arm
        assert calls[0]['text'].strip() == result['raw'].strip()
        job = read(HERE / 'inputs' / f'{sid}_{arm}.job.json')
        with Image.open(job['image'] if state else job['sheet']) as image:
            assert calls[0]['image_sha256'] == [prior.rgb_sha(prior.bounded(image))]
            assert calls[0]['image_sizes'] == [list(prior.bounded(image).size)]
        assert calls[0]['prompt'] == job['prompt'] and calls[0]['max_new_tokens'] == job['max_new_tokens']
        records[(sid,arm)] = result
    return records


def evaluate():
    import numpy as np
    from solution import stage2_uncapped_jerk_v6c as reference
    records = results()
    sources = {r['ID']:r for r in read(prior.OLD / 'ccd_intake/inputs.json')}
    labels = {r['ID']:r['entry'] for r in read(prior.OLD / 'ccd_adjudication/records.json')['records']}
    original = {r['ID']:r for r in read(prior.OLD / 'mac_run/report.json')['videos']}
    rows, replay_count = [], 0
    for sid in read(HERE / 'protocol.json')['cases']:
        old, source, label = original[sid], sources[sid], labels[sid]
        times = {im['frame']:im['pts_seconds'] for im in source['images']}
        lower, upper = times[label['lower_frame']], times[label['upper_frame']]
        row = dict(ID=sid, interval_seconds=[lower,upper], arms={})
        history = read(MARKER / 'run' / f'{sid}_marked/result.json')
        control, legend = records[(sid,'control')], records[(sid,'legend')]
        assert control['call']['processor_input_sha256'] == history['call']['processor_input_sha256']
        assert control['call']['image_sha256'] == legend['call']['image_sha256']
        assert control['call']['processor_input_sha256']['pixel_values'] == legend['call']['processor_input_sha256']['pixel_values']
        assert control['call']['processor_input_sha256']['image_grid_thw'] == legend['call']['processor_input_sha256']['image_grid_thw']
        row['historical_control_raw_match'] = control['raw'].strip() == history['raw'].strip()
        row['historical_control_frame_match'] = control['entry_frame'] == history['entry_frame']
        paths = [ROOT/im['path'] for im in source['images']]
        motion = np.load(prior.OLD / 'mac_run' / sid / 'motion.npz')
        for arm in ['control','legend']:
            answer = records[(sid,arm)]
            class Replay:
                count = 0
                def ask(self, images, prompt, max_new_tokens):
                    i = self.count
                    self.count += 1
                    call = old['calls'][i]
                    assert prompt == call['prompt'] and max_new_tokens == call['max_new_tokens']
                    assert [prior.rgb_sha(prior.bounded(im)) for im in images] == call['image_sha256']
                    return answer['raw'] if i == 2 else call['text']
            replay = Replay()
            prediction, _ = reference._predict_file(paths, motion['base_scores'], motion['new_scores'], replay)
            assert replay.count == 4 and prediction['entry_frame'] == answer['entry_frame']
            assert all(prediction[k] == old['baseline_prediction'][k] for k in ['collision_frame','entry_side','evasion_space'])
            replay_count += replay.count - 1
            row['arms'][arm] = dict(prediction=prediction, raw=answer['raw'], valid_offered=answer['valid_offered_integer'],
                  parser_changed=answer['parser_changed'], time_seconds=times[answer['entry_frame']], **prior.grade(times[answer['entry_frame']],lower,upper))
        a,b = (row['arms'][x] for x in ['control','legend'])
        row['gain'] = a['result']=='wrong' and b['result']=='correct' and b['valid_offered'] and not b['parser_changed']
        row['loss'] = a['result']=='correct' and b['result']=='wrong'
        row['new_false_first'] = lower > 0 and b['time_seconds']==0 and a['time_seconds']!=0
        row['paired_accuracy_delta'] = prior.paired_accuracy_delta(a['time_seconds'],b['time_seconds'],lower,upper)
        changes = [abs(b['time_seconds']-t)-abs(a['time_seconds']-t) for t in [lower,upper]+[t for t in [a['time_seconds'],b['time_seconds']] if lower<=t<=upper]]
        row['paired_absolute_error_delta_seconds'] = [min(changes),max(changes)]
        rows.append(row)
    n=len(rows)
    mae=[sum(r['paired_absolute_error_delta_seconds'][i] for r in rows)/n for i in [0,1]]
    gate=dict(gains=sum(r['gain'] for r in rows), losses=sum(r['loss'] for r in rows), new_false_first=sum(r['new_false_first'] for r in rows),
              paired_MAE_nonincrease=mae[1]<=1e-9, control_matches_history=all(r['historical_control_raw_match'] and r['historical_control_frame_match'] for r in rows))
    gate['pass']=gate['gains']>=1 and gate['losses']==0 and gate['new_false_first']==0 and gate['paired_MAE_nonincrease'] and gate['control_matches_history']
    write(HERE/'evaluation.json',dict(status='complete',actual_model_calls=len(records),cached_answers_replayed=replay_count,rows=rows,
          metrics={a:prior.aggregate([r['arms'][a] for r in rows]) for a in ['control','legend']},
          paired_accuracy_delta_bounds=[sum(r['paired_accuracy_delta'][i] for r in rows)/n for i in [0,1]],
          paired_MAE_delta_bounds_seconds=mae,gate=gate,other_three_outputs_unchanged=n,official_S2=None))


def state_evaluate():
    records=results(state=True)
    refs={r['ID']:r for who in ['a','b'] for r in read(HERE/f'state_reference_{who}.json')['cases']}
    rows=[]
    for (sid,_), result in records.items():
        try:
            obj=json.loads(result['raw'])
        except (ValueError,TypeError):
            obj=None
        value=obj.get('lane_state') if isinstance(obj,dict) else None
        valid=isinstance(value,str) and value in ['INSIDE','OUTSIDE','UNCERTAIN']
        assert result['valid_enum']==valid and result['state']==(value if valid else None)
        target=refs[sid]['state']
        rows.append(dict(ID=sid,reference=target,eligible=target!='UNKNOWN',prediction=result['state'],valid_enum=valid,
                         correct=(target==value) if target!='UNKNOWN' else None,raw=result['raw']))
    eligible=[r for r in rows if r['eligible']]
    classes=['INSIDE','OUTSIDE']
    predicted=['INSIDE','OUTSIDE','UNCERTAIN',None]
    confusion=[[sum(r['reference']==t and r['prediction']==p for r in eligible) for p in predicted] for t in classes]
    write(HERE/'state_evaluation.json',dict(status='complete',actual_model_calls=len(records),rows=rows,
          eligible=len(eligible),correct=sum(r['correct'] for r in eligible),accuracy=sum(r['correct'] for r in eligible)/len(eligible) if eligible else None,
          confusion=dict(reference_rows=classes,prediction_columns=predicted,matrix=confusion),
          uncertain=sum(r['prediction']=='UNCERTAIN' for r in rows),invalid=sum(not r['valid_enum'] for r in rows),
          scope='Separate first-frame state diagnosis; not timing prediction and no automatic entry correction',official_S2=None))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('action',choices=['freeze','run','evaluate','state_run','state_worker','state_evaluate'])
    parser.add_argument('--job',type=Path)
    parser.add_argument('--out',type=Path)
    args=parser.parse_args()
    if args.action=='state_worker':state_worker(args.job,args.out)
    elif args.action=='state_run':run(state=True)
    else:globals()[args.action]()
