"""Fresh Mac worker per new clip; four real calls shared by two final contact rules."""
import argparse
from datetime import datetime, timezone
import hashlib
from importlib import metadata
import json
import os
from pathlib import Path
import platform
import resource
import socket
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'scripts/mac'))
from run_stage2 import ENV, write


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def worker(job_path, out):
    job = read(job_path)
    assert set(job) == {'ID', 'paths', 'image_sha256'}
    paths = [Path(p) for p in job['paths']]
    assert len(paths) == len(job['image_sha256'])
    assert all(sha(p) == h for p, h in zip(paths, job['image_sha256']))
    out.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    report = dict(ID=job['ID'], status='running', model_calls=0, network_attempts=0,
                  started_utc=datetime.now(timezone.utc).isoformat())
    def deny(*args, **kwargs):
        report['network_attempts'] += 1
        raise RuntimeError('Unexpected Python network connection in offline Mac worker')
    socket.socket.connect = socket.socket.connect_ex = socket.create_connection = deny
    vlm = None
    try:
        sys.path.insert(0, str(ROOT / 'artifacts/submissions/verify_v6/model/stage2/code'))
        sys.path.insert(0, str(HERE.parent / 'candidate_runtime'))
        import candidate
        from mlx_stage2 import MLXVLM
        import numpy as np
        import pandas as pd
        import torch
        torch.set_num_threads(2)
        candidate.reference.primitives.cv2.setNumThreads(2)
        np.random.seed(0)
        torch.manual_seed(0)
        numbers = [candidate.reference.primitives._frame_number(path) for path in paths]
        assert numbers and numbers == sorted(set(numbers))
        vlm = MLXVLM(ROOT / 'artifacts/mac_experiments/stage2_mlx/model', out)
        motion_started = time.perf_counter()
        valid, base, new, features = candidate.reference._dual_motion_scan(paths)
        motion_seconds = time.perf_counter() - motion_started
        assert valid == paths
        prediction, diagnostics = candidate.predict_file(paths, base, new, features, vlm)
        baseline = diagnostics['baseline_prediction']
        assert len(vlm.calls) == 4
        for value in (baseline, prediction):
            assert set(value) == {'collision_frame', 'entry_frame', 'entry_side', 'evasion_space'}
            assert all(type(value[key]) is int and value[key] in numbers for key in ('collision_frame', 'entry_frame'))
            assert type(value['evasion_space']) is int and value['evasion_space'] in (0, 1)
            assert value['entry_side'] in ('LEFT', 'RIGHT')
        assert all(prediction[key] == baseline[key] for key in ('entry_frame', 'entry_side', 'evasion_space'))
        assert report['network_attempts'] == 0
        np.savez_compressed(out / 'motion.npz', features=features, base_scores=base, new_scores=new,
                            candidate_scores=candidate.scores_from_features(features))
        for label, values in [('baseline', baseline), ('candidate', prediction)]:
            pd.DataFrame([dict(ID=job['ID'], **values)], columns=candidate.COLUMNS).to_csv(out / (label + '.csv'), index=False)
        write(out / 'result.json', dict(ID=job['ID'], group='new_source_ai_review', frames=len(paths),
                                       baseline_prediction=baseline, candidate_prediction=prediction, diagnostics=diagnostics,
                                       calls=vlm.calls, motion_seconds=motion_seconds, model_load_seconds=vlm.load_seconds,
                                       max_peak_memory_GB=max(call['peak_memory'] for call in vlm.calls),
                                       execution_policy='fresh_worker_per_case_v1', model_path=str(ROOT / 'artifacts/mac_experiments/stage2_mlx/model')))
        report['status'] = 'complete'
    except Exception as error:
        report.update(status='failed', error_type=type(error).__name__, error=str(error))
        raise
    finally:
        report.update(model_calls=0 if vlm is None else len(vlm.calls), wall_seconds=time.perf_counter() - started,
                      ended_utc=datetime.now(timezone.utc).isoformat(),
                      max_process_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        write(out / 'worker_report.json', report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker-job', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.worker_job:
        worker(args.worker_job, args.output.resolve())
        return
    # Review records and interpretation rules must precede any predictions on these clips.
    locked = read(HERE / 'review_lock.json')
    assert locked['status'] == 'locked_before_model_predictions' and locked['model_predictions_seen'] is False
    for name, digest in locked['files'].items():
        assert sha(ROOT / name) == digest, name
    inputs = read(HERE / 'inputs.json')
    assert len(inputs) == 6 and len({case['ID'] for case in inputs}) == 6
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    assert platform.system() == 'Darwin' and platform.machine() == 'arm64'
    env = dict(os.environ, **ENV, BLACKBOX_POLICY='baseline', BLACKBOX_DEEPSTACK_FIX='1', BLACKBOX_COMPUTE_DTYPE='native', BLACKBOX_DECODE_MODE='sync')
    report = dict(status='running', platform='Mac MLX', configuration=dict(decode_mode='sync', compute_dtype='native',
                  deepstack_fix=True, policy='single_frozen_auxiliary_sum_removal'), review_lock_sha256=sha(HERE / 'review_lock.json'),
                  source_sha256=sha(Path(__file__)), workers=[], videos=[],
                  started_utc=datetime.now(timezone.utc).isoformat(),
                  environment=dict(platform=platform.platform(), python=sys.version, executable=sys.executable,
                      chip=subprocess.check_output(['sysctl', '-n', 'machdep.cpu.brand_string'], text=True).strip(),
                      physical_memory_bytes=int(subprocess.check_output(['sysctl', '-n', 'hw.memsize'], text=True)),
                      packages={name: metadata.version(name) for name in ['mlx', 'mlx-vlm', 'torch', 'transformers', 'numpy', 'opencv-python', 'Pillow']}))
    write(out / 'report.json', report)
    for case in inputs:
        job = dict(ID=case['ID'], paths=[str((ROOT / image['path']).resolve()) for image in case['images']],
                   image_sha256=[image['sha256'] for image in case['images']])
        job_path, destination = out / (case['ID'] + '.job.json'), out / case['ID']
        write(job_path, job)
        started = time.perf_counter()
        with (out / (case['ID'] + '.log')).open('x') as log:
            run = subprocess.run([sys.executable, str(Path(__file__)), '--worker-job', str(job_path), '--output', str(destination)],
                                 cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
        report['workers'].append(dict(ID=case['ID'], exit_status=run.returncode, wall_seconds=time.perf_counter() - started))
        if run.returncode:
            report['status'] = 'failed'
            write(out / 'report.json', report)
            raise SystemExit(run.returncode)
        report['videos'].append(read(destination / 'result.json'))
        write(out / 'report.json', report)
        print(json.dumps(report['workers'][-1]), flush=True)
    report['status'] = 'complete'
    report['ended_utc'] = datetime.now(timezone.utc).isoformat()
    write(out / 'report.json', report)


if __name__ == '__main__':
    main()
