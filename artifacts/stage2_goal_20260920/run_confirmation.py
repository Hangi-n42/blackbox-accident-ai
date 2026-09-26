"""Run the already verified Mac paired worker on frozen CCD confirmation inputs."""
from datetime import datetime, timezone
import hashlib
from importlib import metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
WORKER = ROOT / 'artifacts/stage2_goal_20260919/expert_mac_review/run_paired_mac.py'
sys.path.insert(0, str(ROOT / 'scripts/mac'))
from run_stage2 import ENV, write


def read(path):
    return json.loads(path.read_text())


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


if __name__ == '__main__':
    assert platform.system() == 'Darwin' and platform.machine() == 'arm64'
    frozen = read(HERE / 'freeze.json')
    assert frozen['status'] == 'locked_before_predictions'
    for name, digest in frozen['files'].items():
        assert sha(ROOT / name) == digest, name
    inputs = read(HERE / 'ccd_intake/inputs.json')
    excluded = set(frozen['excluded_duplicate_ids'])
    cases = [case for case in inputs if case['ID'] not in excluded]
    assert len(cases) == frozen['run_cases'] and cases
    out = HERE / 'mac_run'
    out.mkdir(exist_ok=False)
    report = dict(status='running', started_utc=datetime.now(timezone.utc).isoformat(),
                  freeze_sha256=sha(HERE / 'freeze.json'), worker_source_sha256=sha(WORKER),
                  platform=platform.platform(), python=sys.version,
                  chip=subprocess.check_output(['sysctl', '-n', 'machdep.cpu.brand_string'], text=True).strip(),
                  packages={name: metadata.version(name) for name in ['mlx', 'mlx-vlm', 'torch', 'transformers', 'numpy', 'Pillow']},
                  configuration=dict(policy='fixed_auxiliary_sum_removal', compute_dtype='native', decode_mode='sync', deepstack_fix=True),
                  workers=[], videos=[])
    write(out / 'report.json', report)
    env = dict(os.environ, **ENV, BLACKBOX_POLICY='baseline', BLACKBOX_DEEPSTACK_FIX='1',
               BLACKBOX_COMPUTE_DTYPE='native', BLACKBOX_DECODE_MODE='sync')
    for case in cases:
        sid = case['ID']
        job = dict(ID=sid, paths=[str(ROOT / image['path']) for image in case['images']],
                   image_sha256=[image['sha256'] for image in case['images']])
        job_path = out / (sid + '.job.json')
        write(job_path, job)
        started = time.perf_counter()
        with (out / (sid + '.log')).open('x') as log:
            process = subprocess.run([sys.executable, str(WORKER), '--worker-job', str(job_path), '--output', str(out / sid)],
                                     env=env, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        report['workers'].append(dict(ID=sid, exit_status=process.returncode, wall_seconds=time.perf_counter() - started))
        if process.returncode:
            report['status'] = 'failed'
            write(out / 'report.json', report)
            raise SystemExit(process.returncode)
        # The reused worker's fixed group string is a logging legacy; manifest defines CCD provenance.
        record = read(out / sid / 'result.json')
        report['videos'].append(record)
        write(out / 'report.json', report)
        print(json.dumps(report['workers'][-1]), flush=True)
    report.update(status='complete', ended_utc=datetime.now(timezone.utc).isoformat())
    write(out / 'report.json', report)
