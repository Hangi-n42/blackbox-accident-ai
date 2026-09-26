"""One canonical Mac Stage2 path: a fresh, identically configured worker per case."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
ENGINE = Path(__file__).with_name('mlx_stage2.py')
ENV = dict(OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', MKL_NUM_THREADS='2',
           HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', HF_HUB_DISABLE_IMPLICIT_TOKEN='1',
           TOKENIZERS_PARALLELISM='false', PYTHONDONTWRITEBYTECODE='1', PYTHONHASHSEED='0')

def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2))

def worker(job_path, output, model):
    # Identical order for every single-case and batch invocation.
    sys.path.insert(0, str(ENGINE.parent))
    from mlx_stage2 import MLXVLM
    sys.path.insert(0, str(ROOT/'artifacts/submissions/verify_v6/model/stage2/code'))
    from solution import stage2_uncapped_jerk_v6c as policy
    import numpy as np
    import pandas as pd
    import torch
    policy.primitives.cv2.setNumThreads(2)
    torch.set_num_threads(2)
    np.random.seed(0)
    torch.manual_seed(0)
    job = json.loads(job_path.read_text())
    paths = [Path(x) for x in job['paths']]
    numbers = [policy.primitives._frame_number(p) for p in paths]
    if not paths or len(numbers)!=len(set(numbers)):
        raise ValueError('Require nonempty unique original frame numbers')
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    vlm = MLXVLM(model, output)
    motion_start = time.perf_counter()
    valid, base, new, features = policy._dual_motion_scan(paths)
    if valid != paths:
        raise ValueError('Input decode failure; do not silently drop a frame')
    motion_seconds = time.perf_counter()-motion_start
    prediction, diagnostics = policy._predict_file(valid, base, new, vlm)
    baseline_prediction = dict(prediction)
    entry_policy = os.environ.get('BLACKBOX_POLICY', 'baseline')
    if entry_policy in ('temporal_v1', 'temporal_final_contact'):
        from temporal_entry import refine_entry
        prediction, detail = refine_entry(valid, prediction, diagnostics, vlm,
                                          final_contact=entry_policy == 'temporal_final_contact')
        diagnostics['temporal_entry'] = detail
    if prediction['collision_frame'] not in numbers or prediction['entry_frame'] not in numbers:
        raise ValueError('Prediction references missing frame')
    if prediction['entry_side'] not in ('LEFT','RIGHT') or prediction['evasion_space'] not in (0,1):
        raise ValueError('Invalid prediction category')
    expected_calls = 4 + diagnostics.get('temporal_entry', {}).get('calls', 0)
    if len(vlm.calls)!=expected_calls:
        raise ValueError('Expected four VLM calls')
    pd.DataFrame([dict(ID=job['ID'], **prediction)], columns=policy.COLUMNS).to_csv(output/'predictions.csv',index=False)
    np.savez_compressed(output/'motion.npz', features=features, base_scores=base, new_scores=new)
    result=dict(ID=job['ID'], group=job['group'], prediction=prediction, baseline_prediction=baseline_prediction, diagnostics=diagnostics,
                frames=len(paths), calls=vlm.calls, model_load_seconds=vlm.load_seconds,
                motion_seconds=motion_seconds, total_seconds=time.perf_counter()-started,
                execution_policy='fresh_worker_per_case_v1', model_path=str(model),
                source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob('*.py')})
    write(output/'result.json', result)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',type=Path,default=ROOT/'artifacts/public_eval/stage2')
    parser.add_argument('--model',type=Path,default=ROOT/'artifacts/mac_experiments/stage2_mlx/model')
    parser.add_argument('--deepstack-fix', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--compute-dtype', choices=['native','float32'], default='native')
    parser.add_argument('--decode-mode', choices=['legacy','sync'], default='sync')
    parser.add_argument('--policy', choices=['baseline','temporal_v1','temporal_final_contact'], default='baseline')
    parser.add_argument('--case',nargs='+',help='One or more image-folder IDs; default S2_001')
    parser.add_argument('--manifest',type=Path,help='Baseline inputs.json; mutually exclusive with --case')
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--worker-job',type=Path,help=argparse.SUPPRESS)
    args=parser.parse_args()
    if args.worker_job:
        worker(args.worker_job,args.output_dir,args.model.resolve());return
    if args.manifest and args.case:parser.error('Use --manifest or --case, not both')
    jobs=[]
    if args.manifest:
        for case in json.loads(args.manifest.read_text()):
            paths=[]
            for img in case['images']:
                p=(ROOT/img['path']).resolve()
                if hashlib.file_digest(p.open('rb'),'sha256').hexdigest()!=img['sha256']:
                    raise ValueError(f'Input hash mismatch: {p}')
                paths.append(str(p))
            jobs.append(dict(ID=case['ID'],group=case['group'],paths=paths))
    else:
        for case in args.case or ['S2_001']:
            if Path(case).name!=case:raise ValueError('Case must be one folder name')
            folder=args.data.resolve()/'images'/case
            paths=sorted((p for p in folder.iterdir() if p.suffix.lower() in {'.jpg','.jpeg','.png','.bmp','.webp'}),key=lambda p:int(p.stem.split('_')[-1]))
            jobs.append(dict(ID=case,group='public',paths=[str(p) for p in paths]))
    if len({(j['group'],j['ID']) for j in jobs})!=len(jobs):raise ValueError('Duplicate cases')
    for j in jobs:
        if any(Path(j[k]).name!=j[k] for k in ('group','ID')):raise ValueError('Unsafe case identifier')
    out=args.output_dir.resolve();out.mkdir(parents=True,exist_ok=False)
    env=dict(os.environ,**ENV)
    env['BLACKBOX_DEEPSTACK_FIX']='1' if args.deepstack_fix else '0'
    env['BLACKBOX_COMPUTE_DTYPE']=args.compute_dtype
    env['BLACKBOX_DECODE_MODE']=args.decode_mode
    env['BLACKBOX_POLICY']=args.policy
    report=dict(status='running', configuration=dict(decode_mode=args.decode_mode,compute_dtype=args.compute_dtype,deepstack_fix=args.deepstack_fix,policy=args.policy),execution_policy='fresh_worker_per_case_v1',videos=[],workers=[])
    write(out/'stage2_report.json',report)
    for i,job in enumerate(jobs):
        job_path=out/f'job_{i:03d}.json';write(job_path,job)
        dest=out/'stage2'/job['group']/job['ID'];started=time.perf_counter()
        command=[sys.executable,str(Path(__file__).resolve()),'--worker-job',str(job_path),'--model',str(args.model.resolve()),'--output-dir',str(dest)]
        with (out/f'worker_{i:03d}.log').open('x') as log:
            run=subprocess.run(command,env=env,stdout=log,stderr=subprocess.STDOUT,cwd=ROOT)
        report['workers'].append(dict(ID=job['ID'],exit_status=run.returncode,wall_seconds=time.perf_counter()-started))
        if run.returncode:
            report['status']='failed';write(out/'stage2_report.json',report);raise SystemExit(run.returncode)
        report['videos'].append(json.loads((dest/'result.json').read_text()))
        write(out/'stage2_report.json',report)
        print(json.dumps(report['workers'][-1]),flush=True)
    report['status']='complete';write(out/'stage2_report.json',report)

if __name__=='__main__':main()
