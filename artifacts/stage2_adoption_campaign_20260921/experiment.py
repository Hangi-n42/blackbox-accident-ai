"""Frozen prospective Q3 packaging comparison; existing runtime and four-call baseline."""
import argparse, hashlib, json, os, platform, resource, socket, subprocess, sys, time
from datetime import datetime, timezone
from pathlib import Path
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT/'scripts/mac'))
from run_stage2 import ENV
read=lambda p:json.loads(p.read_text())
def write(p,v): p.write_text(json.dumps(v,ensure_ascii=False,indent=2))
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def now(): return datetime.now(timezone.utc)
def check():
    assert now()<datetime.fromisoformat(read(HERE/'budget_protocol.json')['deadline_utc'].replace('Z','+00:00'))
    for p,h in read(HERE/'freeze.json')['files'].items(): assert sha(ROOT/p)==h,p

def worker(sid,arm):
    check(); assert platform.system()=='Darwin' and platform.machine()=='arm64'
    out=HERE/'run'/f'{sid}_{arm}';out.mkdir(parents=True,exist_ok=False)
    def deny(*a,**k): raise RuntimeError('Offline inference network attempt')
    socket.socket.connect=socket.socket.connect_ex=socket.create_connection=deny
    sys.path.insert(0,str(ROOT/'artifacts/submissions/verify_v6/model/stage2/code'))
    from solution import stage2_v2 as v2
    from mlx_stage2 import MLXVLM
    import numpy as np, torch
    torch.set_num_threads(2);np.random.seed(0);torch.manual_seed(0)
    source=read(HERE/'intake'/sid/'input_manifest.json')
    paths=[ROOT/i['path'] for i in source['images']]
    assert all(sha(path)==item['sha256'] for path,item in zip(paths,source['images']))
    base=read(HERE/'baseline'/sid/'result.json')
    frames=base['diagnostics']['entry_candidates']
    assert 1<=len(frames)<=12 and len(set(frames))==len(frames)
    indices=[next(i for i,p in enumerate(paths) if v2._frame_number(p)==f) for f in frames]
    sheet=v2._sheet(paths,indices,columns=4)
    prompt=base['calls'][2]['prompt']
    assert prompt.count('These frames are chronological, left to right then top to bottom. ')==1
    images=[sheet]
    if arm=='candidate':
        columns=min(4,len(frames))
        images=[sheet.crop(((i%columns)*384,(i//columns)*256,(i%columns+1)*384,(i//columns+1)*256)) for i in range(len(frames))]
        prompt=prompt.replace('These frames are chronological, left to right then top to bottom. ', 'These images are chronological in the order provided. ',1)
    started=time.perf_counter();vlm=MLXVLM(ROOT/'artifacts/mac_experiments/stage2_mlx/model',out)
    answer=vlm.ask(images,prompt,max_new_tokens=40)
    parsed=v2._json_object(answer); index=v2._choice(parsed,'entry_frame',paths,indices,0)
    pred=dict(base['baseline_prediction']);pred['entry_frame']=v2._frame_number(paths[index])
    assert all(pred[k]==base['baseline_prediction'][k] for k in ('collision_frame','entry_side','evasion_space'))
    write(out/'result.json',dict(ID=sid,arm=arm,prediction=pred,raw=answer,parsed=parsed,candidate_frames=frames,
      calls=vlm.calls,model_load_seconds=vlm.load_seconds,wall_seconds=time.perf_counter()-started,
      process_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
      valid_raw_choice=type(parsed.get('entry_frame')) is int and parsed.get('entry_frame') in frames))
    if arm=='control':
        original=base['calls'][2]; current=vlm.calls[0]
        for key in ('text','token_trace','processor_input_sha256','image_sizes','image_sha256','prompt_sha256'):
            assert original[key]==current[key], ('control_mismatch',sid,key)
        assert pred==base['baseline_prediction']

def run(mode):
    check(); env=dict(os.environ,**ENV,BLACKBOX_POLICY='baseline',BLACKBOX_DEEPSTACK_FIX='1',BLACKBOX_COMPUTE_DTYPE='native',BLACKBOX_DECODE_MODE='sync')
    cases=read(HERE/'intake/inputs.json')
    report=dict(mode=mode,status='running',started_utc=now().isoformat(),workers=[])
    out=HERE/mode;out.mkdir(exist_ok=False)
    for case in cases:
        sid=case['ID'];check()
        arms=['baseline'] if mode=='baseline' else ['control','candidate']
        for arm in arms:
            check()
            remaining=(datetime.fromisoformat(read(HERE/'budget_protocol.json')['deadline_utc'].replace('Z','+00:00'))-now()).total_seconds()
            if arm=='baseline':
                job=out/f'{sid}.job.json';write(job,dict(ID=sid,paths=[str(ROOT/i['path']) for i in case['images']],image_sha256=[i['sha256'] for i in case['images']]))
                args=[str(ROOT/'artifacts/stage2_goal_20260919/expert_mac_review/run_paired_mac.py'),'--worker-job',str(job),'--output',str(out/sid)]
            else: args=[str(Path(__file__)),'worker','--id',sid,'--arm',arm]
            started=time.perf_counter()
            with (out/f'{sid}_{arm}.log').open('x') as log:
                try:
                    proc=subprocess.run([sys.executable,*args],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=min(180,remaining))
                except Exception as error:
                    report.update(status='failed',failed_case=sid,failed_arm=arm,error=repr(error));write(out/'execution.json',report);raise
            row=dict(ID=sid,arm=arm,exit_code=proc.returncode,wall_seconds=time.perf_counter()-started)
            report['workers'].append(row);write(out/'execution.json',report);print(json.dumps(row),flush=True)
            if proc.returncode:
                report.update(status='failed',failed_case=sid,failed_arm=arm);write(out/'execution.json',report);raise SystemExit(proc.returncode)
    report.update(status='complete',ended_utc=now().isoformat());write(out/'execution.json',report)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['baseline','paired','worker']);p.add_argument('--id');p.add_argument('--arm');a=p.parse_args()
    if a.mode=='worker':worker(a.id,a.arm)
    else:run(a.mode)
