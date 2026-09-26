"""One automatic start-state gate; cached other outputs preserved, Q3 freshly controlled."""
import argparse,hashlib,importlib.util,json,os,platform,subprocess,sys,time
from datetime import datetime,timezone
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
LEGEND=ROOT/'artifacts/stage2_entry_legend_20260920'
spec=importlib.util.spec_from_file_location('legend_worker',LEGEND/'experiment.py');old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
read,write,sha=old.read,old.write,old.sha
PROMPT="The large image on the LEFT is the first frame of the clip. The small images in the RIGHT column are chronological context around an automatically estimated collision time. Identify the OTHER vehicle that collides with the camera car in that context, then find that SAME vehicle in the first frame. In the first frame, is that vehicle already inside the camera car's driving lane? INSIDE means at least one of its wheels touches the lane boundary or is on its inside. OUTSIDE means the vehicle is outside without touching the lane. At an intersection, extend the camera car's lane boundaries forward. Return JSON with lane_state only: INSIDE, OUTSIDE, or UNCERTAIN. Use UNCERTAIN if the same collision counterpart cannot be identified in the first frame or its lane relation cannot be determined. Do not report a later lane entry."

def apply_policy(base,state,first):
    return {**base,'entry_frame':first if state=='INSIDE' else base['entry_frame']}
assert apply_policy({'entry_frame':9,'collision_frame':20},'INSIDE',7)=={'entry_frame':7,'collision_frame':20}
assert apply_policy({'entry_frame':9},None,7)=={'entry_frame':9}

def prepare():
    from PIL import Image
    from solution import stage2_v2 as v2
    decomposition=read(HERE/'decomposition.json');assert decomposition['summary']['stage_full_coverage']['candidates']==decomposition['summary']['eligible']
    assert decomposition['summary']['before_start']==4
    out=HERE/'inputs';out.mkdir(exist_ok=False);checks=[]
    traces={r['ID']:r for r in read(HERE/'path_replay.json')['rows']}
    for row in read(HERE/'inventory.json')['cases']:
        if not row['review_selected']:continue
        sid=row['ID'];src=row['source'];paths=[ROOT/r['path'] for r in src['images']];numbers=[r['frame'] for r in src['images']];trace=traces[sid]
        anchor=numbers.index(trace['vlm_contact_frame']);context=sorted({max(0,anchor-2),anchor,min(len(paths)-1,anchor+2)})
        low=v2._sheet(paths,[0],columns=1);first=low.resize((1152,768),Image.Resampling.NEAREST)
        gate=Image.new('RGB',(1536,768),'#171717');gate.paste(first,(0,0));gate.paste(v2._sheet(paths,context,columns=1),(1152,0))
        gate.save(out/f'{sid}_gate.png')
        control=HERE/'q3_inputs'/f'{sid}.png';calls=read(ROOT/row['baseline_folder']/'calls.json')
        for arm,path,prompt in [('control',control,calls[2]['prompt']),('gate',out/f'{sid}_gate.png',PROMPT)]:
            write(out/f'{sid}_{arm}.job.json',dict(ID=sid,image=str(path),sha256=sha(path),prompt=prompt,max_new_tokens=40))
        checks.append(dict(ID=sid,first_frame=numbers[0],anchor_original_frame=numbers[anchor],context_frames=[numbers[i] for i in context],
            canvas=[1536,768],first_view_rectangle=[0,0,1152,768],first_resize='unmarked384x256 nearest3x',
            first_view_rgb_sha256=hashlib.sha256(first.tobytes()).hexdigest(),gate_rgb_sha256=hashlib.sha256(gate.tobytes()).hexdigest(),
            default_1200000_budget_preserves_size=True,expert_pixels_or_labels_in_model_input=False))
    write(HERE/'input_checks.json',dict(status='prepared',cases=checks))
    write(HERE/'protocol.json',dict(created_utc=datetime.now(timezone.utc).isoformat(),selected_change='Automatic start-state gate only; preserve original Q3 candidate generator and base outputs',
        selection_reason='Re-reviewed8: all stages retain interval coverage;7definite selection errors,1unresolved;4before_start all missed. Privileged-marker gate prior evidence motivates a fully automatic candidate, not repeating coordinate or temporal-narrowing diagnostics.',
        frames_source='Existing 8 prior-reference CCD cases, fixed before new candidate calls; prior exposed development data',prompt=PROMPT,max_new_tokens=40,
        model_calls_planned=16,arms=['control','gate'],policy='INSIDE -> first original frame; OUTSIDE/UNCERTAIN/invalid -> baseline entry. Other three output values exactly copied.',
        input_policy='One1536x768canvas: left1152x768 unmarked first frame from native _sheet384x256 nearest3x; right384x256tiles around parsed Q2 chosen contact index±2. No manual counterpart or event coordinates. Existing1.2M budget and model fixed.',
        evaluation='Fresh originalQ3 controls must reproduce all8historical raw selections and processor tensors. Score conservative interval refs, paired accuracy/MAE bounds; unknowns unscored. No officialS2.',
        success_gate='At least1definite entry gain; no definite/possible accuracy loss; no new first-frame selection on during_clip; pairedMAE maximum delta <=0; both strata present; allcontrols exact; other3outputs unchanged; no INSIDE commitment on preflight identity-unresolvable CCD_000052.',
        limitations='This policy adds high-resolution first-frame exposure, automatic contact context and a separate binary decision together. Do not isolate their causal contributions. Q2 context may identify wrong counterpart; same-target reasoning not certified. MLX response != CUDA NF4.',
        stop='One frozen candidate, exactly8Q3 control+8gate calls; no prompt tuning or retries after responses. No submission-source change.'))
    write(HERE/'STATUS.json',dict(status='prepared_before_single_candidate_calls',planned_model_calls=16,actual_model_calls=0))

def freeze():
    assert not (HERE/'freeze.json').exists() and not (HERE/'run').exists()
    assert read(HERE/'preflight_review.json')['status']=='PASS' and read(HERE/'input_peer_review.json')['status']=='PASS'
    files=dict(read(ROOT/'artifacts/stage2_spatial_grounding_540_20260920/freeze.json')['files'])
    for name,digest in files.items():assert sha(ROOT/name)==digest,name
    for row in read(HERE/'inventory.json')['cases']:
        for n in ['result.json','calls.json','motion.npz','worker_report.json']:
            p=ROOT/row['baseline_folder']/n;files[str(p.relative_to(ROOT))]=sha(p)
        for item in row['source']['images']:files[item['path']]=item['sha256']
    for p in (ROOT/'releases/v7/source/model/stage2/code').rglob('*.py'):files[str(p.relative_to(ROOT))]=sha(p)
    for p in HERE.rglob('*'):
        if p.is_file() and '__pycache__' not in p.parts and p.name!='STATUS.json':files[str(p.relative_to(ROOT))]=sha(p)
    for name,digest in files.items():assert sha(ROOT/name)==digest,name
    write(HERE/'freeze.json',dict(status='locked_before_automatic_start_gate',created_utc=datetime.now(timezone.utc).isoformat(),model_calls_planned=16,files=files))

def run():
    assert platform.system()=='Darwin' and platform.machine()=='arm64'
    frozen=read(HERE/'freeze.json')
    for name,digest in frozen['files'].items():assert sha(ROOT/name)==digest,name
    out=HERE/'run';out.mkdir(exist_ok=False)
    env=dict(os.environ,**old.prior.ENV,BLACKBOX_DEEPSTACK_FIX='1',BLACKBOX_COMPUTE_DTYPE='native',BLACKBOX_DECODE_MODE='sync')
    report=dict(status='running',started_utc=datetime.now(timezone.utc).isoformat(),freeze_sha256=sha(HERE/'freeze.json'),workers=[])
    write(out/'report.json',report)
    for ref in read(HERE/'references.json')['cases']:
        for arm in ['control','gate']:
            sid=ref['ID'];start=time.perf_counter();name=f'{sid}_{arm}'
            with (out/f'{name}.log').open('x') as log:
                p=subprocess.run([sys.executable,str(LEGEND/'experiment.py'),'state_worker','--job',str(HERE/'inputs'/f'{name}.job.json'),'--out',str(out/name)],env=env,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
            report['workers'].append(dict(ID=sid,arm=arm,exit_status=p.returncode,wall_seconds=time.perf_counter()-start));write(out/'report.json',report)
            print(json.dumps(report['workers'][-1]),flush=True)
            if p.returncode:raise SystemExit(p.returncode)
    report.update(status='complete',ended_utc=datetime.now(timezone.utc).isoformat());write(out/'report.json',report)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','freeze','run']);globals()[p.parse_args().action]()
