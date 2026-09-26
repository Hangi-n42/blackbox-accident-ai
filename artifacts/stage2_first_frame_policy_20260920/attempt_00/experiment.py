"""One frozen first-frame override policy, reusing the existing Mac workers."""
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

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
PREV=ROOT/'artifacts/stage2_state_resolution_20260920'
LEGEND=ROOT/'artifacts/stage2_entry_legend_20260920'
BASEWORKER=ROOT/'artifacts/stage2_goal_20260919/expert_mac_review/run_paired_mac.py'
spec=importlib.util.spec_from_file_location('previous_state',LEGEND/'experiment.py')
old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
read,write,sha=old.read,old.write,old.sha


def apply_policy(baseline,state,first_frame):
    result=dict(baseline)
    if state=='INSIDE':result['entry_frame']=first_frame
    return result


assert apply_policy({'entry_frame':9,'collision_frame':20},'INSIDE',7)=={'entry_frame':7,'collision_frame':20}
for _state in ['OUTSIDE','UNCERTAIN',None]:
    assert apply_policy({'entry_frame':9},_state,7)=={'entry_frame':9}


def prepare():
    from PIL import Image,ImageDraw,ImageOps
    from solution import stage2_v2 as v2
    out=HERE/'inputs';out.mkdir(exist_ok=False)
    sources=read(HERE/'intake/inputs.json')
    refs={r['ID']:r for r in read(HERE/'references.json')['cases']}
    prompt=read(LEGEND/'inputs/CCD_000688_state.job.json')['prompt']
    rows=[]
    for source in sources:
        sid=source['ID'];ref=refs[sid]
        paths=[ROOT/r['path'] for r in source['images']]
        for path,meta in zip(paths,source['images']):assert sha(path)==meta['sha256']
        write(out/f'{sid}_baseline.job.json',dict(ID=sid,paths=[str(p) for p in paths],image_sha256=[r['sha256'] for r in source['images']]))
        row=dict(ID=sid,mark_approved=ref['mark_approved'],marker_source_box=ref['marker_box_xyxy'])
        if ref['mark_approved']:
            low=v2._sheet(paths,[0],columns=1)
            with Image.open(paths[0]) as image:
                width,height=image.size;iw,ih=ImageOps.contain(image,(384,228)).size
            x0,y0,x1,y1=ref['marker_box_xyxy']
            assert 0<=x0<x1<width and 0<=y0<y1<height
            box=[(384-iw)//2+round(x0*iw/width),28+round(y0*ih/height),
                 (384-iw)//2+round(x1*iw/width),28+round(y1*ih/height)]
            assert box[0]<box[2] and box[1]<box[3]
            ImageDraw.Draw(low).rectangle(box,outline=(255,255,0),width=2)
            image=low.resize((1152,768),Image.Resampling.NEAREST)
            low.save(out/f'{sid}_low.png');path=out/f'{sid}_state.png';image.save(path)
            write(out/f'{sid}_state.job.json',dict(ID=sid,image=str(path),sha256=sha(path),prompt=prompt,max_new_tokens=40))
            row.update(marker_low_box=box,source_size=[width,height],low_size=list(low.size),size=list(image.size),rgb_sha256=old.prior.rgb_sha(image))
        rows.append(row)
    write(HERE/'input_checks.json',dict(cases=rows,status='built_pending_peer_QA'))


def check_frozen():
    frozen=read(HERE/'freeze.json')
    for name,digest in frozen['files'].items():assert sha(ROOT/name)==digest,name
    return frozen


def freeze():
    assert not (HERE/'freeze.json').exists() and not (HERE/'run').exists()
    for who in ['a','b']:
        record=read(HERE/f'review_{who}.json')
        assert record['model_predictions_seen'] is False and record['peer_review_seen'] is False
    assert read(HERE/'input_peer_review.json')['status']=='PASS'
    assert read(HERE/'preflight_review.json')['status']=='PASS'
    refs=read(HERE/'references.json')['cases'];sources=read(HERE/'intake/inputs.json')
    assert len(sources)==len(refs)==12 and {r['ID'] for r in refs}=={r['ID'] for r in sources}
    files=dict(read(PREV/'freeze.json')['files'])
    for name,digest in files.items():assert sha(ROOT/name)==digest,name
    for path in [PREV/'freeze.json',LEGEND/'experiment.py',BASEWORKER]:files[str(path.relative_to(ROOT))]=sha(path)
    for path in HERE.rglob('*'):
        if path.is_file() and '__pycache__' not in path.parts and path.name not in ['STATUS.json']:
            files[str(path.relative_to(ROOT))]=sha(path)
    write(HERE/'freeze.json',dict(status='locked_before_all_new_predictions',created_utc=datetime.now(timezone.utc).isoformat(),
          run_cases=12,state_cases=sum(r['mark_approved'] for r in refs),files=files))


def run():
    assert platform.system()=='Darwin' and platform.machine()=='arm64'
    frozen=check_frozen();out=HERE/'run';out.mkdir(exist_ok=False)
    env=dict(os.environ,**old.prior.ENV,BLACKBOX_POLICY='baseline',BLACKBOX_DEEPSTACK_FIX='1',BLACKBOX_COMPUTE_DTYPE='native',BLACKBOX_DECODE_MODE='sync',PYTHONDONTWRITEBYTECODE='1')
    report=dict(status='running',started_utc=datetime.now(timezone.utc).isoformat(),platform=platform.platform(),
                freeze_sha256=sha(HERE/'freeze.json'),workers=[])
    write(out/'report.json',report)
    for case in read(HERE/'intake/inputs.json'):
        sid=case['ID']
        for arm in ['baseline','state']:
            job=HERE/'inputs'/f'{sid}_{arm}.job.json'
            if not job.exists():assert arm=='state';continue
            name=f'{sid}_{arm}';start=time.perf_counter()
            args=[str(BASEWORKER),'--worker-job',str(job),'--output',str(out/name)] if arm=='baseline' else [str(LEGEND/'experiment.py'),'state_worker','--job',str(job),'--out',str(out/name)]
            with (out/f'{name}.log').open('x') as log:
                proc=subprocess.run([sys.executable,*args],env=env,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
            report['workers'].append(dict(ID=sid,arm=arm,exit_status=proc.returncode,wall_seconds=time.perf_counter()-start))
            print(json.dumps(report['workers'][-1]),flush=True)
            if proc.returncode:
                report['status']='failed';write(out/'report.json',report);raise SystemExit(proc.returncode)
            write(out/'report.json',report)
    report.update(status='complete',ended_utc=datetime.now(timezone.utc).isoformat())
    write(out/'report.json',report)


def evaluate():
    from PIL import Image
    check_frozen()
    sources=read(HERE/'intake/inputs.json');refs={r['ID']:r for r in read(HERE/'references.json')['cases']}
    run=read(HERE/'run/report.json')
    expected={(r['ID'],'baseline') for r in sources}|{(r['ID'],'state') for r in refs.values() if r['mark_approved']}
    assert run['status']=='complete' and len(run['workers'])==len(expected)
    assert {(w['ID'],w['arm']) for w in run['workers']}==expected and all(w['exit_status']==0 for w in run['workers'])
    rows=[];total_calls=0
    for source in sources:
        sid=source['ID'];ref=refs[sid];folder=HERE/'run'/f'{sid}_baseline'
        base=read(folder/'result.json');wr=read(folder/'worker_report.json');calls=read(folder/'calls.json')
        assert base['ID']==wr['ID']==sid and wr['status']=='complete' and wr['model_calls']==len(calls)==4 and wr['network_attempts']==0
        assert base['calls']==calls
        total_calls+=4
        baseline=base['baseline_prediction'];first=source['images'][0]['frame'];state=None;raw=None;state_call=None
        if ref['mark_approved']:
            folder=HERE/'run'/f'{sid}_state';sr=read(folder/'result.json');sw=read(folder/'worker_report.json');sc=read(folder/'calls.json')
            assert sr['ID']==sw['ID']==sid and sw['status']=='complete' and sw['model_calls']==len(sc)==1 and sw['network_attempts']==0
            assert sc[0]==sr['call'] and sc[0]['text'].strip()==sr['raw'].strip()
            try:obj=json.loads(sr['raw'])
            except (ValueError,TypeError):obj=None
            value=obj.get('lane_state') if isinstance(obj,dict) else None
            valid=isinstance(value,str) and value in ['INSIDE','OUTSIDE','UNCERTAIN']
            state=value if valid else None
            assert sr['valid_enum']==valid and sr['state']==state
            job=read(HERE/'inputs'/f'{sid}_state.job.json')
            with Image.open(job['image']) as image:
                assert sc[0]['image_sha256']==[old.prior.rgb_sha(old.prior.bounded(image))]
                assert sc[0]['image_sizes']==[[1152,768]]
            assert sc[0]['prompt']==job['prompt'] and sc[0]['max_new_tokens']==40
            total_calls+=1;raw=sr['raw'];state_call=sc[0]
        treatment=apply_policy(baseline,state,first)
        assert all(treatment[k]==baseline[k] for k in ['collision_frame','entry_side','evasion_space'])
        times={r['frame']:r['pts_seconds'] for r in source['images']}
        for pred in [baseline,treatment]:
            assert all(type(pred[k]) is int and pred[k] in times for k in ['entry_frame','collision_frame'])
        row=dict(ID=sid,source_group=source['source_group'],baseline=baseline,treatment=treatment,mark_approved=ref['mark_approved'],
                 state=state,state_raw=raw,state_reference=ref['first_frame_state'],state_reference_correct=(state==ref['first_frame_state']) if ref['mark_approved'] and ref['first_frame_state']!='UNKNOWN' else None,
                 state_prompt_tokens=state_call['prompt_tokens'] if state_call else None,overridden=state=='INSIDE',entry_changed=baseline['entry_frame']!=treatment['entry_frame'],
                 other_three_unchanged=True,entry_reference=ref['entry'],timing=None,
                 known_outside_inside_response=ref['first_frame_state']=='OUTSIDE' and state=='INSIDE',
                 known_outside_new_first=ref['first_frame_state']=='OUTSIDE' and state=='INSIDE' and baseline['entry_frame']!=first)
        truth=ref['entry']
        if truth['evaluation_eligible']:
            lower,upper=times[truth['lower_frame']],times[truth['upper_frame']]
            a,b=times[baseline['entry_frame']],times[treatment['entry_frame']]
            row['timing']=dict(interval_seconds=[lower,upper],baseline_seconds=a,treatment_seconds=b,
                 baseline=old.prior.grade(a,lower,upper),treatment=old.prior.grade(b,lower,upper),
                 paired_accuracy_delta=old.prior.paired_accuracy_delta(a,b,lower,upper),
                 new_false_first=truth['status']=='during_clip' and b==times[first] and a!=times[first])
            t=row['timing'];t['definite_gain']=t['baseline']['result']=='wrong' and t['treatment']['result']=='correct'
            t['definite_loss']=t['baseline']['result']=='correct' and t['treatment']['result']=='wrong'
            points=[lower,upper]+[x for x in [a,b] if lower<=x<=upper]
            errors=[abs(b-x)-abs(a-x) for x in points]
            t['paired_MAE_delta_seconds']=[min(errors),max(errors)]
        rows.append(row)
    eligible=[r for r in rows if r['timing'] is not None];n=len(eligible)
    metrics={arm:old.prior.aggregate([r['timing'][arm] for r in eligible]) for arm in ['baseline','treatment']}
    strata={kind:{arm:old.prior.aggregate([r['timing'][arm] for r in eligible if r['entry_reference']['status']==kind]) for arm in ['baseline','treatment']} for kind in ['before_start','during_clip']}
    bounds={name:([sum(r['timing'][field][i] for r in eligible)/n for i in [0,1]] if n else None)
            for name,field in [('accuracy','paired_accuracy_delta'),('mae_seconds','paired_MAE_delta_seconds')]}
    gate=dict(both_strata=all(strata[k]['baseline']['n']>=1 for k in strata),gains=sum(r['timing']['definite_gain'] for r in eligible),
              losses=sum(r['timing']['definite_loss'] for r in eligible),possible_loss_cases=sum(r['timing']['paired_accuracy_delta'][0]<0 for r in eligible),
              new_false_first=sum(r['timing']['new_false_first'] for r in eligible),
              known_outside_new_first=sum(r['known_outside_new_first'] for r in rows),paired_MAE_nonincrease=bounds['mae_seconds'] is not None and bounds['mae_seconds'][1]<=1e-9)
    gate['pass']=gate['both_strata'] and gate['gains']>=1 and gate['losses']==gate['possible_loss_cases']==gate['new_false_first']==gate['known_outside_new_first']==0 and gate['paired_MAE_nonincrease']
    result=dict(status='complete',actual_model_calls=total_calls,rows=rows,metrics=metrics,strata=strata,paired_delta_bounds=bounds,gate=gate,
                selected_cases=len(rows),entry_eligible=n,state_queried=sum(r['mark_approved'] for r in rows),state_reference_scored=sum(r['state_reference_correct'] is not None for r in rows),
                state_reference_correct=sum(r['state_reference_correct'] is True for r in rows),
                known_outside_queried=sum(r['mark_approved'] and r['state_reference']=='OUTSIDE' for r in rows),
                known_outside_inside_responses=sum(r['known_outside_inside_response'] for r in rows),other_three_unchanged=len(rows),official_S2=None)
    write(HERE/'evaluation.json',result)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','freeze','run','evaluate']);globals()[p.parse_args().action]()
