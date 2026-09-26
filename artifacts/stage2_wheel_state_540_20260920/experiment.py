"""Six fixed single-frame wheel-state queries; only canvas enlargement differs."""
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
PREV=ROOT/'artifacts/stage2_first_frame_policy_20260920'
LEGEND=ROOT/'artifacts/stage2_entry_legend_20260920'
spec=importlib.util.spec_from_file_location('state_impl',LEGEND/'experiment.py')
old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
read,write,sha=old.read,old.write,old.sha
ARMS=['low','upsampled']


def prepare():
    from PIL import Image,ImageDraw,ImageOps
    from solution import stage2_v2 as v2
    out=HERE/'inputs';out.mkdir(exist_ok=False)
    protocol=read(HERE/'protocol.json')
    source=next(r for r in read(PREV/'intake/inputs.json') if r['ID']==protocol['case'])
    paths=[ROOT/r['path'] for r in source['images']]
    rows=[]
    for ref in read(HERE/'references.json')['cases']:
        frame=ref['frame'];meta=next(r for r in source['images'] if r['frame']==frame)
        assert sha(ROOT/meta['path'])==meta['sha256']
        row=dict(frame=frame,source=meta,mark_approved=ref['mark_approved'])
        if ref['mark_approved']:
            index=next(i for i,r in enumerate(source['images']) if r['frame']==frame)
            clean=v2._sheet(paths,[index],columns=1);low=clean.copy()
            with Image.open(paths[index]) as native:
                w,h=native.size;iw,ih=ImageOps.contain(native,(384,228)).size
            x0,y0,x1,y1=ref['marker_box_xyxy']
            assert 0<=x0<x1<w and 0<=y0<y1<h
            ox=(384-iw)//2
            box=[ox+round(x0*iw/w),28+round(y0*ih/h),ox+round(x1*iw/w),28+round(y1*ih/h)]
            assert box[0]<box[2] and box[1]<box[3]
            ImageDraw.Draw(low).rectangle(box,outline=(255,255,0),width=2)
            enlarged=low.resize((1152,768),Image.Resampling.NEAREST)
            clean.save(out/f'f{frame}_unmarked.png')
            for arm,image in zip(ARMS,[low,enlarged]):
                path=out/f'f{frame}_{arm}.png';image.save(path)
                write(out/f'f{frame}_{arm}.job.json',dict(ID=f'CCD_000540_f{frame}',image=str(path),sha256=sha(path),prompt=protocol['prompt'],max_new_tokens=40))
            row.update(marker_source_box=ref['marker_box_xyxy'],marker_low_box=box,
                       low_rgb_sha256=old.prior.rgb_sha(low),upsampled_rgb_sha256=old.prior.rgb_sha(enlarged))
        rows.append(row)
    write(HERE/'input_checks.json',dict(status='built_pending_peer_QA',cases=rows))


def check_frozen():
    result=read(HERE/'freeze.json')
    for name,digest in result['files'].items():assert sha(ROOT/name)==digest,name
    return result


def freeze():
    assert not (HERE/'freeze.json').exists() and not (HERE/'run').exists()
    assert read(HERE/'input_peer_review.json')['status']=='PASS'
    assert read(HERE/'preflight_review.json')['status']=='PASS'
    for who in ['a','b']:
        ref=read(HERE/f'review_{who}.json')
        assert ref['new_predictions_seen'] is False and ref['peer_review_seen'] is False
    refs=read(HERE/'references.json')['cases']
    assert [r['frame'] for r in refs]==read(HERE/'protocol.json')['frames']==[31,35,39]
    files=dict(read(PREV/'freeze.json')['files'])
    for name,digest in files.items():assert sha(ROOT/name)==digest,name
    for path in [PREV/'freeze.json',PREV/'evaluation.json',PREV/'run/CCD_000540_baseline/result.json',PREV/'post_result_review.md']:
        files[str(path.relative_to(ROOT))]=sha(path)
    for path in HERE.rglob('*'):
        if path.is_file() and '__pycache__' not in path.parts and path.name!='STATUS.json':
            files[str(path.relative_to(ROOT))]=sha(path)
    write(HERE/'freeze.json',dict(status='locked_before_new_diagnostic_calls',created_utc=datetime.now(timezone.utc).isoformat(),
          model_calls_planned=2*sum(r['mark_approved'] for r in refs),files=files))


def run():
    assert platform.system()=='Darwin' and platform.machine()=='arm64'
    frozen=check_frozen();out=HERE/'run';out.mkdir(exist_ok=False)
    env=dict(os.environ,**old.prior.ENV,BLACKBOX_DEEPSTACK_FIX='1',BLACKBOX_COMPUTE_DTYPE='native',BLACKBOX_DECODE_MODE='sync')
    report=dict(status='running',started_utc=datetime.now(timezone.utc).isoformat(),platform=platform.platform(),freeze_sha256=sha(HERE/'freeze.json'),workers=[])
    write(out/'report.json',report)
    for ref in read(HERE/'references.json')['cases']:
        if not ref['mark_approved']:continue
        frame=ref['frame']
        for arm in ARMS:
            name=f'f{frame}_{arm}';start=time.perf_counter()
            with (out/f'{name}.log').open('x') as log:
                proc=subprocess.run([sys.executable,str(LEGEND/'experiment.py'),'state_worker','--job',str(HERE/'inputs'/f'{name}.job.json'),'--out',str(out/name)],env=env,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
            report['workers'].append(dict(frame=frame,arm=arm,exit_status=proc.returncode,wall_seconds=time.perf_counter()-start))
            print(json.dumps(report['workers'][-1]),flush=True)
            if proc.returncode:
                report['status']='failed';write(out/'report.json',report);raise SystemExit(proc.returncode)
            write(out/'report.json',report)
    report.update(status='complete',ended_utc=datetime.now(timezone.utc).isoformat());write(out/'report.json',report)


def evaluate():
    from PIL import Image
    check_frozen()
    refs=read(HERE/'references.json')['cases'];report=read(HERE/'run/report.json')
    expected={(r['frame'],arm) for r in refs if r['mark_approved'] for arm in ARMS}
    assert report['status']=='complete' and len(report['workers'])==len(expected)
    assert {(w['frame'],w['arm']) for w in report['workers']}==expected and all(w['exit_status']==0 for w in report['workers'])
    rows=[]
    for ref in refs:
        frame=ref['frame'];row=dict(frame=frame,reference=ref['state'],mark_approved=ref['mark_approved'],eligible=ref['mark_approved'] and ref['state']!='UNKNOWN',arms={})
        if ref['mark_approved']:
            for arm in ARMS:
                name=f'f{frame}_{arm}';folder=HERE/'run'/name
                result,worker,calls=[read(folder/x) for x in ['result.json','worker_report.json','calls.json']]
                job=read(HERE/'inputs'/f'{name}.job.json')
                assert result['ID']==worker['ID']==job['ID']==f'CCD_000540_f{frame}'
                assert worker['status']=='complete' and worker['model_calls']==len(calls)==1 and worker['network_attempts']==0
                call=calls[0];assert call==result['call'] and call['text'].strip()==result['raw'].strip()
                with Image.open(job['image']) as image:
                    assert call['image_sha256']==[old.prior.rgb_sha(old.prior.bounded(image))]
                    assert call['image_sizes']==[list(image.size)]
                assert call['prompt']==job['prompt']==read(HERE/'protocol.json')['prompt'] and call['max_new_tokens']==40
                try:obj=json.loads(result['raw'])
                except (ValueError,TypeError):obj=None
                value=obj.get('lane_state') if isinstance(obj,dict) else None
                valid=isinstance(value,str) and value in ['INSIDE','OUTSIDE','UNCERTAIN']
                state=value if valid else None
                assert result['state']==state and result['valid_enum']==valid
                row['arms'][arm]=dict(prediction=state,raw=result['raw'],valid_enum=valid,correct=(state==ref['state']) if row['eligible'] else None,prompt_tokens=call['prompt_tokens'])
        rows.append(row)
    eligible=[r for r in rows if r['eligible']];metrics={}
    classes=['OUTSIDE','INSIDE'];outputs=['OUTSIDE','INSIDE','UNCERTAIN',None]
    for arm in ARMS:
        n=len(eligible);correct=sum(r['arms'][arm]['correct'] for r in eligible)
        both={r['reference'] for r in eligible}==set(classes)
        queried=[r for r in rows if r['mark_approved']]
        sequence=[dict(frame=r['frame'],prediction=r['arms'][arm]['prediction']) for r in queried]
        metrics[arm]=dict(n=n,correct=correct,accuracy=correct/n if n else None,
             confusion=dict(reference_rows=classes,prediction_columns=outputs,matrix=[[sum(r['reference']==a and r['arms'][arm]['prediction']==b for r in eligible) for b in outputs] for a in classes]),
             distinguish_definite_outside_inside=(correct==n) if both else None,
             chronological_responses=sequence,
             inside_to_outside_adjacent_pairs=[[a['frame'],b['frame']] for a,b in zip(sequence,sequence[1:]) if a['prediction']=='INSIDE' and b['prediction']=='OUTSIDE'],
             uncertain_all_queried=sum(r['arms'][arm]['prediction']=='UNCERTAIN' for r in queried),invalid_all_queried=sum(not r['arms'][arm]['valid_enum'] for r in queried))
    gain=[r['frame'] for r in eligible if not r['arms']['low']['correct'] and r['arms']['upsampled']['correct']]
    loss=[r['frame'] for r in eligible if r['arms']['low']['correct'] and not r['arms']['upsampled']['correct']]
    write(HERE/'evaluation.json',dict(status='complete',actual_model_calls=len(expected),rows=rows,metrics=metrics,
          enlargement=dict(gained_frames=gain,lost_frames=loss,accuracy_delta=(len(gain)-len(loss))/len(eligible) if eligible else None),
          entry_output_modified=False,official_S2=None,scope='One exposed incident, separate wheel-state diagnostic. No temporal entry estimate or historical prompt accuracy comparison.'))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['prepare','freeze','run','evaluate']);globals()[parser.parse_args().action]()
