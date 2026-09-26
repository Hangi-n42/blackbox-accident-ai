"""Fixed equal-size native-detail comparison, reusing the frozen Mac state runner."""
import argparse
from datetime import datetime,timezone
import importlib.util
from pathlib import Path
import shutil

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
PREV=ROOT/'artifacts/stage2_wheel_state_540_20260920'
spec=importlib.util.spec_from_file_location('wheel_runner',PREV/'experiment.py')
wheel=importlib.util.module_from_spec(spec);spec.loader.exec_module(wheel)
spec=importlib.util.spec_from_file_location('detail_renderer',ROOT/'artifacts/stage2_state_resolution_20260920/experiment.py')
render=importlib.util.module_from_spec(spec);spec.loader.exec_module(render)
read,write,sha=wheel.read,wheel.write,wheel.sha
ARMS=['upsampled','source_detail']


def prepare():
    import numpy as np
    from PIL import Image
    out=HERE/'inputs';out.mkdir(exist_ok=False)
    assert sha(HERE/'references.json')==sha(PREV/'references.json')
    protocol=read(HERE/'protocol.json');rows=[]
    for old_meta in read(PREV/'input_checks.json')['cases']:
        f=old_meta['frame'];source=ROOT/old_meta['source']['path']
        assert sha(source)==old_meta['source']['sha256']
        with Image.open(source) as im:native=im.convert('RGB')
        with Image.open(PREV/'inputs'/f'f{f}_low.png') as im:low=im.convert('RGB')
        up,high,mask=render.rebuild_high(native,low,old_meta['marker_low_box'])
        old_job=read(PREV/'inputs'/f'f{f}_upsampled.job.json')
        assert old_job['prompt']==protocol['prompt'] and old_job['max_new_tokens']==40
        control=out/f'f{f}_upsampled.png';shutil.copyfile(old_job['image'],control)
        assert sha(control)==old_job['sha256']
        with Image.open(control) as im:assert im.convert('RGB').tobytes()==up.tobytes()
        high.save(out/f'f{f}_source_detail.png');mask.save(out/f'f{f}_marker_mask.png')
        changed=np.any(np.asarray(up)!=np.asarray(high),axis=2);m=np.asarray(mask,dtype=bool)
        scene=np.zeros(changed.shape,dtype=bool);scene[84:732,:]=True
        assert changed.any() and not np.any(changed & (~scene | m))
        assert np.array_equal(np.asarray(up)[m],np.asarray(high)[m])
        for arm in ARMS:
            path=out/f'f{f}_{arm}.png'
            write(out/f'f{f}_{arm}.job.json',dict(ID=old_job['ID'],image=str(path),sha256=sha(path),prompt=old_job['prompt'],max_new_tokens=40))
        rows.append(dict(frame=f,source=old_meta['source'],marker_low_box=old_meta['marker_low_box'],source_size=list(native.size),
                         canvas=[1152,768],scene_rectangle=[0,84,1152,732],source_detail_size=[1152,648],
                         upsampled_rgb_sha256=wheel.old.prior.rgb_sha(up),source_detail_rgb_sha256=wheel.old.prior.rgb_sha(high),
                         source_detail_changed_pixels=int(changed.sum()),marker_pixels=int(m.sum()),control_file_bytes_exact=True))
    write(HERE/'input_checks.json',dict(status='built_pending_peer_QA',cases=rows))


def freeze():
    assert not (HERE/'freeze.json').exists() and not (HERE/'run').exists()
    assert sha(HERE/'references.json')==sha(PREV/'references.json')
    assert read(HERE/'input_peer_review.json')['status']=='PASS'
    assert read(HERE/'preflight_review.json')['status']=='PASS'
    files=dict(read(PREV/'freeze.json')['files'])
    for name,digest in files.items():assert sha(ROOT/name)==digest,name
    for path in [PREV/'freeze.json',PREV/'evaluation.json',PREV/'run/report.json',PREV/'independent_verification.json']:
        files[str(path.relative_to(ROOT))]=sha(path)
    for f in read(HERE/'protocol.json')['frames']:
        for name in ['result.json','calls.json','worker_report.json']:
            path=PREV/'run'/f'f{f}_upsampled'/name;files[str(path.relative_to(ROOT))]=sha(path)
    for path in HERE.rglob('*'):
        if path.is_file() and '__pycache__' not in path.parts and path.name!='STATUS.json':files[str(path.relative_to(ROOT))]=sha(path)
    write(HERE/'freeze.json',dict(status='locked_before_equal_size_detail_predictions',created_utc=datetime.now(timezone.utc).isoformat(),model_calls_planned=6,files=files))


def run():
    # Reuse the exact fresh-worker loop and state_worker; bind only this run's directory and arm names.
    wheel.HERE=HERE;wheel.ARMS=ARMS
    wheel.run()


def evaluate():
    import json
    from PIL import Image
    frozen=read(HERE/'freeze.json')
    for name,digest in frozen['files'].items():assert sha(ROOT/name)==digest,name
    report=read(HERE/'run/report.json');refs=read(HERE/'references.json')['cases'];protocol=read(HERE/'protocol.json')
    expected={(f,a) for f in protocol['frames'] for a in ARMS}
    assert report['status']=='complete' and len(report['workers'])==len(expected)==6
    assert {(w['frame'],w['arm']) for w in report['workers']}==expected and all(w['exit_status']==0 for w in report['workers'])
    rows=[]
    for ref in refs:
        f=ref['frame'];row=dict(frame=f,reference=ref['state'],eligible=ref['state']!='UNKNOWN',arms={});records={}
        for arm in ARMS:
            folder=HERE/'run'/f'f{f}_{arm}'
            result,worker,calls=[read(folder/n) for n in ['result.json','worker_report.json','calls.json']]
            job=read(HERE/'inputs'/f'f{f}_{arm}.job.json')
            assert result['ID']==worker['ID']==job['ID']==f'CCD_000540_f{f}'
            assert worker['status']=='complete' and worker['model_calls']==len(calls)==1 and worker['network_attempts']==0
            call=calls[0];assert call==result['call'] and call['text'].strip()==result['raw'].strip()
            with Image.open(job['image']) as im:
                assert call['image_sha256']==[wheel.old.prior.rgb_sha(wheel.old.prior.bounded(im))]
                assert call['image_sizes']==[list(im.size)]==[[1152,768]]
            assert call['prompt']==job['prompt']==protocol['prompt'] and call['max_new_tokens']==40
            try:obj=json.loads(result['raw'])
            except (ValueError,TypeError):obj=None
            value=obj.get('lane_state') if isinstance(obj,dict) else None
            valid=isinstance(value,str) and value in ['OUTSIDE','INSIDE','UNCERTAIN'];state=value if valid else None
            assert result['state']==state and result['valid_enum']==valid
            row['arms'][arm]=dict(prediction=state,raw=result['raw'],valid_enum=valid,correct=(state==ref['state']) if row['eligible'] else None,prompt_tokens=call['prompt_tokens'])
            records[arm]=result
        a,b=[records[x] for x in ARMS]
        history=read(PREV/'run'/f'f{f}_upsampled/result.json')
        row['historical_control_raw_enum_match']=(a['raw'].strip()==history['raw'].strip() and a['state']==history['state'])
        assert a['call']['processor_input_sha256']==history['call']['processor_input_sha256']
        row['historical_control_processor_exact']=True
        for key in ['input_ids','attention_mask','image_grid_thw']:
            assert a['call']['processor_input_sha256'][key]==b['call']['processor_input_sha256'][key]
        assert a['call']['prompt_tokens']==b['call']['prompt_tokens'] and a['call']['processor_input_sha256']['pixel_values']!=b['call']['processor_input_sha256']['pixel_values']
        row['same_text_grid_tokens_different_pixels']=True;rows.append(row)
    eligible=[r for r in rows if r['eligible']];n=len(eligible);metrics={}
    for arm in ARMS:
        correct=sum(r['arms'][arm]['correct'] for r in eligible)
        metrics[arm]=dict(n=n,correct=correct,accuracy=correct/n if n else None,
            distinguish_definite_outside_inside=(correct==n) if {r['reference'] for r in eligible}=={'OUTSIDE','INSIDE'} else None,
            chronological_responses=[dict(frame=r['frame'],prediction=r['arms'][arm]['prediction']) for r in rows],
            uncertain=sum(r['arms'][arm]['prediction']=='UNCERTAIN' for r in rows),invalid=sum(not r['arms'][arm]['valid_enum'] for r in rows))
    gains=[r['frame'] for r in eligible if not r['arms']['upsampled']['correct'] and r['arms']['source_detail']['correct']]
    losses=[r['frame'] for r in eligible if r['arms']['upsampled']['correct'] and not r['arms']['source_detail']['correct']]
    gate=dict(gains=len(gains),losses=len(losses),historical_control_match=all(r['historical_control_raw_enum_match'] for r in rows),equal_text_grid_tokens=all(r['same_text_grid_tokens_different_pixels'] for r in rows))
    gate['pass']=gate['gains']>=1 and gate['losses']==0 and gate['historical_control_match'] and gate['equal_text_grid_tokens']
    write(HERE/'evaluation.json',dict(status='complete',actual_model_calls=6,rows=rows,metrics=metrics,
        source_detail_comparison=dict(gained_frames=gains,lost_frames=losses,accuracy_delta=(len(gains)-len(losses))/n if n else None),gate=gate,
        entry_output_modified=False,official_S2=None,scope='Equal-size source-detail diagnostic on one exposed incident; source detail/blur/aliasing not individually separated.'))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','freeze','run','evaluate']);globals()[p.parse_args().action]()
