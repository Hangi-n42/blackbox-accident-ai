"""Frozen first-frame state comparison: low, low enlarged, source-detail high."""
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
LEGEND=ROOT/'artifacts/stage2_entry_legend_20260920'
spec=importlib.util.spec_from_file_location('legend_experiment',LEGEND/'experiment.py')
old=importlib.util.module_from_spec(spec)
spec.loader.exec_module(old)
read,write,sha=old.read,old.write,old.sha
ARMS=['low','upsampled','high']


def rebuild_high(source,low,box):
    from PIL import Image,ImageDraw
    up=low.resize((1152,768),Image.Resampling.NEAREST)
    high=up.copy()
    high.paste(source.resize((1152,648),Image.Resampling.BICUBIC),(0,84))
    mask=Image.new('1',(384,256))
    ImageDraw.Draw(mask).rectangle(box,outline=1,width=2)
    mask=mask.resize((1152,768),Image.Resampling.NEAREST)
    high.paste((255,255,0),(0,0,1152,768),mask)
    return up,high,mask


def audit_inputs():
    import numpy as np
    from PIL import Image
    protocol=read(HERE/'protocol.json')
    metadata={r['ID']:r for r in read(HERE/'input_checks.json')['cases']}
    for sid in protocol['cases']:
        ref=read(LEGEND/'inputs'/f'{sid}_state.job.json')
        meta=metadata[sid]
        source=Path(meta['source_frame']['path'])
        assert sha(source)==meta['source_frame']['sha256']
        with Image.open(ref['image']) as im:low=im.convert('RGB')
        with Image.open(source) as im:source_image=im.convert('RGB')
        assert low.size==(384,256) and source_image.size==(1280,720)
        up,high,mask=rebuild_high(source_image,low,meta['marker_low_box'])
        for arm,expected in zip(ARMS,[low,up,high]):
            job=read(HERE/'inputs'/f'{sid}_{arm}.job.json')
            assert set(job)=={'ID','image','sha256','prompt','max_new_tokens'}
            assert job['ID']==sid and job['prompt']==ref['prompt'] and job['max_new_tokens']==ref['max_new_tokens']==40
            assert sha(Path(job['image']))==job['sha256']
            with Image.open(job['image']) as image:
                assert image.size==expected.size and old.prior.rgb_sha(image)==old.prior.rgb_sha(expected)
                assert old.prior.rgb_sha(old.prior.bounded(image))==old.prior.rgb_sha(image)
        with Image.open(HERE/'inputs'/f'{sid}_marker_mask.png') as saved:
            assert np.array_equal(np.asarray(saved),np.asarray(mask))
        a,b,m=np.asarray(up),np.asarray(high),np.asarray(mask,dtype=bool)
        changed=np.any(a!=b,axis=2)
        scene=np.zeros(changed.shape,dtype=bool)
        scene[84:732,:]=True
        assert changed.any() and not np.any(changed & (~scene | m))
        assert np.array_equal(a[m],b[m]) and np.all(b[m]==[255,255,0])


def freeze():
    assert not (HERE/'freeze.json').exists() and not (HERE/'run').exists()
    files=dict(read(LEGEND/'freeze.json')['files'])
    for name,digest in files.items():assert sha(ROOT/name)==digest,name
    audit_inputs()
    reviewed=[]
    for who in ['a','b']:
        review=read(HERE/f'visual_review_{who}.json')
        assert review['status']=='PASS' and review['new_predictions_seen'] is False
        reviewed.extend(r['ID'] for r in review['cases'])
    ids=read(HERE/'protocol.json')['cases']
    assert sorted(reviewed)==sorted(ids)
    for path in [LEGEND/'freeze.json',LEGEND/'state_evaluation.json',LEGEND/'state_run/report.json']:
        files[str(path.relative_to(ROOT))]=sha(path)
    for sid in ids:
        for name in ['result.json','calls.json','worker_report.json']:
            path=LEGEND/'state_run'/f'{sid}_state'/name
            files[str(path.relative_to(ROOT))]=sha(path)
    for path in HERE.rglob('*'):
        if path.is_file() and path.name!='STATUS.json':files[str(path.relative_to(ROOT))]=sha(path)
    write(HERE/'freeze.json',dict(status='locked_before_resolution_predictions',created_utc=datetime.now(timezone.utc).isoformat(),files=files))


def check_frozen():
    frozen=read(HERE/'freeze.json')
    for name,digest in frozen['files'].items():assert sha(ROOT/name)==digest,name


def run():
    assert platform.system()=='Darwin' and platform.machine()=='arm64'
    check_frozen()
    out=HERE/'run'
    out.mkdir(exist_ok=False)
    env=dict(os.environ,**old.prior.ENV,BLACKBOX_DEEPSTACK_FIX='1',BLACKBOX_COMPUTE_DTYPE='native',BLACKBOX_DECODE_MODE='sync')
    report=dict(status='running',started_utc=datetime.now(timezone.utc).isoformat(),platform=platform.platform(),
                freeze_sha256=sha(HERE/'freeze.json'),workers=[])
    write(out/'report.json',report)
    for sid in read(HERE/'protocol.json')['cases']:
        for arm in ARMS:
            name=f'{sid}_{arm}'
            start=time.perf_counter()
            with (out/f'{name}.log').open('x') as log:
                proc=subprocess.run([sys.executable,str(LEGEND/'experiment.py'),'state_worker',
                    '--job',str(HERE/'inputs'/f'{name}.job.json'),'--out',str(out/name)],
                    env=env,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
            report['workers'].append(dict(ID=sid,arm=arm,exit_status=proc.returncode,wall_seconds=time.perf_counter()-start))
            if proc.returncode:
                report['status']='failed'
                write(out/'report.json',report)
                raise SystemExit(proc.returncode)
            write(out/'report.json',report)
    report.update(status='complete',ended_utc=datetime.now(timezone.utc).isoformat())
    write(out/'report.json',report)


def evaluate():
    from PIL import Image
    check_frozen()
    audit_inputs()
    ids=read(HERE/'protocol.json')['cases']
    report=read(HERE/'run/report.json')
    expected={(sid,arm) for sid in ids for arm in ARMS}
    assert report['status']=='complete' and len(report['workers'])==len(expected)
    assert {(w['ID'],w['arm']) for w in report['workers']}==expected
    assert all(w['exit_status']==0 for w in report['workers'])
    refs={r['ID']:r for who in ['a','b'] for r in read(LEGEND/f'state_reference_{who}.json')['cases']}
    rows=[]
    calls_count=0
    for sid in ids:
        target=refs[sid]['state']
        row=dict(ID=sid,reference=target,eligible=target!='UNKNOWN',arms={})
        records={}
        for arm in ARMS:
            folder=HERE/'run'/f'{sid}_{arm}'
            wr,calls,result=(read(folder/name) for name in ['worker_report.json','calls.json','result.json'])
            assert wr['status']=='complete' and wr['model_calls']==len(calls)==1 and wr['network_attempts']==0
            assert wr['ID']==result['ID']==sid and wr['arm']=='state'
            assert calls[0]==result['call'] and calls[0]['text'].strip()==result['raw'].strip()
            job=read(HERE/'inputs'/f'{sid}_{arm}.job.json')
            with Image.open(job['image']) as image:
                assert calls[0]['image_sha256']==[old.prior.rgb_sha(old.prior.bounded(image))]
                assert calls[0]['image_sizes']==[list(image.size)]
            assert calls[0]['prompt']==job['prompt'] and calls[0]['max_new_tokens']==job['max_new_tokens']
            try:obj=json.loads(result['raw'])
            except (ValueError,TypeError):obj=None
            value=obj.get('lane_state') if isinstance(obj,dict) else None
            valid=isinstance(value,str) and value in ['INSIDE','OUTSIDE','UNCERTAIN']
            assert result['valid_enum']==valid and result['state']==(value if valid else None)
            row['arms'][arm]=dict(prediction=result['state'],valid_enum=valid,raw=result['raw'],
                correct=(valid and value==target) if target!='UNKNOWN' else None,prompt_tokens=calls[0]['prompt_tokens'],
                image_sizes=calls[0]['image_sizes'])
            records[arm]=result
            calls_count+=len(calls)
        history=read(LEGEND/'state_run'/f'{sid}_state/result.json')
        assert records['low']['call']['processor_input_sha256']==history['call']['processor_input_sha256']
        row['historical_low_raw_and_enum_match']=(records['low']['raw'].strip()==history['raw'].strip() and records['low']['state']==history['state'])
        for key in ['input_ids','attention_mask','image_grid_thw']:
            assert records['high']['call']['processor_input_sha256'][key]==records['upsampled']['call']['processor_input_sha256'][key]
        assert records['high']['call']['processor_input_sha256']['pixel_values']!=records['upsampled']['call']['processor_input_sha256']['pixel_values']
        row['high_vs_upsampled_same_text_and_grid']=True
        rows.append(row)
    eligible=[r for r in rows if r['eligible']]
    classes=['INSIDE','OUTSIDE']
    predicted=['INSIDE','OUTSIDE','UNCERTAIN',None]
    metrics={}
    for arm in ARMS:
        matrix=[[sum(r['reference']==t and r['arms'][arm]['prediction']==p for r in eligible) for p in predicted] for t in classes]
        metrics[arm]=dict(n=len(eligible),correct=sum(r['arms'][arm]['correct'] for r in eligible),
            accuracy=sum(r['arms'][arm]['correct'] for r in eligible)/len(eligible) if eligible else None,
            confusion=dict(reference_rows=classes,prediction_columns=predicted,matrix=matrix),
            uncertain=sum(r['arms'][arm]['prediction']=='UNCERTAIN' for r in eligible),invalid=sum(not r['arms'][arm]['valid_enum'] for r in eligible))
    comparisons={}
    for control,treatment in [('low','high'),('low','upsampled'),('upsampled','high')]:
        gain=[r['ID'] for r in eligible if not r['arms'][control]['correct'] and r['arms'][treatment]['correct']]
        loss=[r['ID'] for r in eligible if r['arms'][control]['correct'] and not r['arms'][treatment]['correct']]
        comparisons[f'{control}_to_{treatment}']=dict(gained_ids=gain,lost_ids=loss,accuracy_delta=(len(gain)-len(loss))/len(eligible) if eligible else None)
    primary=comparisons['low_to_high']
    gate=dict(gains=len(primary['gained_ids']),losses=len(primary['lost_ids']),
              inside_gains=sum(r['reference']=='INSIDE' and r['ID'] in primary['gained_ids'] for r in rows),
              historical_control_match=all(r['historical_low_raw_and_enum_match'] for r in rows))
    gate['pass']=gate['gains']>=1 and gate['inside_gains']>=1 and gate['losses']==0 and gate['historical_control_match']
    write(HERE/'evaluation.json',dict(status='complete',actual_model_calls=calls_count,rows=rows,metrics=metrics,
          comparisons=comparisons,gate=gate,official_S2=None,scope='Exposed first-frame state diagnostic, not temporal localization or automatic submission policy'))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('action',choices=['audit_inputs','freeze','run','evaluate'])
    args=parser.parse_args()
    globals()[args.action]()
