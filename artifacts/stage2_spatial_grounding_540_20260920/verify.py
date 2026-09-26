"""Independent two-call spatial-output audit. No model weights or inference."""
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sys

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PREV = ROOT / 'artifacts/stage2_wheel_detail_540_20260920'
spec = importlib.util.spec_from_file_location('previous_detail_audit', PREV / 'verify.py')
previous = importlib.util.module_from_spec(spec)
spec.loader.exec_module(previous)
read, sha, rgb_sha, bounded = previous.read, previous.sha, previous.rgb_sha, previous.bounded


def audit_inputs():
    from PIL import Image
    protocol = read(HERE / 'protocol.json')
    assert protocol['frames'] == [31,39] and protocol['actual_calls_planned'] == 2 and protocol['max_new_tokens'] == 768
    jobs = {}
    for frame in protocol['frames']:
        old = read(PREV / 'inputs' / f'f{frame}_source_detail.job.json')
        job = read(HERE / 'inputs' / f'f{frame}_spatial.job.json')
        assert job == {**old, 'prompt':protocol['prompt'], 'max_new_tokens':768}
        assert job['prompt'] != old['prompt'] and old['max_new_tokens'] == 40
        assert sha(Path(job['image'])) == job['sha256']
        with Image.open(job['image']) as image:
            assert image.size == (1152,768) and bounded(image).tobytes() == image.convert('RGB').tobytes()
        jobs[frame] = job
    return protocol, jobs


def schema_check(raw):
    """Basic independent JSON/schema validation; no coordinate repair or GT use."""
    def reject(value):
        raise ValueError(value)
    def unique(pairs):
        result = {}
        for key,value in pairs:
            if key in result:
                raise ValueError('duplicate key')
            result[key] = value
        return result
    try:
        obj = json.loads(raw, parse_constant=reject, object_pairs_hook=unique)
    except (ValueError,TypeError):
        return dict(valid_json=False,valid_schema=False,errors=['invalid JSON'],object=None)
    errors = []
    if not isinstance(obj,dict) or set(obj) != {'bbox_xyxy','wheels','boundary','inside_point_xy','lane_state'}:
        return dict(valid_json=True,valid_schema=False,errors=['top-level keys'],object=obj)
    def number(x):
        return type(x) in (int,float) and math.isfinite(x)
    def point(p):
        return isinstance(p,list) and len(p)==2 and all(number(v) for v in p) and 0<=p[0]<=1151 and 0<=p[1]<=767
    box = obj['bbox_xyxy']
    if box is not None and not (isinstance(box,list) and len(box)==4 and all(number(v) for v in box) and 0<=box[0]<box[2]<=1151 and 0<=box[1]<box[3]<=767):
        errors.append('bbox')
    wheels = obj['wheels']
    if not isinstance(wheels,list) or len(wheels)>4:
        errors.append('wheels')
    else:
        for wheel in wheels:
            if not isinstance(wheel,dict) or set(wheel)!={'xy','role','lane_relation'} or (wheel['xy'] is not None and not point(wheel['xy'])) or wheel['role'] not in ('front','rear','unknown') or wheel['lane_relation'] not in ('INSIDE','OUTSIDE','UNCERTAIN'):
                errors.append('wheel item')
    boundary = obj['boundary']
    if not isinstance(boundary,list) or len(boundary) not in (0,2,3,4,5,6):
        errors.append('boundary')
    elif boundary:
        valid_points = all(isinstance(p,dict) and set(p)=={'xy','kind'} and point(p['xy']) and p['kind'] in ('visible','extended') for p in boundary)
        if not valid_points:
            errors.append('boundary item')
        elif any(a['xy'][1]>=b['xy'][1] for a,b in zip(boundary,boundary[1:])):
            errors.append('boundary far-to-near ordering')
    if obj['inside_point_xy'] is not None and not point(obj['inside_point_xy']):
        errors.append('inside point')
    if obj['lane_state'] not in ('INSIDE','OUTSIDE','UNCERTAIN'):
        errors.append('lane state')
    return dict(valid_json=True,valid_schema=not errors,errors=errors,object=obj)


def main():
    report, evaluation = read(HERE / 'run/report.json'), read(HERE / 'evaluation.json')
    assert report['status'] == evaluation['status'] == 'complete'
    outputs = [HERE / ('independent_verification'+s) for s in ('.json','.md')]
    assert not any(p.exists() for p in outputs)
    frozen, old_files = read(HERE / 'freeze.json'), read(PREV / 'freeze.json')['files']
    assert frozen['status']=='locked_before_spatial_predictions' and frozen['model_calls_planned']==2
    assert len(old_files)==1866 and all(frozen['files'].get(k)==v for k,v in old_files.items())
    for name,digest in frozen['files'].items():
        assert sha(ROOT/name)==digest,name
    stamp = datetime.fromisoformat(frozen['created_utc'])
    assert stamp < datetime.fromisoformat(report['started_utc']) < datetime.fromisoformat(report['ended_utc'])
    assert report['freeze_sha256']==sha(HERE/'freeze.json') and read(HERE/'preflight_review.json')['status']=='PASS'
    protocol,jobs = audit_inputs()
    references=read(HERE/'references.json')
    assert [(r['frame'],r['mark_approved']) for r in references['cases']]==[(31,True),(39,True)]
    for meta in references['expert_files'].values():
        assert sha(HERE/meta['path'])==meta['sha256']
        review=read(HERE/meta['path'])
        assert review['new_predictions_seen'] is False
    assert [(w['frame'],w['arm']) for w in report['workers']]==[(31,'spatial'),(39,'spatial')]
    assert all(w['exit_status']==0 for w in report['workers'])
    old_ledger = read(PREV / 'independent_verification.json')['artifact_sha256']
    hashes = {str(p.relative_to(ROOT)):sha(p) for p in [Path(__file__),PREV/'verify.py',HERE/'freeze.json',HERE/'evaluation.json',HERE/'run/report.json']}
    records, rows, resources = {}, [], []
    from PIL import Image
    for frame,job in jobs.items():
        folder = HERE / 'run' / f'f{frame}_spatial'
        worker,result,calls = [read(folder/n) for n in ('worker_report.json','result.json','calls.json')]
        assert worker['ID']==result['ID']==job['ID'] and worker['arm']=='state' and worker['status']=='complete'
        assert worker['model_calls']==len(calls)==1 and worker['network_attempts']==0
        assert stamp < datetime.fromisoformat(worker['started_utc']) <= datetime.fromisoformat(worker['ended_utc'])
        call = calls[0]
        assert call==result['call'] and call['text'].strip()==result['raw'].strip()
        assert call['prompt']==protocol['prompt'] and call['max_new_tokens']==768 and call['image_sizes']==[[1152,768]]
        assert call['deepstack_fix'] is True and call['compute_dtype']=='native' and call['decode_mode']=='sync'
        with Image.open(job['image']) as im:
            assert call['image_sha256']==[rgb_sha(bounded(im))]
        assert 0<len(call['token_trace'])==call['generation_tokens']<=768 and call['peak_memory']>0
        history_path = PREV / 'run' / f'f{frame}_source_detail/result.json'
        assert sha(history_path)==old_ledger[str(history_path.relative_to(ROOT))]
        history = read(history_path)['call']
        assert call['image_sha256']==history['image_sha256'] and call['image_sizes']==history['image_sizes']
        for key in ('pixel_values','image_grid_thw'):
            assert call['processor_input_sha256'][key]==history['processor_input_sha256'][key]
        assert call['prompt']!=history['prompt'] and call['prompt_sha256']!=history['prompt_sha256']
        # State-only worker validity is recorded separately from full spatial schema.
        try:
            old_obj=json.loads(result['raw'])
        except (ValueError,TypeError):
            old_obj=None
        lane=old_obj.get('lane_state') if isinstance(old_obj,dict) else None
        enum_valid=isinstance(lane,str) and lane in ('INSIDE','OUTSIDE','UNCERTAIN')
        assert result['valid_enum']==enum_valid and result['state']==(lane if enum_valid else None)
        schema=schema_check(result['raw'])
        evaluated=next(r for r in evaluation['rows'] if r['frame']==frame)
        assert evaluated['raw']==result['raw'] and evaluated['strict_schema_valid']==schema['valid_schema']
        assert evaluated['parsed']==schema['object']
        rows.append(dict(frame=frame,raw=result['raw'],schema=schema,worker_lane_enum_valid=enum_valid,worker_lane_state=result['state'],generation_tokens=call['generation_tokens'],at_token_limit=call['generation_tokens']==768,last_token=call['token_trace'][-1]['token'],historical_pixels_grid_exact=True))
        records[frame]=(job,call)
        parent=next(w for w in report['workers'] if w['frame']==frame)
        assert parent['wall_seconds']>=worker['wall_seconds']>0
        resources.append(dict(frame=frame,parent_wall_seconds=parent['wall_seconds'],worker_wall_seconds=worker['wall_seconds'],mlx_peak_GB=call['peak_memory'],rss_bytes=worker['rss_bytes']))
        for name in ('worker_report.json','result.json','calls.json'):
            hashes[str((folder/name).relative_to(ROOT))]=sha(folder/name)

    import numpy as np
    os.environ.update(HF_HUB_OFFLINE='1',HF_HUB_DISABLE_IMPLICIT_TOKEN='1',TOKENIZERS_PARALLELISM='false')
    import socket
    attempts=[]
    def deny(*args,**kwargs):
        attempts.append(True)
        raise RuntimeError('Processor audit attempted network')
    socket.socket.connect=socket.socket.connect_ex=socket.create_connection=deny
    import mlx.core as mx
    mx.set_default_device(mx.cpu)
    import torch
    torch.set_num_threads(2)
    from mlx_vlm.utils import load_processor,load_image_processor,prepare_inputs
    model=ROOT/'artifacts/mac_experiments/stage2_mlx/model';cfg=read(model/'config.json')
    processor=load_processor(model,True,eos_token_ids=cfg.get('eos_token_id'),trust_remote_code=False,local_files_only=True)
    optional=load_image_processor(model,trust_remote_code=False,local_files_only=True)
    if optional is not None:
        processor.image_processor=optional
    tensors=[]
    for frame,(job,call) in records.items():
        with Image.open(job['image']) as image:
            image=bounded(image)
        text=processor.apply_chat_template([{'role':'user','content':[{'type':'image'},{'type':'text','text':job['prompt']}]}],tokenize=False,add_generation_prompt=True)
        inputs=prepare_inputs(processor,images=[image],prompts=text,image_token_index=cfg.get('image_token_index') or cfg['image_token_id'],add_special_tokens=True)
        mx.eval(*[v for v in inputs.values() if isinstance(v,mx.array)])
        arrays={k:np.asarray(v) for k,v in inputs.items() if isinstance(v,mx.array)}
        assert {k:hashlib.sha256(v.tobytes()).hexdigest() for k,v in arrays.items()}==call['processor_input_sha256']
        assert hashlib.sha256(text.encode()).hexdigest()==call['prompt_sha256']
        grid=arrays['image_grid_thw'].tolist();visual=int(np.sum(arrays['input_ids']==cfg['image_token_id']))
        assert grid==[[1,48,72]] and visual==864 and arrays['input_ids'].shape==arrays['attention_mask'].shape
        assert arrays['input_ids'].shape[-1]==call['prompt_tokens']
        tokens=[t['token'] for t in call['token_trace']]
        eos=cfg.get('eos_token_id') or cfg.get('text_config',{}).get('eos_token_id') or []
        eos=[eos] if type(eos) is int else eos
        tensors.append(dict(frame=frame,image_grid_thw=grid,image_tokens=visual,input_tokens=call['prompt_tokens'],nonimage_tokens=int(arrays['input_ids'].size)-visual,processor_hashes_exact=True,last_token_is_configured_eos=tokens[-1] in eos,configured_eos_ids=eos))
    assert str(mx.default_device())=='Device(cpu, 0)' and not attempts
    for name,digest in {**frozen['files'],**hashes}.items():
        assert sha(ROOT/name)==digest,name
    summary=dict(status='PASS',created_utc=datetime.now(timezone.utc).isoformat(),frozen_files_verified_pre_post=len(frozen['files']),previous_freeze_preserved=len(old_files),actual_model_calls_verified=2,verifier_model_loads=0,verifier_model_calls=0,verifier_device='Device(cpu, 0)',verifier_network_attempts=0,rows=rows,processor_metadata=tensors,resources=resources,artifact_sha256=hashes,
        scope='Runtime/input/strict basic spatial schema audit. Geometry reference scores require a separate read-only mathematical review.',
        limits=['One exposed incident, two frames and expert-agent references are not human GT or performance validation.','New structured question and768-token cap jointly differ from the old state question.','A valid lane_state alone does not imply a valid or accurate spatial response.','At-token-limit and configured-EOS flags are observations; JSON/schema completeness is evaluated separately.','No forced old INSIDE geometry, automatic coordinate repair, entry override or officialS2.'])
    outputs[0].write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
    lines=['# Spatial diagnostic independent runtime verification','','**PASS.** Two actual calls, frozen files and CPU processor reconstructions verified without model inference.','',f"Frozen files pre/post: {len(frozen['files'])}; prior1,866 preserved. Existing source_detail RGB/pixel_values/image_grid preserved; new question and768-token cap retained.",'','| Frame | JSON | Spatial schema | Worker lane enum | Generated tokens | Token cap reached |','|---|---|---|---|---:|---|']
    lines += [f"| {r['frame']} | {r['schema']['valid_json']} | {r['schema']['valid_schema']} | {r['worker_lane_state']} | {r['generation_tokens']} | {r['at_token_limit']} |" for r in rows]
    lines += ['',summary['scope'],'','## Limits','']+['- '+s for s in summary['limits']]
    outputs[1].write_text('\n'.join(lines)+'\n')
    print(json.dumps({k:summary[k] for k in ('status','frozen_files_verified_pre_post','actual_model_calls_verified','processor_metadata','resources')},ensure_ascii=False))


if __name__=='__main__':
    main()
