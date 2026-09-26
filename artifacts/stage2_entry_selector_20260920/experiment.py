"""Mac-only fixed-candidate training pilot. No production edits or VLM text generation."""
import argparse,hashlib,importlib.util,importlib.metadata,json,os,platform,sys,time
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
OLD=ROOT/'artifacts/stage2_entry_path_audit_20260920'
sys.path.insert(0,str(ROOT/'artifacts/submissions/verify_v6/model/stage2/code'))
from solution import stage2_v2 as v2
from selector import pool_tiles,fit,choose,certain_pairs,PCA_DIM,L2
read=lambda p:json.loads(p.read_text())
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def write(p,o):p.write_text(json.dumps(o,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def now():return datetime.now(timezone.utc).isoformat()
def environment(names):return dict(python=sys.version,executable=sys.executable,platform=platform.platform(),machine=platform.machine(),packages={n:importlib.metadata.version(n) for n in names})
def load_old(name):
    spec=importlib.util.spec_from_file_location(name,OLD/(name+'.py'));m=importlib.util.module_from_spec(spec);sys.path.insert(0,str(OLD));spec.loader.exec_module(m);return m

def prepare():
    out=HERE/'inputs';out.mkdir(exist_ok=False)
    native=read(ROOT/'artifacts/stage2_goal_20260919/core_v2/evaluation_manifest.json')['cases']
    ccd={r['ID']:r for r in read(OLD/'inventory.json')['cases']};traces={r['ID']:r for r in read(OLD/'path_replay.json')['rows']};jobs=[]
    human=read(ROOT/'artifacts/mac_experiments/baseline_20260916/human_labels.json')
    selected={r['ID'] for r in human if r['draft']['review']['entry']['status']=='observed'}
    for row in native:
        if row['ID'] not in selected:continue
        sid=row['ID'];folder=ROOT/'artifacts/stage2_goal_20260919/current_baseline/run/stage2/human_dev'/sid
        result=read(folder/'result.json');call=read(folder/'calls.json')[2];frames=result['diagnostics']['entry_candidates'];source=row['images'];numbers=[r['frame'] for r in source]
        paths=[ROOT/r['path'] for r in source];image=v2._sheet(paths,[numbers.index(f) for f in frames],columns=4);image.save(out/f'{sid}.png')
        assert [hashlib.sha256(image.tobytes()).hexdigest()]==call['image_sha256']
        jobs.append(dict(ID=sid,role='train_pool',image=str((out/f'{sid}.png').relative_to(ROOT)),image_sha256=sha(out/f'{sid}.png'),frames=frames,times=[source[numbers.index(f)]['pts_seconds'] for f in frames],source_images=[source[numbers.index(f)] for f in frames],history_call=str((folder/'calls.json').relative_to(ROOT)),baseline=result['baseline_prediction'],human_draft=row['review_source'],source_sha256=row['source_sha256'],source_group_status=row['source_group_audit'],full_case_source=row['time_mapping_source']))
    for ref in read(OLD/'references.json')['cases']:
        sid=ref['ID'];source=ccd[sid]['source']['images'];trace=traces[sid];frames=trace['entry_candidates'];numbers=[r['frame'] for r in source];image=OLD/'q3_inputs'/f'{sid}.png'
        jobs.append(dict(ID=sid,role='evaluation',image=str(image.relative_to(ROOT)),image_sha256=sha(image),frames=frames,times=[source[numbers.index(f)]['pts_seconds'] for f in frames],source_images=[source[numbers.index(f)] for f in frames],history_call=str(Path(ccd[sid]['baseline_folder'])/'calls.json'),baseline=trace['baseline'],source_sha256=ccd[sid]['source']['source_sha256'],source_group=ccd[sid]['source']['source_group'],evaluation_reference=ref['entry']))
    assert len(jobs)==16 and all(len(j['frames'])==12 for j in jobs)
    write(HERE/'inputs.json',dict(created_utc=now(),jobs=jobs,train_pool=8,evaluation=8,new_text_generations=0))
    write(HERE/'training_inputs.json',dict(jobs=[j for j in jobs if j['role']=='train_pool']))
    write(HERE/'evaluation_inputs.json',dict(jobs=[j for j in jobs if j['role']=='evaluation']))

def freeze():
    assert not (HERE/'freeze.json').exists() and not (HERE/'features').exists()
    assert read(HERE/'preflight_review.json')['status']=='PASS'
    files=dict(read(OLD/'freeze.json')['files'])
    for name,digest in files.items():assert sha(ROOT/name)==digest,name
    for job in read(HERE/'inputs.json')['jobs']:
        names=[job['image'],job['history_call']]
        names +=[r['path'] for r in job['source_images']]
        if job['role']=='train_pool':names +=[job['human_draft'],job['full_case_source']]
        for name in names:files[name]=sha(ROOT/name)
    for review in read(HERE/'training_reference_review.json')['cases']:
        for name in review['viewed_files']:files[name]=sha(ROOT/name)
    for p in HERE.rglob('*'):
        if p.is_file() and not any(x in p.parts for x in ['__pycache__','training_review_partial_attempt1']) and p.name not in ['STATUS.json','verify.py']:files[str(p.relative_to(ROOT))]=sha(p)
    for name,digest in files.items():assert sha(ROOT/name)==digest,name
    write(HERE/'freeze.json',dict(created_utc=now(),status='locked_before_feature_extraction_and_training',files=files))

def check_freeze():
    frozen=read(HERE/'freeze.json')
    for name,digest in frozen['files'].items():assert sha(ROOT/name)==digest,name
    return frozen

def feature_inputs(vlm,job):
    from PIL import Image
    from mlx_vlm.utils import prepare_inputs
    history=read(ROOT/job['history_call'])[2]
    with Image.open(ROOT/job['image']) as im:image=im.convert('RGB')
    assert image.size==(1536,768) and [hashlib.sha256(image.tobytes()).hexdigest()]==history['image_sha256']
    text=vlm.processor.apply_chat_template([{'role':'user','content':[{'type':'image'},{'type':'text','text':history['prompt']}]}],tokenize=False,add_generation_prompt=True)
    inputs=prepare_inputs(vlm.processor,images=[image],prompts=text,image_token_index=vlm.model.config.image_token_index,add_special_tokens=True)
    mx=vlm.mx;mx.eval(*[v for v in inputs.values() if isinstance(v,mx.array)])
    hashes={k:hashlib.sha256(np.asarray(v).tobytes()).hexdigest() for k,v in inputs.items() if isinstance(v,mx.array)}
    assert hashes==history['processor_input_sha256'] and hashlib.sha256(text.encode()).hexdigest()==history['prompt_sha256']
    assert np.asarray(inputs['image_grid_thw']).tolist()==[[1,48,96]]
    return inputs,hashes

def encode(vlm,job,save_hidden=False):
    inputs,hashes=feature_inputs(vlm,job);mx=vlm.mx;mx.synchronize();start=time.perf_counter()
    hidden,_=vlm.model.vision_tower(inputs['pixel_values'].astype(vlm.model.vision_tower.patch_embed.proj.weight.dtype),inputs['image_grid_thw'])
    mx.eval(hidden);mx.synchronize();hidden=np.asarray(hidden.astype(mx.float32));pooled=pool_tiles(hidden)
    assert hidden.shape==(1152,2560) and pooled.shape==(12,2560) and np.isfinite(pooled).all()
    if save_hidden:np.save(HERE/'features'/(job['ID']+'.hidden.npy'),hidden,allow_pickle=False)
    return pooled,dict(ID=job['ID'],role=job['role'],processor_hashes=hashes,hidden_shape=list(hidden.shape),hidden_sha256=hashlib.sha256(hidden.tobytes()).hexdigest(),pooled_sha256=hashlib.sha256(pooled.tobytes()).hexdigest(),seconds=time.perf_counter()-start,mlx_peak_GB=mx.get_peak_memory()/1e9)

def extract():
    assert platform.system()=='Darwin' and platform.machine()=='arm64';check_freeze()
    out=HERE/'features';out.mkdir(exist_ok=False)
    import socket
    attempts=[]
    def deny(*a,**kw):attempts.append(True);raise RuntimeError('Offline feature extraction attempted network')
    socket.socket.connect=socket.socket.connect_ex=socket.create_connection=deny
    os.environ.update(HF_HUB_OFFLINE='1',HF_HUB_DISABLE_IMPLICIT_TOKEN='1',TOKENIZERS_PARALLELISM='false',BLACKBOX_DEEPSTACK_FIX='1',BLACKBOX_COMPUTE_DTYPE='native',BLACKBOX_DECODE_MODE='sync')
    sys.path.insert(0,str(ROOT/'scripts/mac'))
    from mlx_stage2 import MLXVLM
    start=time.perf_counter();vlm=MLXVLM(ROOT/'artifacts/mac_experiments/stage2_mlx/model',out)
    jobs=read(HERE/'inputs.json')['jobs'];report=dict(started_utc=now(),status='running',rows=[],text_generations=0,environment=environment(['numpy','mlx','mlx-vlm','transformers']))
    first=None
    for job in jobs:
        pooled,row=encode(vlm,job,save_hidden=True);np.save(out/(job['ID']+'.npy'),pooled,allow_pickle=False)
        if first is None:first=pooled.copy()
        report['rows'].append(row);write(out/'report.json',report);print(job['ID'],row['seconds'],flush=True)
    repeated,repeat_record=encode(vlm,jobs[0]);np.testing.assert_array_equal(first,repeated)
    assert not attempts and not vlm.calls
    report.update(status='complete',ended_utc=now(),vision_forwards=17,repeat_first_exact=True,repeat_record=repeat_record,network_attempts=len(attempts),model_load_seconds=vlm.load_seconds,wall_seconds=time.perf_counter()-start)
    write(out/'report.json',report);check_freeze()

def train():
    check_freeze();assert read(HERE/'features/report.json')['status']=='complete'
    assert not (HERE/'model.npz').exists();refs=read(HERE/'training_references.json')['cases'];jobs={j['ID']:j for j in read(HERE/'training_inputs.json')['jobs']}
    selected=[r for r in refs if r['eligible']];ids=[r['ID'] for r in selected]
    assert len(ids)>=4 and len({r['entry_status'] for r in selected})==2,'Pilot needs >=4 incidents and both entry types; no forced recovery'
    assert all(jobs[s]['role']=='train_pool' for s in ids)
    embeddings=np.stack([np.load(HERE/'features'/f'{s}.npy',allow_pickle=False) for s in ids]);times=[jobs[s]['times'] for s in ids];ranges=[r['reference_seconds'] for r in selected]
    started=now();t=time.perf_counter();state,log=fit(embeddings,times,ranges);np.savez(HERE/'model.npz',**state)
    scores,indices=choose(embeddings,times,state)
    log.update(environment=environment(['numpy','scipy']),status='complete',started_utc=started,ended_utc=now(),wall_seconds=time.perf_counter()-t,real_training_runs=1,training_ids=ids,excluded_ids=[r['ID'] for r in refs if not r['eligible']],evaluation_examples_used_by_fit=0,model_sha256=sha(HERE/'model.npz'),train_selected_frames={s:jobs[s]['frames'][i] for s,i in zip(ids,indices)})
    position_losses=[float(np.mean([max(abs(t[i]-lo),abs(t[i]-hi)) for t,(lo,hi) in zip(times,ranges)])) for i in range(12)]
    log['constant_position_baseline']={'index':int(np.argmin(position_losses)),'mean_worst_mae_by_index':position_losses,'fit_scope':'training references only; constant index minimizes training worst-case MAE'}
    write(HERE/'training_report.json',log);print(json.dumps(log,ensure_ascii=False));check_freeze()

def evaluate():
    check_freeze();assert not (HERE/'evaluation.json').exists();state=dict(np.load(HERE/'model.npz',allow_pickle=False));old=load_old('evaluate');jobs=read(HERE/'evaluation_inputs.json')['jobs']
    em=np.stack([np.load(HERE/'features'/f'{j["ID"]}.npy',allow_pickle=False) for j in jobs]);scores,indices=choose(em,[j['times'] for j in jobs],state);traces={r['ID']:r for r in read(OLD/'path_replay.json')['rows']};rows=[]
    for j,score,i in zip(jobs,scores,indices):
        sid=j['ID'];base=j['baseline'];candidate={**base,'entry_frame':j['frames'][i]};ref=j['evaluation_reference'];times=traces[sid]['original_times'];lo,hi=[times[str(ref[k])] for k in ['lower_frame','upper_frame']];a,b=times[str(base['entry_frame'])],times[str(candidate['entry_frame'])]
        timing=dict(baseline=old.grade(a,lo,hi),candidate=old.grade(b,lo,hi),**old.paired(a,b,lo,hi),reference_seconds=[lo,hi])
        rows.append(dict(ID=sid,baseline=base,candidate=candidate,reference=ref,timing=timing,scores=score.tolist(),changed=base['entry_frame']!=candidate['entry_frame'],new_false_first=ref['status']=='during_clip' and candidate['entry_frame']==j['frames'][0],other_three_unchanged=all(base[k]==candidate[k] for k in ['collision_frame','entry_side','evasion_space'])))
    metrics={arm:old.aggregate([r['timing'] for r in rows],arm) for arm in ['baseline','candidate']}
    delta={k:[sum(r['timing'][k][i] for r in rows)/len(rows) for i in [0,1]] for k in ['accuracy_delta','mae_delta_seconds']}
    gate=dict(gain_ids=[r['ID'] for r in rows if r['timing']['baseline']['result']=='wrong' and r['timing']['candidate']['result']=='correct'],possible_loss_ids=[r['ID'] for r in rows if r['timing']['accuracy_delta'][0]<0],new_false_first_ids=[r['ID'] for r in rows if r['new_false_first']],other_three_preserved=all(r['other_three_unchanged'] for r in rows))
    gate['pass']=bool(gate['gain_ids']) and not gate['possible_loss_ids'] and not gate['new_false_first_ids'] and delta['mae_delta_seconds'][1]<=0 and gate['other_three_preserved']
    constant=read(HERE/'training_report.json')['constant_position_baseline']['index'];shortcuts={}
    for name,index in [('always_first',0),('training_best_constant_position',constant)]:
        details=[]
        for j,r in zip(jobs,rows):
            lo,hi=r['timing']['reference_seconds'];g=old.grade(j['times'][index],lo,hi)
            details.append({'ID':j['ID'],'entry_frame':j['frames'][index],'grade':g})
        shortcuts[name]={'index':index,'rows':details,'metrics':old.aggregate([{'candidate':d['grade']} for d in details],'candidate')}
    output=dict(status='complete',shortcut_controls=shortcuts,model_sha256=sha(HERE/'model.npz'),rows=rows,metrics=metrics,paired_delta_bounds=delta,gate=gate,official_S2=None,scope='One fit on eligible reviewed Nexar human drafts. CCD8 exposed AI references excluded from fit; source-separated development comparison, not independent human validation. No CUDA or counterpart identity accuracy claim.')
    write(HERE/'evaluation.json',output);print(json.dumps({k:output[k] for k in ['metrics','paired_delta_bounds','gate']},ensure_ascii=False));check_freeze()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','freeze','extract','train','evaluate']);globals()[p.parse_args().action]()
