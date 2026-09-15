"""Explicit 8B NF4 export and fresh offline raw-answer replay; never adoption."""
import os,sys,json,hashlib,argparse,shutil,socket,importlib.util,time
from pathlib import Path
sys.dont_write_bytecode=True
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[key]='2'
os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
REPO='Qwen/Qwen3-VL-8B-Instruct';REV='0c351dd01ed87e9c1b53cbc748cba10e6187ff3b'
PLAN=HERE/'model_capacity_source_manifest.json'
PLAN_SHA='7630c131d8759237b38c31ef7966e3d12bbdb07f8040850b6a0bb78d63d4ba17'
CODE=ROOT/'artifacts/submissions/verify_v6/model/stage2/code'
CPU_POLICY=HERE/'solution/cpu_embedding.py'
NETWORK_ATTEMPTS=0

def require(ok,message):
    if not ok:raise ValueError(message)
def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def put(path,value):
    with Path(path).open('x',encoding='utf8') as f:json.dump(value,f,indent=2,ensure_ascii=False,allow_nan=False)
def member(root,name):
    require(isinstance(name,str) and name and '\\' not in name and ':' not in name,'Unsafe member')
    p=Path(name);require(not p.is_absolute() and '..' not in p.parts,'Unsafe member traversal')
    dest=(Path(root)/p).resolve();require(dest.is_relative_to(Path(root).resolve()),'Member escape');return dest
def new_output(path):
    p=Path(path).resolve();require(p.is_relative_to(HERE) and not p.exists(),'New output under research/v7 required');return p
def validate_identity(plan,download):
    require(plan.get('repository')==download.get('repository')==REPO,'Pinned official 8B repository required')
    require(plan.get('revision')==download.get('revision')==REV,'Pinned 8B revision required')
    require(plan.get('license')=='apache-2.0' and download.get('status')=='complete','Completed licensed acquisition required')
    selected={x['name']:x for x in plan['files'] if x['name']!='.gitattributes'}
    recorded={x['name']:x for x in download['files']}
    require(len(recorded)==len(download['files']) and set(selected)==set(recorded),'Acquisition file inventory differs')
    for name,item in selected.items():
        member(HERE,name);d=recorded[name]
        require(item['url']==f'https://huggingface.co/{REPO}/resolve/{REV}/{name}','Unpinned URL')
        require(d['bytes']==item['bytes'] and isinstance(d['sha256'],str) and len(d['sha256'])==64,'Source size/hash schema')
        if item.get('lfs_sha256'):require(d['sha256']==d.get('expected_lfs_sha256')==item['lfs_sha256'],'Shard LFS digest mismatch')
    return selected,recorded
def validate_quant(q):
    require(q.get('quant_method')=='bitsandbytes' and q.get('bnb_4bit_quant_type')=='nf4','NF4 required')
    require(q.get('load_in_4bit',q.get('_load_in_4bit')) is True,'4-bit required')
    require(q.get('bnb_4bit_compute_dtype')=='float16' and q.get('bnb_4bit_use_double_quant') is False,'FP16 without double quantization required')
def validate_config(c):
    t=c.get('text_config',{});require(c.get('model_type')=='qwen3_vl' and t.get('vocab_size')==151936 and t.get('hidden_size')==4096,'8B architecture mismatch')
    require(c.get('tie_word_embeddings') is False,'Untied 8B required')
def verify_source(path):
    path=Path(path).resolve();require(sha(PLAN)==PLAN_SHA,'Pinned source plan changed')
    plan=read(PLAN);download=read(path/'download_manifest.json');selected,recorded=validate_identity(plan,download)
    for name,item in selected.items():
        p=member(path,name);require(p.stat().st_size==item['bytes'] and sha(p)==recorded[name]['sha256'],'Source file changed: '+name)
        if not item.get('lfs_sha256'):
            payload=p.read_bytes();require(hashlib.sha1(b'blob '+str(len(payload)).encode()+b'\0'+payload).hexdigest()==item['git_blob_id'],'Pinned Git blob differs')
    validate_config(read(path/'config.json'))
    card=(path/'README.md').read_text('utf8');require(card.startswith('---') and 'license: apache-2.0' in card.split('---',2)[1],'Apache declaration absent')
    index=read(path/'model.safetensors.index.json');require(set(index['weight_map'].values())<=(set(recorded)),'Unverified shard referenced')
    return download
def verify_development(run,source):
    run=Path(run).resolve();report=read(run/'report.json');evaluation=read(run/'evaluation.json')
    freeze=read(run/'freeze.json');source_manifest=Path(source).resolve()/'download_manifest.json'
    require(freeze['files'].get(str(source_manifest))==sha(source_manifest),'Development source binding differs')
    runtime=report.get('runtime',{});require(runtime.get('precision')=='nf4' and runtime.get('compute_dtype')=='float16' and runtime.get('double_quant') is False,'Development runtime differs')
    require(report.get('status')=='complete' and evaluation.get('report_sha256')==sha(run/'report.json'),'Completed bound development report required')
    require({v['ID'] for v in report['videos']}=={'00000','00003','00004','00005','00006','00007','00008','00010','00013'} and len(report['videos'])==9,'Full exposed nine development sources required')
    scores=evaluation['scores'];b=scores['baseline'];c=scores['candidate'];improved=False
    for field,metric in [('contact','accuracy'),('entry','accuracy'),('side','macro_f1'),('space','macro_f1')]:
        require(b[field]['n']==c[field]['n'] and b[field]['n']>0,'Known-label denominator mismatch')
        x,y=b[field][metric],c[field][metric];require(type(x) in (int,float) and type(y) in (int,float) and 0<=x<=1 and 0<=y<=1,'Invalid metric')
        require(y==x if field=='contact' else y>=x,'Development nondecrease gate failed')
        if field!='contact':improved|=y>x
    require(improved,'No changed field improves; export blocked')
    require(all(v['baseline']['collision_frame']==v['candidate']['collision_frame'] for v in report['videos']),'Contact policy differs')
    return {'report':str(run/'report.json'),'report_sha256':sha(run/'report.json'),'freeze':str(run/'freeze.json'),'freeze_sha256':sha(run/'freeze.json'),'evaluation':str(run/'evaluation.json'),'evaluation_sha256':sha(run/'evaluation.json'),
            'scope':'Already exposed development gate only. Not independent accuracy or submission approval.'}
def reference_calls(run,ID,source):
    run=Path(run).resolve();report=read(run/'report.json');freeze=read(run/'freeze.json')
    require(report.get('status')=='complete','Reference run incomplete')
    require(freeze['files'].get(str(Path(source).resolve()/'download_manifest.json'))==sha(Path(source)/'download_manifest.json'),'Reference is not bound to same verified source')
    runtime=report.get('runtime',{});require(runtime.get('precision')=='nf4' and runtime.get('compute_dtype')=='float16' and runtime.get('double_quant') is False,'Reference runtime mismatch')
    rows=[v for v in report['videos'] if v['ID']==ID];require(len(rows)==1 and len(rows[0]['calls'])==4,'Exactly four reference calls required')
    result=[]
    for i,call in enumerate(rows[0]['calls'],1):
        folder=run/ID/f'call_{i}';record=read(folder/'record.json')
        require(call['status']==record['status']=='complete' and all(record[k]==call[k] for k in ('prompt','raw','max_new_tokens')),'Reference call record differs')
        require(isinstance(call['raw'],str) and isinstance(call['prompt'],str) and type(call['max_new_tokens']) is int and 0<call['max_new_tokens']<=128,'Invalid reference schema')
        images=sorted(folder.glob('input_*.png'),key=lambda p:int(p.stem.split('_')[-1]));require(bool(images),'Missing original call image')
        result.append({'prompt':call['prompt'],'max_new_tokens':call['max_new_tokens'],'raw':call['raw'],
            'images':[{'path':str(p.resolve()),'sha256':sha(p)} for p in images],
            'record_path':str(folder/'record.json'),'record_sha256':sha(folder/'record.json')})
    return result,{'report_sha256':sha(run/'report.json'),'freeze_sha256':sha(run/'freeze.json'),'ID':ID,'run':str(run)}
def offline():
    def deny(*a,**kw):
        global NETWORK_ATTEMPTS
        NETWORK_ATTEMPTS+=1
        raise RuntimeError('Network prohibited during export/reload')
    socket.socket.connect=deny;socket.socket.connect_ex=deny;socket.create_connection=deny
def load_class():
    sys.path.insert(0,str(CODE));from solution.vlm_candidate import CandidateVLM
    return CandidateVLM
def install_policy(candidate,policy):
    if policy=='all_gpu':require(candidate.model.get_input_embeddings().weight.device.type=='cuda','AllGPU embedding expected');return None
    require(policy=='cpu_fp16','Unknown embedding policy')
    spec=importlib.util.spec_from_file_location('nf4_8b_cpu_embedding',CPU_POLICY);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    result=m.install(candidate.model);candidate.torch.cuda.empty_cache();return result
def schema(model):return {k:{'shape':list(v.shape),'dtype':str(v.dtype)} for k,v in model.state_dict().items()}
def embedding_hash(model):
    tensor=model.get_input_embeddings().weight.detach().cpu().contiguous()
    return hashlib.sha256(memoryview(tensor.numpy()).cast('B')).hexdigest()
def replay(candidate,calls):
    from PIL import Image
    result=[]
    for call in calls:
        require(sha(call['record_path'])==call['record_sha256'],'Reference record changed')
        images=[]
        for row in call['images']:
            require(sha(row['path'])==row['sha256'],'Reference image changed')
            with Image.open(row['path']) as im:images.append(im.convert('RGB'))
        start=time.perf_counter();raw=candidate.ask(images,call['prompt'],max_new_tokens=call['max_new_tokens'])
        result.append({'raw':raw,'expected_raw':call['raw'],'equal':raw==call['raw'],'seconds':time.perf_counter()-start})
        require(raw==call['raw'],'Raw replay differs; no export/promotion')
    return result
def license_text(path):
    p=Path(path).resolve();text=p.read_text('utf8');require('Apache License' in text and 'Version 2.0, January 2004' in text and 'END OF TERMS AND CONDITIONS' in text,'Full Apache2 license text required');return p

def export_existing(candidate,target,source,download,gate,calls,ref,policy,policy_info,license_path,replayed):
    target=new_output(target);source=Path(source).resolve()
    require(NETWORK_ATTEMPTS==0,'Network was attempted before export')
    require(candidate.model_path.resolve()==source,'Instance source mismatch')
    require(candidate.candidate_metadata['precision']=='nf4','Instance is not NF4')
    validate_config(candidate.model.config.to_dict());q=candidate.model.config.quantization_config
    validate_quant(q.to_dict() if hasattr(q,'to_dict') else q)
    require(candidate.pixel_budget==1200000 and len(replayed)==4 and all(x['equal'] for x in replayed),'Full-budget same-instance raw replay required')
    emb=candidate.model.get_input_embeddings();require(str(emb.weight.dtype)=='torch.float16','Embedding dtype changed')
    require(emb.weight.device.type==('cpu' if policy=='cpu_fp16' else 'cuda'),'Embedding placement differs')
    before=schema(candidate.model);eh=embedding_hash(candidate.model);license_path=license_text(license_path)
    bindings={str(p):sha(p) for p in [Path(__file__),PLAN,CPU_POLICY,CODE/'solution/vlm.py',CODE/'solution/vlm_candidate.py',source/'download_manifest.json',license_path]}
    target.mkdir(parents=True);marker=target/'EXPORT_INCOMPLETE.json';put(marker,{'status':'saving','repository':REPO,'revision':REV})
    candidate.model.save_pretrained(str(target),safe_serialization=True,max_shard_size='2GB');candidate.processor.save_pretrained(str(target))
    validate_quant(read(target/'config.json')['quantization_config']);validate_config(read(target/'config.json'))
    require(schema(candidate.model)==before and embedding_hash(candidate.model)==eh,'Export changed live keys/dtype/embedding values')
    shutil.copyfile(license_path,target/'LICENSE-APACHE-2.0.txt');shutil.copyfile(source/'README.md',target/'README.original.md')
    shutil.copyfile(source/'download_manifest.json',target/'SOURCE_DOWNLOAD_MANIFEST.json')
    if (source/'NOTICE').is_file():shutil.copyfile(source/'NOTICE',target/'NOTICE')
    text=f'---\nlicense: apache-2.0\nbase_model: {REPO}\n---\n\nLocal NF4 derivative of {REPO} at {REV}.\n\nNF4/FP16 without double quantization; no fine-tuning. Processor reserialized. Not an official publisher checkpoint.\nEmbedding policy: {policy}. CPU subclass is NOT serialized: reinstall after every reload with identical FP16 values/state keys.\nExport is not reload verification or adoption. See LICENSE-APACHE-2.0.txt and README.original.md.\n'
    (target/'README.md').write_text(text,encoding='utf8');(target/'CHANGES.md').write_text(text,encoding='utf8')
    put(target/'RELOAD_REFERENCE.json',{'calls':calls,'reference':ref,'same_instance_replay':replayed})
    put(target/'STATE_SCHEMA.json',before)
    files=[{'name':p.relative_to(target).as_posix(),'bytes':p.stat().st_size,'sha256':sha(p)} for p in sorted(target.rglob('*')) if p.is_file() and p!=marker]
    manifest={'status':'saved_unverified','repository':REPO,'revision':REV,'source':f'https://huggingface.co/{REPO}/tree/{REV}','license':'Apache-2.0',
        'embedding_policy':policy,'embedding_policy_info':policy_info,'embedding_weight_sha256':eh,'pixel_budget':1200000,
        'export_pid':os.getpid(),'runtime':candidate.candidate_metadata,'bindings':bindings,'development_gate':gate,'files':files,
        'reload_equivalence_verified':False,'offline_reload_verified':False,'submission_approved':False}
    put(target/'EXPORT_MANIFEST.json',manifest);marker.unlink();return manifest

def verify_export(path):
    path=Path(path).resolve();require(not (path/'EXPORT_INCOMPLETE.json').exists(),'Incomplete export')
    m=read(path/'EXPORT_MANIFEST.json');require(m['repository']==REPO and m['revision']==REV and m['status']=='saved_unverified','Unexpected derivative')
    seen=set()
    for f in m['files']:
        p=member(path,f['name']);require(f['name'] not in seen,'Duplicate export file');seen.add(f['name'])
        require(p.stat().st_size==f['bytes'] and sha(p)==f['sha256'],'Export payload changed')
    require({p.relative_to(path).as_posix() for p in path.rglob('*') if p.is_file()}==seen|{'EXPORT_MANIFEST.json'},'Unmanifested export file')
    for p,h in m['bindings'].items():require(sha(p)==h,'Bound loader/policy/provenance changed')
    validate_config(read(path/'config.json'));validate_quant(read(path/'config.json')['quantization_config']);return m

def main():
    p=argparse.ArgumentParser();sub=p.add_subparsers(dest='mode',required=True)
    e=sub.add_parser('export');e.add_argument('--source',type=Path,required=True);e.add_argument('--target',type=Path,required=True)
    e.add_argument('--development-run',type=Path,required=True);e.add_argument('--reference-run',type=Path,required=True);e.add_argument('--reference-id',required=True)
    e.add_argument('--embedding-policy',choices=['all_gpu','cpu_fp16'],required=True);e.add_argument('--license-text',type=Path,required=True)
    r=sub.add_parser('reload');r.add_argument('--model',type=Path,required=True);r.add_argument('--output',type=Path,required=True)
    a=p.parse_args();require(sys.flags.isolated,'Use fresh isolated python -I -B');offline()
    if a.mode=='export':
        new_output(a.target);gate=verify_development(a.development_run,a.source);download=verify_source(a.source)
        calls,ref=reference_calls(a.reference_run,a.reference_id,a.source);license_text(a.license_text)
        Loader=load_class()
        with Loader(a.source,precision='nf4',pixel_budget=1200000) as candidate:
            candidate.torch.set_num_threads(2);info=install_policy(candidate,a.embedding_policy);replayed=replay(candidate,calls)
            export_existing(candidate,a.target,a.source,download,gate,calls,ref,a.embedding_policy,info,a.license_text,replayed)
    else:
        out=new_output(a.output);manifest=verify_export(a.model);require(manifest['export_pid']!=os.getpid(),'Fresh reload process required')
        ref=read(a.model/'RELOAD_REFERENCE.json');Loader=load_class();out.mkdir(parents=True)
        result={'status':'running','export_manifest_sha256':sha(a.model/'EXPORT_MANIFEST.json'),'model_loads':0,'accuracy_evaluation':False,'submission_approved':False}
        try:
            with Loader(a.model,precision='nf4',pixel_budget=manifest['pixel_budget']) as candidate:
                result['model_loads']=1;candidate.torch.set_num_threads(2);result['policy']=install_policy(candidate,manifest['embedding_policy'])
                require(schema(candidate.model)==read(a.model/'STATE_SCHEMA.json'),'Reload state keys/dtype/shape changed')
                require(embedding_hash(candidate.model)==manifest['embedding_weight_sha256'],'Reload embedding values changed')
                result['calls']=replay(candidate,ref['calls']);result['runtime']=candidate.candidate_metadata
                require(NETWORK_ATTEMPTS==0,'Network attempted during reload')
                result.update(status='passed',offline_raw_reload_verified=True,network_attempts=NETWORK_ATTEMPTS,scope='Four frozen prompts only; no accuracy or all-input guarantee')
        except BaseException as error:result.update(status='failed',error=repr(error));raise
        finally:put(out/'reload_report.json',result)

if __name__=='__main__':main()
