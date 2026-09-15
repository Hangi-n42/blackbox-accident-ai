"""Four frozen real prompts before/after CPU lookup. Not an accuracy trial."""
import os,sys,json,hashlib,socket,time,importlib.util
from pathlib import Path
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[k]='2'
os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
ROOT=Path(__file__).resolve().parents[2];HERE=Path(__file__).resolve().parent;OUT=HERE/'embedding_equivalence_4b'
def put(path,v):
    with path.open('x',encoding='utf8') as f:json.dump(v,f,ensure_ascii=False,indent=2)
def sha(path):
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def deny(*a,**k):raise RuntimeError('Offline only')
def main():
    assert sys.flags.isolated and not OUT.exists();OUT.mkdir()
    trace=ROOT/'research/v6_stage2/v5_fullframe_diagnostic/traces/00000'
    rows=[json.loads((trace/f'call_{i}/record.json').read_text('utf8')) for i in range(1,5)]
    bind={str(p):sha(p) for i in range(1,5) for p in (trace/f'call_{i}').glob('*') if p.is_file()}
    bind[str(HERE/'solution/cpu_embedding.py')]=sha(HERE/'solution/cpu_embedding.py')
    put(OUT/'plan.json',dict(scope='Same frozen4B weights/images/prompts; allGPU then CPU input lookup;8 calls; no GT evaluation, no export',files=bind,tied_4b_diagnostic_only=True))
    socket.socket.connect=deny;socket.create_connection=deny
    sys.path.insert(0,str(ROOT/'artifacts/submissions/verify_v6/model/stage2/code'))
    from solution.vlm_candidate import CandidateVLM
    from PIL import Image
    spec=importlib.util.spec_from_file_location('cpu_embedding_v7',HERE/'solution/cpu_embedding.py');cpu=importlib.util.module_from_spec(spec);spec.loader.exec_module(cpu)
    report=dict(status='running',variants={})
    try:
        with CandidateVLM(ROOT/'artifacts/submissions/verify_v6/model/stage2/vlm',precision='nf4') as model:
            model.torch.set_num_threads(2)
            for variant in ('all_gpu','cpu_lookup'):
                if variant=='cpu_lookup':report['policy']=cpu.install(model.model,allow_tied_diagnostic=True);model.torch.cuda.empty_cache()
                answers=[]
                for i,row in enumerate(rows,1):
                    images=[Image.open(p).convert('RGB') for p in sorted((trace/f'call_{i}').glob('input_*.png'))]
                    assert images
                    start=time.perf_counter();raw=model.ask(images,row['prompt'],max_new_tokens=row['max_new_tokens'])
                    answers.append(dict(call=i,raw=raw,seconds=time.perf_counter()-start));print(json.dumps(dict(variant=variant,call=i)),flush=True)
                report['variants'][variant]=answers
        equal=[a['raw']==b['raw'] for a,b in zip(report['variants']['all_gpu'],report['variants']['cpu_lookup'])]
        report.update(status='PASS' if all(equal) else 'FAIL',exact_raw_equal=equal,
            limitation='Four real prompts with4B only;8B inference and export/reload still unverified. Tied4B test does not claim memory savings.')
        assert all(sha(Path(p))==h for p,h in bind.items())
    except BaseException as e:report.update(status='failed',error=repr(e));raise
    finally:put(OUT/'report.json',report)
if __name__=='__main__':main()
