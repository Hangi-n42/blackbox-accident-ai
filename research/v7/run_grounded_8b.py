"""Fourth fixed cell: official8B and unchanged unified grounding, no GT in inference."""
import sys, os, json, hashlib, argparse, importlib.util, socket, time
from pathlib import Path
from datetime import datetime, timezone
sys.dont_write_bytecode = True
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key] = '2'
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PACKAGE = ROOT/'artifacts/submissions/verify_v6'

def read(p): return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):
    with Path(p).open('rb') as f: return hashlib.file_digest(f,'sha256').hexdigest()
def put(p,v):
    with Path(p).open('x',encoding='utf8') as f: json.dump(v,f,ensure_ascii=False,indent=2,allow_nan=False)
def require(ok,msg):
    if not ok: raise ValueError(msg)
def load(name,p):
    spec=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

def corpus():
    sources=[('v5_fullframe_diagnostic/freeze.json','uncapped_jerk_fullframe_replay/report.json'),
             ('round2_validation/run/freeze.json','round2_validation/run/report.json')]
    rows=[];bindings={}
    stage=ROOT/'research/v6_stage2'
    for fp,rp in sources:
        fp,rp=stage/fp,stage/rp
        f,r=read(fp),read(rp)
        require(r['status']=='complete','Unfinished V6 reference')
        bindings.update({str(fp):sha(fp),str(rp):sha(rp)})
        by={v['ID']:v for v in r['videos']}
        for record in f['videos']:
            trace=by[record['ID']]
            require(trace['frame_numbers']==[x['frame'] for x in record['input']['input_images']],'Cached original frame mismatch')
            rows.append(dict(ID=record['ID'],record=record,trace=trace))
    require(len(rows)==9 and len({x['ID'] for x in rows})==9,'Expected nine exposed development sources')
    return rows,bindings

class Recorder:
    def __init__(self,model,folder): self.model=model;self.folder=folder;self.calls=[]
    def ask(self,images,prompt,max_new_tokens=128):
        require(len(self.calls)<4,'Four-call development budget exceeded')
        folder=self.folder/f'call_{len(self.calls)+1}';folder.mkdir()
        for i,im in enumerate(images): im.save(folder/f'input_{i}.png')
        row=dict(prompt=prompt,max_new_tokens=max_new_tokens,input_sizes=[list(im.size) for im in images],status='running')
        self.calls.append(row);start=time.perf_counter()
        try:
            raw=self.model.ask(images,prompt,max_new_tokens=max_new_tokens)
            row.update(status='complete',raw=raw);return raw
        except BaseException as error: row.update(status='failed',error=type(error).__name__);raise
        finally:
            row['seconds']=time.perf_counter()-start;put(folder/'record.json',row)
            print(json.dumps(dict(ID=self.folder.name,call=len(self.calls),status=row['status'])),flush=True)

def main():
    p=argparse.ArgumentParser();p.add_argument('--candidate',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--model',type=Path,required=True)
    p.add_argument('--ids',nargs='+',default=['00000','00003','00004','00005','00006','00007','00008','00010','00013'])
    a=p.parse_args();out=a.output.resolve();candidate=a.candidate.resolve()
    require(sys.flags.isolated and out.is_relative_to(ROOT) and not out.exists(),'Isolated Python and new workspace output required')
    rows,bindings=corpus();rows=[r for r in rows if r['ID'] in a.ids]
    require(len(rows)==len(set(a.ids))==len(a.ids),'Unknown/duplicate IDs')
    manifest=read(ROOT/'artifacts/submissions/submit_v6.manifest.json')
    for item in manifest['files']:
        path=PACKAGE/item['path'];require(sha(path)==item['sha256'],'V6 package changed');bindings[str(path)]=item['sha256']
    bindings[str(candidate)]=sha(candidate);bindings[str(Path(__file__))]=sha(__file__)
    protocol=HERE/'grounded_8b_protocol.json';bindings[str(protocol)]=sha(protocol)
    model_path=a.model.resolve();download=read(model_path/'download_manifest.json')
    require(download['status']=='complete' and download['repository']=='Qwen/Qwen3-VL-8B-Instruct','Verified official8B assets required')
    for item in download['files']:
        path=model_path/item['name'];require(sha(path)==item['sha256'],'8B source asset changed');bindings[str(path)]=item['sha256']
    bindings[str(model_path/'download_manifest.json')]=sha(model_path/'download_manifest.json')
    for row in rows:
        rec=row['record'];source=Path(rec['input']['source_path'])
        require(sha(source)==rec['input']['source_sha256'],'Original source changed');bindings[str(source)]=sha(source)
        for item in rec['input']['input_images']:
            image=Path(rec['input_root'])/item['path']
            require(sha(image)==item['file_sha256'],'Frozen original PNG changed');bindings[str(image)]=item['file_sha256']
    out.mkdir(parents=True)
    put(out/'freeze.json',dict(created_utc=datetime.now(timezone.utc).isoformat(),candidate=str(candidate),
        files=bindings,ids=a.ids,source_role='All nine sources already exposed in V6; development only, not independent validation',
        input_policy='Exact frozen full-native PNG and cached V6 motion feature arrays',human_answers_read_by_inference=False))
    report=dict(status='running',network_attempts=0,videos=[],model_loads=0,source_role='exposed_development')
    def deny(*args,**kwargs): report['network_attempts']+=1;raise RuntimeError('Offline run attempted network')
    socket.socket.connect=deny;socket.socket.connect_ex=deny;socket.create_connection=deny
    try:
        import numpy as np
        code=PACKAGE/'model/stage2/code';sys.path.insert(0,str(code))
        module=load('solution.stage2_v7_grounded',candidate)
        from solution.vlm_candidate import CandidateVLM
        from solution import stage2_uncapped_jerk_v6c as v6
        v6.primitives.cv2.setNumThreads(2)
        with CandidateVLM(model_path,precision='nf4') as model:
            report['model_loads']=1;model.torch.set_num_threads(2)
            report['runtime']=model.candidate_metadata
            for row in rows:
                rec,trace=row['record'],row['trace'];ID=row['ID']
                paths=[Path(rec['input_root'])/x['path'] for x in rec['input']['input_images']]
                features=np.asarray(trace['features'],dtype=np.float32)
                base_scores,new_scores=v6._scores_from_features(features)
                require(np.array_equal(base_scores,np.asarray(trace['base_scores'],dtype=np.float32)) and
                        np.array_equal(new_scores,np.asarray(trace['new_scores'],dtype=np.float32)),'Cached score recomputation differs')
                folder=out/ID;folder.mkdir();recorder=Recorder(model,folder);start=time.perf_counter()
                prediction,diagnostics=module.predict_from_scan(paths,base_scores,new_scores,recorder)
                require(len(recorder.calls)==4,'Expected four calls')
                require(set(prediction)=={'collision_frame','entry_frame','entry_side','evasion_space'},'Output schema')
                for k in ('collision_frame','entry_frame'):
                    require(type(prediction[k]) is int and prediction[k] in trace['frame_numbers'],'Original output number')
                require(prediction['entry_side'] in ('LEFT','RIGHT') and type(prediction['evasion_space']) is int and prediction['evasion_space'] in (0,1),'Output category')
                result=dict(ID=ID,baseline=trace['candidate'],candidate=prediction,diagnostics=diagnostics,
                    calls=recorder.calls,seconds=time.perf_counter()-start,source_sha256=rec['input']['source_sha256'],
                    frame_pts=rec['input']['source_frame_pts'],peak_allocated_bytes=model.torch.cuda.max_memory_allocated(),peak_reserved_bytes=model.torch.cuda.max_memory_reserved())
                put(folder/'result.json',result);report['videos'].append(result)
                print(json.dumps(dict(ID=ID,status='complete')),flush=True)
        for path,h in bindings.items(): require(sha(path)==h,'Inputs/code changed during run')
        require(report['network_attempts']==0,'Offline contract')
        report['status']='complete'
    except BaseException as error: report.update(status='failed',error=type(error).__name__);raise
    finally: put(out/'report.json',report)

if __name__=='__main__':main()
