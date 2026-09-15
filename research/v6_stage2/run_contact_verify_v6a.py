"""V6A controlled experiment. Inference never reads user labels.

freeze development|reserved --output NEW; run --output FROZEN
Reserved freeze requires a passing development gate written by CPU evaluation.
"""
from __future__ import annotations
import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import socket
import sys
import time

sys.dont_write_bytecode = True
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key] = '2'
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
PACKAGE = ROOT/'artifacts/submissions/verify_v5'
CANDIDATE = HERE/'candidates/stage2_contact_verify_v6a.py'
PROTOCOL = HERE/'contact_verify_v6a_protocol.json'

def read(p): return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):
    with Path(p).open('rb') as f: return hashlib.file_digest(f,'sha256').hexdigest()
def put(p,x):
    with Path(p).open('x',encoding='utf-8') as f: json.dump(x,f,ensure_ascii=False,indent=2,allow_nan=False)
def require(x,m):
    if not x: raise ValueError(m)
def load_module(name,p):
    spec=importlib.util.spec_from_file_location(name,p)
    mod=importlib.util.module_from_spec(spec);sys.modules[name]=mod;spec.loader.exec_module(mod);return mod

def freeze(phase,output):
    require(not output.exists(),'Refuse overwrite')
    baseline_runner=load_module('bound_baseline',HERE/'run_nexar_baseline.py')
    files={str(p):sha(p) for p in (Path(__file__),CANDIDATE,PROTOCOL)}
    binding=baseline_runner.package_binding()
    records=[]
    if phase=='development':
        runs=[(HERE/'nexar_baseline_run',('00000','00003')),(HERE/'nexar_dev00004_baseline_run',('00004',))]
        for directory,ids in runs:
            frozen=read(directory/'freeze.json');report=read(directory/'report.json')
            require(report['status']=='complete' and report['network_attempts']==0,'Incomplete baseline')
            require(report['call_count']==4*len(ids),'Baseline call contract')
            require(frozen['binding']['files']==binding['files'],'Cached baseline uses different code or weights')
            for p in (directory/'freeze.json',directory/'report.json'):files[str(p)]=sha(p)
            require(report['freeze_sha256']==sha(directory/'freeze.json'),'Baseline freeze binding')
            require(tuple(x['ID'] for x in report['videos'])==ids,'Unexpected baseline IDs')
            for video,trace in zip(frozen['videos'],report['videos']):
                require(video['ID']==trace['ID'],'Baseline order')
                require(video['input_manifest_sha256']==trace['input_manifest_sha256'],'Input binding')
                records.append(dict(ID=video['ID'],input_root=str(directory),input=video,baseline_trace=trace))
    else:
        dev=HERE/'contact_verify_v6a_development'
        gate=read(dev/'evaluation.json');oldfreeze=verify(dev);devreport=read(dev/'report.json')
        require(gate['gate_passed'] is True,'Development gate not passed')
        require(gate['report_sha256']==sha(dev/'report.json'),'Development result changed')
        require(devreport['status']=='complete' and devreport['phase']=='development','Invalid development result')
        require(devreport['call_count']==3 and devreport['network_attempts']==0,'Development call/network contract')
        require([v['ID'] for v in devreport['videos']]==['00000','00003','00004'],'Development cohort mismatch')
        require(devreport['freeze_sha256']==sha(dev/'freeze.json'),'Development freeze mismatch')
        require(gate['freeze_sha256']==sha(dev/'freeze.json') and gate['protocol_sha256']==sha(PROTOCOL),'Gate protocol mismatch')
        require(gate['phase']=='development','Not a development gate')
        require(sha(gate['evaluator_path'])==gate['evaluator_sha256'],'Evaluator changed after gate')
        files[gate['evaluator_path']]=gate['evaluator_sha256']
        for p,digest in gate['review_file_sha256s'].items():
            require(sha(p)==digest,'Human reference changed after gate');files[p]=digest
        require(oldfreeze['package_binding']['files']==binding['files'],'Package changed after development')
        for p in (CANDIDATE,PROTOCOL,Path(__file__)):
            require(oldfreeze['files'][str(p)]==sha(p),'Candidate changed after development')
        for p in (dev/'evaluation.json',dev/'freeze.json',dev/'report.json'):files[str(p)]=sha(p)
        directory=HERE/'nexar_reserved_inputs';frozen=read(directory/'freeze.json')
        files[str(directory/'freeze.json')]=sha(directory/'freeze.json')
        require(tuple(x['ID'] for x in frozen['videos'])==('00005','00006','00007'),'Reserved split changed')
        records=[dict(ID=v['ID'],input_root=str(directory),input=v) for v in frozen['videos']]
    for r in records:
        v=r['input']; require(sha(v['source_path'])==v['source_sha256'],'Source changed')
        files[v['source_path']]=v['source_sha256']
        for item in v['input_images']:
            p=Path(r['input_root'])/item['path'];require(sha(p)==item['file_sha256'],'Input changed')
    output.mkdir()
    put(output/'freeze.json',dict(created_utc=datetime.now(timezone.utc).isoformat(),phase=phase,
        files=files,package_binding=binding,videos=records,gt_read=False,max_calls=3 if phase=='development' else 15))
    print(json.dumps(dict(status='frozen',phase=phase,ids=[r['ID'] for r in records],sha256=sha(output/'freeze.json'))))

def verify(output):
    frozen=read(output/'freeze.json')
    for p,digest in frozen['files'].items():require(sha(p)==digest,f'Bound file changed: {p}')
    for p,digest in frozen['package_binding']['files'].items():require(sha(PACKAGE/p)==digest,f'Package changed: {p}')
    for record in frozen['videos']:
        for item in record['input']['input_images']:
            require(sha(Path(record['input_root'])/item['path'])==item['file_sha256'],'Bound PNG changed')
    return frozen

class Recorder:
    def __init__(self,model,folder,report):self.model=model;self.folder=folder;self.report=report;self.calls=[]
    def ask(self,images,prompt,max_new_tokens=128):
        require(self.report['call_count']<self.report['max_calls'],'Call budget')
        idx=len(self.calls)+1;folder=self.folder/f'call_{idx}';folder.mkdir()
        for i,img in enumerate(images):img.save(folder/f'input_{i}.png')
        row=dict(prompt=prompt,max_new_tokens=max_new_tokens,status='pending')
        self.report['call_count']+=1;self.calls.append(row)
        self.model.torch.cuda.synchronize();start=time.perf_counter()
        try:
            answer=self.model.ask(images,prompt,max_new_tokens=max_new_tokens)
            self.model.torch.cuda.synchronize();row.update(status='complete',raw=answer)
            return answer
        except BaseException as e:row.update(status='failed',error=repr(e));raise
        finally:
            row['seconds']=time.perf_counter()-start;put(folder/'record.json',row)
            print(json.dumps(dict(folder=str(self.folder),call=idx,**row)),flush=True)

def run(output):
    require(sys.flags.isolated,'Use -I')
    frozen=verify(output)
    require(not (output/'report.json').exists() and not (output/'traces').exists(),'No overwrite/retry')
    report=dict(status='running',phase=frozen['phase'],freeze_sha256=sha(output/'freeze.json'),
        max_calls=frozen['max_calls'],call_count=0,network_attempts=0,videos=[])
    def deny(*a,**kw):report['network_attempts']+=1;raise RuntimeError('Network blocked')
    socket.socket.connect=deny;socket.socket.connect_ex=deny;socket.create_connection=deny
    sys.path.insert(0,str(PACKAGE/'model/stage2/code'))
    try:
        import cv2
        cv2.setNumThreads(2)
        candidate=load_module('solution.stage2_contact_verify_v6a',CANDIDATE)
        baseline=candidate.baseline
        with baseline.CandidateVLM(PACKAGE/'model/stage2/vlm',precision='nf4') as model:
            model.torch.set_num_threads(2);report['runtime']=model.candidate_metadata
            for r in frozen['videos']:
                v=r['input'];paths=[Path(r['input_root'])/x['path'] for x in v['input_images']]
                folder=output/'traces'/r['ID'];folder.mkdir(parents=True);rec=Recorder(model,folder,report)
                if frozen['phase']=='development':
                    old=r['baseline_trace'];by_num={baseline.base._frame_number(p):p for p in paths}
                    paths=[by_num[n] for n in old['frame_numbers']]
                    pred,diag=deepcopy(old['prediction']),deepcopy(old['diagnostics'])
                    scan_seconds=None
                else:
                    t=time.perf_counter();paths,scores,_=baseline.base._motion_scan(paths);scan_seconds=time.perf_counter()-t
                    pred,diag=baseline._predict_file(paths,scores,rec)
                before=deepcopy(pred);after,newdiag=candidate.refine_collision(paths,pred,diag,rec)
                require(all(before[k]==after[k] for k in ('entry_frame','entry_side','evasion_space')),'Noncollision regression')
                require(len(rec.calls)==(1 if frozen['phase']=='development' else 5),'Per-file budget')
                row=dict(ID=r['ID'],baseline=before,candidate=after,baseline_diagnostics=diag,
                    diagnostics=newdiag,calls=rec.calls,scan_seconds=scan_seconds,
                    frame_numbers=[baseline.base._frame_number(p) for p in paths])
                put(folder/'trace.json',row);report['videos'].append(row)
        require(report['call_count']==report['max_calls'] and report['network_attempts']==0,'Budget/offline')
        verify(output);report['status']='complete'
    except BaseException as e:report.update(status='failed',error=repr(e));raise
    finally:put(output/'report.json',report)
    print(json.dumps(dict(status=report['status'],phase=report['phase'],calls=report['call_count'])))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['freeze','run']);p.add_argument('--phase',choices=['development','reserved']);p.add_argument('--output',required=True,type=Path);a=p.parse_args()
    if a.command=='freeze':require(a.phase,'Phase required');freeze(a.phase,a.output.resolve())
    else:run(a.output.resolve())
