"""Single score-component intervention; development CPU, reserved four VLM calls."""
import argparse
from contextlib import nullcontext
from copy import deepcopy
from datetime import datetime,timezone
import importlib.util
import json
import os
from pathlib import Path
import socket
import sys
import time

sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('uncapped_shared',HERE/'run_contact_verify_v6a.py')
shared=importlib.util.module_from_spec(spec);sys.modules[spec.name]=shared;spec.loader.exec_module(shared)
read,sha,put,require=shared.read,shared.sha,shared.put,shared.require
PACKAGE=shared.PACKAGE
CANDIDATE=HERE/'candidates/stage2_uncapped_jerk_v6c.py'
PROTOCOL=HERE/'uncapped_jerk_v6c_protocol.json'
DEV=HERE/'uncapped_jerk_v6c_development'

def freeze(phase,output):
    require(not output.exists(),'No overwrite')
    control=shared.verify(HERE/'contact_verify_v6a_development')
    files={str(p):sha(p) for p in (Path(__file__),CANDIDATE,PROTOCOL,HERE/'run_contact_verify_v6a.py')}
    if phase=='development':
        videos=control['videos']
        for p,d in control['files'].items():files[p]=d
        # Preserve A protocol as control provenance, but bind this score experiment too.
        files[str(PROTOCOL)]=sha(PROTOCOL)
    else:
        old=verify(DEV);gate=read(DEV/'evaluation.json');report=read(DEV/'report.json')
        require(gate['gate_passed'] is True and gate['phase']=='development','Development did not pass')
        require(gate['report_sha256']==sha(DEV/'report.json') and gate['freeze_sha256']==sha(DEV/'freeze.json'),'Gate binding changed')
        require(report['status']=='complete' and report['call_count']==0 and report['network_attempts']==0,'Invalid development run')
        require(report['freeze_sha256']==sha(DEV/'freeze.json'),'Development freeze mismatch')
        for p in (Path(__file__),CANDIDATE,PROTOCOL):require(old['files'][str(p)]==sha(p),'Candidate changed')
        require(gate['protocol_sha256']==sha(PROTOCOL) and sha(gate['evaluator_path'])==gate['evaluator_sha256'],'Evaluator/protocol changed')
        files[gate['evaluator_path']]=gate['evaluator_sha256']
        for p,d in gate['review_file_sha256s'].items():require(sha(p)==d,'Reference changed');files[p]=d
        for p in (DEV/'freeze.json',DEV/'report.json',DEV/'evaluation.json'):files[str(p)]=sha(p)
        source=HERE/'nexar_reserved_inputs';inputs=read(source/'freeze.json')
        require([v['ID'] for v in inputs['videos']]==['00005','00006','00007'],'Reserved split mismatch')
        require(inputs['binding']['files']==control['package_binding']['files'],'Reserved package mismatch')
        files[str(source/'freeze.json')]=sha(source/'freeze.json')
        videos=[dict(ID=v['ID'],input_root=str(source),input=v) for v in inputs['videos']]
    for row in videos:
        v=row['input'];require(sha(v['source_path'])==v['source_sha256'],'Source changed');files[v['source_path']]=v['source_sha256']
        for x in v['input_images']:require(sha(Path(row['input_root'])/x['path'])==x['file_sha256'],'PNG changed')
    output.mkdir()
    put(output/'freeze.json',dict(created_utc=datetime.now(timezone.utc).isoformat(),phase=phase,
        files=files,package_binding=control['package_binding'],videos=videos,max_calls=0 if phase=='development' else 12,
        gt_read=False,caller='root',intervention='Only jerk upper clipping removed'))
    print(json.dumps(dict(status='frozen',phase=phase,sha256=sha(output/'freeze.json'))))

def verify(output):
    frozen=read(output/'freeze.json')
    for p,d in frozen['files'].items():require(sha(p)==d,'Bound file changed: '+p)
    for p,d in frozen['package_binding']['files'].items():require(sha(PACKAGE/p)==d,'Package changed: '+p)
    for row in frozen['videos']:
        for x in row['input']['input_images']:require(sha(Path(row['input_root'])/x['path'])==x['file_sha256'],'Input changed')
    return frozen

def run(output):
    require(sys.flags.isolated,'Use -I')
    f=verify(output);require(not (output/'report.json').exists() and not (output/'traces').exists(),'No overwrite/retry')
    report=dict(status='running',phase=f['phase'],freeze_sha256=sha(output/'freeze.json'),
        call_count=0,max_calls=f['max_calls'],network_attempts=0,videos=[])
    def deny(*a,**kw):report['network_attempts']+=1;raise RuntimeError('Network blocked')
    socket.socket.connect=deny;socket.socket.connect_ex=deny;socket.create_connection=deny
    sys.path.insert(0,str(PACKAGE/'model/stage2/code'))
    try:
        import cv2
        import numpy as np
        cv2.setNumThreads(2)
        candidate=shared.load_module('solution.stage2_uncapped_jerk_v6c',CANDIDATE)
        base=candidate.baseline
        cm=nullcontext(None) if f['phase']=='development' else base.CandidateVLM(PACKAGE/'model/stage2/vlm',precision='nf4')
        with cm as model:
            if model is not None:model.torch.set_num_threads(2);report['runtime']=model.candidate_metadata
            for row in f['videos']:
                paths=[Path(row['input_root'])/x['path'] for x in row['input']['input_images']]
                start=time.perf_counter();valid,base_scores,new_scores,features=candidate._dual_motion_scan(paths);seconds=time.perf_counter()-start
                nums=[base.base._frame_number(p) for p in valid]
                folder=output/'traces'/row['ID'];folder.mkdir(parents=True)
                calls=[]
                if f['phase']=='development':
                    old=row['baseline_trace'];require(nums==old['frame_numbers'],'Decodevalid changed')
                    require(np.array_equal(base_scores,np.asarray(old['motion_scores'],dtype=np.float32)),'V5 score reproduction failed')
                    prediction,diagnostics=deepcopy(old['prediction']),deepcopy(old['diagnostics'])
                    exact=True
                else:
                    recorder=shared.Recorder(model,folder,report)
                    prediction,diagnostics=base._predict_file(valid,base_scores,recorder);calls=recorder.calls
                    require(len(calls)==4,'Four-call contract')
                    exact=None
                changed,newdiag=candidate.apply_collision(valid,prediction,diagnostics,new_scores,base_scores)
                require(all(changed[k]==prediction[k] for k in ('entry_frame','entry_side','evasion_space')),'Other field changed')
                trace=dict(ID=row['ID'],baseline=prediction,candidate=changed,baseline_diagnostics=diagnostics,diagnostics=newdiag,
                    frame_numbers=nums,base_scores=base_scores.tolist(),new_scores=new_scores.tolist(),features=features.tolist(),
                    base_scores_equal_frozen_V5=exact,scan_seconds=seconds,calls=calls)
                put(folder/'trace.json',trace);report['videos'].append(trace)
                print(json.dumps(dict(ID=row['ID'],baseline=prediction,candidate=changed,scan_seconds=seconds,base_exact=exact)),flush=True)
        require(report['call_count']==report['max_calls'] and report['network_attempts']==0,'Budget/offline contract')
        verify(output);report['status']='complete'
    except BaseException as e:report.update(status='failed',error=repr(e));raise
    finally:put(output/'report.json',report)
    print(json.dumps(dict(status=report['status'],phase=report['phase'],calls=report['call_count'])))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['freeze','run']);p.add_argument('--phase',choices=['development','reserved']);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.command=='freeze':require(a.phase,'Phase required');freeze(a.phase,a.output.resolve())
    else:run(a.output.resolve())
