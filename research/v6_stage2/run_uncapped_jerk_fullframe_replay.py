"""Frozen C CPU-only replay after full-frame V5. Never reads human annotations.

python -I -B run_uncapped_jerk_fullframe_replay.py freeze|run --output NEW_OR_FROZEN
No GPU, no model calls, no retries/overwrite, no result-driven parameter changes.
"""
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

sys.dont_write_bytecode=True
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='2'
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['HF_HUB_OFFLINE']='1'
os.environ['TRANSFORMERS_OFFLINE']='1'
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
BASE_RUN=HERE/'v5_fullframe_diagnostic'
BASE_RUNNER=HERE/'run_v5_fullframe_diagnostic.py'
PROTOCOL=HERE/'uncapped_jerk_fullframe_replay_protocol.json'
CANDIDATE=HERE/'candidates/stage2_uncapped_jerk_v6c.py'
CANDIDATE_SHA='3e86e5117230fb4680c1d4af630cb02dd7e9aca99b7c4184d7baed13f479d64e'
IDS=['00000','00003','00004','00005','00006','00007']


def require(ok,message):
    if not ok:raise ValueError(message)


def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def put(path,obj):
    with Path(path).open('x',encoding='utf-8') as stream:json.dump(obj,stream,ensure_ascii=False,indent=2,allow_nan=False)


def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    value=importlib.util.module_from_spec(spec);sys.modules[name]=value;spec.loader.exec_module(value)
    return value


def baseline_state():
    original=module('_fullframe_binding_only',BASE_RUNNER)
    frozen=original.verify(BASE_RUN)
    report=read(BASE_RUN/'report.json')
    require(report['status']=='complete' and report['call_count']==24 and report['network_attempts']==0 and report['model_loads']==1,'Full V5 baseline not complete')
    require(report['freeze_sha256']==sha(BASE_RUN/'freeze.json'),'Full V5 report/freeze mismatch')
    require([x['ID'] for x in report['videos']]==[x['ID'] for x in frozen['videos']]==IDS,'Full V5 cohort differs')
    return frozen,report


def freeze(output):
    require(not output.exists() and output.resolve().is_relative_to(ROOT.resolve()),'New workspace output required')
    require(sha(CANDIDATE)==CANDIDATE_SHA,'Frozen C changed')
    protocol=read(PROTOCOL)
    require(protocol['ids']==IDS and protocol['candidate_sha256']==CANDIDATE_SHA and protocol['no_new_model_calls'] is True,'Protocol differs')
    frozen,report=baseline_state()
    files={str(p):sha(p) for p in (Path(__file__).resolve(),BASE_RUNNER,PROTOCOL,CANDIDATE,BASE_RUN/'freeze.json',BASE_RUN/'report.json')}
    records=[]
    for record,trace in zip(frozen['videos'],report['videos']):
        require(record['ID']==trace['ID'] and record['input']['input_manifest_sha256']==trace['input_manifest_sha256'],'Full V5 input binding differs')
        records.append({**record,'baseline_trace':trace})
    output.mkdir()
    put(output/'freeze.json',dict(created_utc=datetime.now(timezone.utc).isoformat(),phase='exposed_six_fullframe_diagnostic',
        files=files,package_binding=frozen['package_binding'],videos=records,base_run=str(BASE_RUN),max_calls=0,gt_read=False,
        previous_density_refs=frozen['baseline10hz_refs'],protocol_sha256=sha(PROTOCOL),candidate_sha256=CANDIDATE_SHA))
    print(json.dumps(dict(status='frozen_no_replay',ids=IDS,output=str(output),freeze_sha256=sha(output/'freeze.json'))))


def verify(output):
    frozen=read(output/'freeze.json')
    require([x['ID'] for x in frozen['videos']]==IDS and frozen['max_calls']==0,'Replay cohort/call budget changed')
    for path,digest in frozen['files'].items():require(sha(path)==digest,'Replay bound artifact changed: '+path)
    require(sha(CANDIDATE)==CANDIDATE_SHA,'C code differs')
    original,report=baseline_state()  # Full package/source/case/mapping/PNG verification.
    require(frozen['package_binding']==original['package_binding'],'Evaluated package differs')
    for replay,current,trace in zip(frozen['videos'],original['videos'],report['videos']):
        require({k:v for k,v in replay.items() if k!='baseline_trace'}==current and replay['baseline_trace']==trace,'Cached full baseline changed')
    return frozen


def run(output):
    require(sys.flags.isolated,'Use Python -I')
    require(not (output/'report.json').exists() and not (output/'traces').exists(),'No overwrite/retry')
    frozen=verify(output)
    report=dict(status='running',phase='exposed_six_fullframe_diagnostic',created_utc=datetime.now(timezone.utc).isoformat(),
                freeze_sha256=sha(output/'freeze.json'),max_calls=0,call_count=0,network_attempts=0,model_loads=0,gt_read=False,videos=[])
    def deny(*args,**kwargs):
        report['network_attempts']+=1;raise RuntimeError('No network in CPU-only fullframe replay')
    socket.socket.connect=deny;socket.socket.connect_ex=deny;socket.create_connection=deny
    package=Path(frozen['package_binding']['package'])
    code=(package/'model/stage2/code').resolve()
    require(not any(k=='solution' or k.startswith('solution.') for k in sys.modules),'Existing solution import')
    sys.path.insert(0,str(code))
    try:
        import numpy as np
        candidate=module('solution.stage2_uncapped_jerk_v6c',CANDIDATE)
        candidate.primitives.cv2.setNumThreads(2)
        for name,value in list(sys.modules.items()):
            if name=='solution' or name.startswith('solution.'):
                actual=Path(value.__file__).resolve()
                require(actual==CANDIDATE.resolve() or actual.is_relative_to(code),'Wrong solution source loaded')
        for record in frozen['videos']:
            ID=record['ID'];data=record['input'];old=record['baseline_trace']
            paths=[Path(record['input_root'])/x['path'] for x in data['input_images']]
            start=time.perf_counter();valid,base,new,features=candidate._dual_motion_scan(paths);seconds=time.perf_counter()-start
            numbers=[candidate.primitives._frame_number(path) for path in valid]
            require(numbers==old['frame_numbers'],'Decodevalid differs from full baseline')
            expected=np.asarray(old['motion_scores'],dtype=np.float32)
            require(base.dtype==np.dtype('float32') and base.tobytes()==expected.tobytes(),'Base score bytes differ from full V5')
            before=deepcopy(old['prediction']);diagnostics=deepcopy(old['diagnostics'])
            after,newdiag=candidate.apply_collision(valid,before,diagnostics,new,base)
            require(all(before[k]==after[k] for k in ('entry_frame','entry_side','evasion_space')),'Noncollision output changed')
            folder=output/'traces'/ID;folder.mkdir(parents=True)
            row=dict(ID=ID,baseline=before,candidate=after,baseline_diagnostics=diagnostics,diagnostics=newdiag,
                     frame_numbers=numbers,base_scores=base.tolist(),new_scores=new.tolist(),features=features.tolist(),
                     base_scores_equal_frozen_V5=True,scan_seconds=seconds,calls=[],
                     input_manifest_sha256=data['input_manifest_sha256'],source_frame_pts=data['source_frame_pts'])
            put(folder/'trace.json',row);report['videos'].append(row)
            print(json.dumps(dict(ID=ID,base_exact=True,scan_seconds=seconds,baseline=before,candidate=after)),flush=True)
        require(report['call_count']==0 and report['model_loads']==0 and report['network_attempts']==0,'CPU-only contract violated')
        verify(output)
        report['status']='complete'
    except BaseException as error:
        report.update(status='failed',error=repr(error));raise
    finally:put(output/'report.json',report)
    print(json.dumps(dict(status='complete',ids=IDS,model_calls=0,output=str(output))))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('command',choices=('freeze','run'));parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args()
    if args.command=='freeze':freeze(args.output.resolve())
    else:run(args.output.resolve())
