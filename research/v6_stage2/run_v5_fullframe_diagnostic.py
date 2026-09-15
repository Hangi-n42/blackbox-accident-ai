"""Frozen V5 full-PNG baseline only; no candidate/GT access or media duplication.

python -I -B run_v5_fullframe_diagnostic.py freeze|run
  --output research/v6_stage2/v5_fullframe_diagnostic
Only the root runs GPU inference. Freeze itself performs hashes/metadata checks.
"""
from datetime import datetime, timezone
import argparse
import hashlib
import importlib.util
import json
import math
import os
import re
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
IDS = ('00000','00003','00004','00005','00006','00007')
PACKAGE = ROOT/'artifacts/submissions/verify_v5'
SELECTION = ROOT/'artifacts/submissions/v5_selection_frozen.json'
DIST = ROOT/'research/v6_review_tool/dist'
PROTOCOL = HERE/'v5_fullframe_diagnostic_protocol.json'
RECORDER_SOURCE = HERE/'run_contact_verify_v6a.py'
TOKENS = [64,48,40,40]


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write_new(path, value):
    with Path(path).open('x',encoding='utf-8') as stream:
        json.dump(value,stream,ensure_ascii=False,indent=2,allow_nan=False)


def package_binding():
    selected = read(SELECTION)
    require(selected['stage2_module']=='stage2_motion_collision','Wrong baseline selection')
    expected = {k:v for k,v in selected['expected_archive_sha256'].items() if k.startswith('model/stage2/') or k=='inference.py'}
    for name,digest in expected.items():
        path=(PACKAGE/name).resolve()
        require(path.is_relative_to(PACKAGE.resolve()) and sha(path)==digest,'Actual V5 package bytes differ: '+name)
    return dict(package=str(PACKAGE),selection_sha256=sha(SELECTION),files=expected)


def case_record(ID):
    folder=DIST/'cases'/f'NEXAR_REVIEW_{ID}'
    cp,mp=folder/'case.json',folder/'mapping.json'
    case,mapping=read(cp),read(mp)
    require(case['ID']==f'NEXAR_REVIEW_{ID}' and case.get('labels') is None,'Unexpected case or labels embedded in case')
    frames=case['frames']; times=mapping['frame_pts']
    require(frames and [x['frame'] for x in frames]==list(range(len(frames))),'Full native frame numbering must be 0..N-1')
    require(times==[dict(frame=x['frame'],pts_seconds=x['pts_seconds']) for x in frames],'Case/mapping PTS differ')
    require(all(math.isfinite(x['pts_seconds']) for x in times) and all(b['pts_seconds']>a['pts_seconds'] for a,b in zip(times,times[1:])),'Invalid native PTS')
    source=Path(case['video_path'])
    require(sha(source)==case['video_sha256']==mapping['source_video_sha256'],'Original source SHA mismatch')
    images=[]
    for frame in frames:
        image=(DIST/frame['image']).resolve()
        require(image.parent==folder.resolve() and image.name==f"frame_{frame['frame']:06d}.png",'Unexpected full-frame PNG path')
        require(sha(image)==frame['sha256'],'Full-frame PNG SHA mismatch')
        images.append(dict(frame=frame['frame'],pts_seconds=frame['pts_seconds'],path=frame['image'],file_sha256=frame['sha256']))
    inventory={p.name for p in folder.glob('frame_*.png') if re.fullmatch(r'frame_[0-9]{6}\.png',p.name)}
    require(inventory=={Path(x['path']).name for x in images},'Full-frame image inventory differs from case')
    ignored=[dict(name=p.name,sha256=sha(p)) for p in sorted(folder.glob('frame_*.interrupted.png'))]
    manifest_sha=hashlib.sha256(json.dumps(images,sort_keys=True).encode()).hexdigest()
    return dict(ID=ID,input_root=str(DIST),case_path=str(cp),case_sha256=sha(cp),mapping_path=str(mp),mapping_sha256=sha(mp),ignored_preserved_interrupted_images=ignored,
                input=dict(ID=ID,source_path=str(source),source_sha256=case['video_sha256'],source_frame_pts=times,
                           input_images=images,input_manifest_sha256=manifest_sha,source_frame_count=len(frames),
                           time_origin=case['time_origin'],mapping_method=case['mapping_method'],
                           all_native_frames_included=True,sampling_hz=None,image_format='existing lossless PNG'))


def freeze(output):
    require(not output.exists(),'Freeze output exists; no overwrite')
    require(output.resolve().is_relative_to(ROOT.resolve()),'Output must be in workspace')
    protocol=read(PROTOCOL)
    require(protocol['ids']==list(IDS) and protocol['max_calls']==24 and protocol['tokens_per_video']==TOKENS,'Protocol differs')
    package=package_binding()  # One complete package hash check for preparation.
    records=[case_record(ID) for ID in IDS]
    refs={}
    for folder in ('nexar_baseline_run','nexar_dev00004_baseline_run','uncapped_jerk_v6c_reserved'):
        for name in ('freeze.json','report.json'):
            path=HERE/folder/name
            refs[str(path)]=sha(path)  # Hash only: prior answers never enter this inference.
    by_id={}
    for ID in IDS:
        previous='nexar_baseline_run' if ID in ('00000','00003') else ('nexar_dev00004_baseline_run' if ID=='00004' else 'uncapped_jerk_v6c_reserved')
        by_id[ID]=dict(report_path=str(HERE/previous/'report.json'),freeze_path=str(HERE/previous/'freeze.json'),
                       prediction_field='prediction' if ID in ('00000','00003','00004') else 'baseline')
    files={str(p):sha(p) for p in (Path(__file__).resolve(),PROTOCOL,RECORDER_SOURCE)}
    output.mkdir()
    write_new(output/'freeze.json',dict(created_utc=datetime.now(timezone.utc).isoformat(),
        experiment='v5_fullframe_diagnostic',ids=list(IDS),files=files,package_binding=package,
        videos=records,baseline10hz_refs=refs,baseline10hz_by_id=by_id,prior_refs_usage='hashed only; baseline comparison by separate evaluator',
        max_calls=24,tokens_per_video=TOKENS,gt_read=False,model_loads=1,source_role='previously consumed development/diagnostic cases',
        interpretation='Density/timeline comparison with PNG fixed; official JPEG preprocessing equivalence not established'))
    print(json.dumps(dict(status='frozen_no_inference',output=str(output),freeze_sha256=sha(output/'freeze.json'),
                         frame_counts=[v['input']['source_frame_count'] for v in records])))


def verify(output):
    frozen=read(output/'freeze.json')
    require(frozen['ids']==list(IDS) and [x['ID'] for x in frozen['videos']]==list(IDS),'ID cohort changed')
    require(frozen['max_calls']==24 and frozen['tokens_per_video']==TOKENS,'Budget changed')
    for path,digest in frozen['files'].items():require(sha(path)==digest,'Frozen script/protocol changed')
    require(package_binding()==frozen['package_binding'],'Actual V5 package changed')
    for path,digest in frozen['baseline10hz_refs'].items():require(sha(path)==digest,'Prior comparison artifact changed')
    for record in frozen['videos']:
        require(case_record(record['ID'])==record,'Source/case/mapping/full-PNG inventory or bytes changed')
    return frozen


def load_recorder():
    spec=importlib.util.spec_from_file_location('_fullframe_recorder',RECORDER_SOURCE)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.Recorder  # Reuse recording only; candidate functions are never called.


def run(output):
    require(sys.flags.isolated,'Run with Python -I')
    require(not (output/'report.json').exists() and not (output/'traces').exists(),'Existing run; no overwrite or retry')
    require(not any(k=='solution' or k.startswith('solution.') for k in sys.modules),'Solution already imported')
    frozen=verify(output)  # One full package/input hash check before the single model load.
    report=dict(status='running',created_utc=datetime.now(timezone.utc).isoformat(),
                experiment='v5_fullframe_diagnostic',freeze_sha256=sha(output/'freeze.json'),max_calls=24,
                call_count=0,network_attempts=0,model_loads=0,gt_read=False,videos=[])
    def deny(*args,**kwargs):
        report['network_attempts']+=1
        raise RuntimeError('Network blocked in full-frame frozen V5 inference')
    socket.socket.connect=deny;socket.socket.connect_ex=deny;socket.create_connection=deny
    code=(PACKAGE/'model/stage2/code').resolve();sys.path.insert(0,str(code))
    Recorder=load_recorder()
    try:
        import cv2
        cv2.setNumThreads(2)
        from solution import stage2_motion_collision as baseline
        for name,module in list(sys.modules.items()):
            if name=='solution' or name.startswith('solution.'):
                require(Path(module.__file__).resolve().is_relative_to(code),'Non-package solution import')
                require('uncapped' not in name and 'contact_verify' not in name,'Candidate import disallowed')
        report['imported_solution']={k:str(v.__file__) for k,v in sys.modules.items() if k=='solution' or k.startswith('solution.')}
        with baseline.CandidateVLM(PACKAGE/'model/stage2/vlm',precision='nf4') as model:
            report['model_loads']+=1;model.torch.set_num_threads(2);report['runtime']=model.candidate_metadata
            for record in frozen['videos']:
                data=record['input'];ID=record['ID']
                paths=[Path(record['input_root'])/x['path'] for x in data['input_images']]
                start=time.perf_counter();valid,scores,_=baseline.base._motion_scan(paths);scan_seconds=time.perf_counter()-start
                folder=output/'traces'/ID;folder.mkdir(parents=True)
                recorder=Recorder(model,folder,report)
                start=time.perf_counter();prediction,diagnostics=baseline._predict_file(valid,scores,recorder);predict_seconds=time.perf_counter()-start
                require(len(recorder.calls)==4 and [c['max_new_tokens'] for c in recorder.calls]==TOKENS,'Frozen four-call policy differs')
                numbers=[baseline.base._frame_number(path) for path in valid]
                internal=diagnostics['collision_replacement']['base_collision_frame']
                center=numbers.index(internal)
                row=dict(ID=ID,prediction=prediction,diagnostics=diagnostics,frame_numbers=numbers,
                         original_frame_numbers=[x['frame'] for x in data['input_images']],source_frame_pts=data['source_frame_pts'],
                         input_manifest_sha256=data['input_manifest_sha256'],motion_scores=scores.tolist(),calls=recorder.calls,
                         internal_collision_frame=internal,final_motion_collision_frame=prediction['collision_frame'],
                         entry_prefix=numbers[:center+1],scan_seconds=scan_seconds,predict_seconds=predict_seconds,
                         missing_decode_frames=[x['frame'] for x in data['input_images'] if x['frame'] not in set(numbers)])
                write_new(folder/'trace.json',row);report['videos'].append(row)
                print(json.dumps(dict(ID=ID,status='complete',valid_frames=len(valid),scan_seconds=scan_seconds,
                                      predict_seconds=predict_seconds,prediction=prediction)),flush=True)
        require(report['model_loads']==1 and report['call_count']==24 and report['network_attempts']==0,'Model/call/offline budget violated')
        verify(output)  # Once after all six videos; no repeated package hash inside the loop.
        report['status']='complete'
    except BaseException as error:
        report.update(status='failed',error=repr(error));raise
    finally:
        write_new(output/'report.json',report)
    print(json.dumps(dict(status='complete',ids=list(IDS),calls=24,freeze_sha256=report['freeze_sha256'])))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=('freeze','run'))
    parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args()
    if args.command=='freeze':freeze(args.output.resolve())
    else:run(args.output.resolve())
