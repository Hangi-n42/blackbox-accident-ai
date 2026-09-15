"""Prepare only until actual reviews are independently imported and bound.

python -I -B round2_validation/run_validation.py freeze|run --output NEW_OR_FROZEN
Inference never reads annotation JSON contents; required importer binding has no answers.
"""
import argparse
from copy import deepcopy
from datetime import datetime,timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import socket
import sys
import time

sys.dont_write_bytecode=True
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='2'
os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
HERE=Path(__file__).resolve().parent
STAGE=HERE.parent
ROOT=STAGE.parents[1]
PROTOCOL=HERE/'protocol.json'
CANDIDATE=STAGE/'candidates/stage2_uncapped_jerk_v6c.py'
CANDIDATE_SHA='3e86e5117230fb4680c1d4af630cb02dd7e9aca99b7c4184d7baed13f479d64e'
FULL_RUNNER=STAGE/'run_v5_fullframe_diagnostic.py'
PRIOR=STAGE/'uncapped_jerk_fullframe_replay'
SOURCES=ROOT/'research/v6/nexar_review_round2'
IDS=['00008','00010','00013']
TOKENS=[64,48,40,40]


def require(ok,msg):
    if not ok:raise ValueError(msg)


def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))


def put(p,x):
    with Path(p).open('x',encoding='utf-8') as f:json.dump(x,f,ensure_ascii=False,indent=2,allow_nan=False)


def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec)
    sys.modules[name]=m;spec.loader.exec_module(m);return m


def prior_gate():
    e=read(PRIOR/'evaluation.json')
    require(e['gate_advance_to_new_validation_only'] is True and e['adoption_allowed'] is False and all(e['criteria'].values()),'Previous gate does not authorize fresh test')
    require(e['freeze_sha256']==sha(PRIOR/'freeze.json') and e['report_sha256']==sha(PRIOR/'report.json'),'Previous result changed')
    require(e['candidate_sha256']==sha(CANDIDATE)==CANDIDATE_SHA,'C changed')
    require(sha(e['evaluator_path'])==e['evaluator_sha256'],'Previous evaluator changed')
    for p,h in e['imported_evaluator_sha256s'].items():require(sha(p)==h,'Previous imported evaluator changed')
    return e


def freeze(output):
    require(not output.exists() and output.resolve().is_relative_to(ROOT.resolve()),'New workspace output required')
    protocol=read(PROTOCOL)
    require(protocol['ids']==IDS and protocol['candidate_sha256']==CANDIDATE_SHA and protocol['max_calls']==12,'Protocol changed')
    previous=prior_gate()
    full=load('_round2_full_input_helpers',FULL_RUNNER)
    package=full.package_binding()
    manifest=read(SOURCES/'manifest.json')
    require([Path(x['path']).stem for x in manifest['selected']]==IDS,'Selection differs')
    records=[]
    for ID,item in zip(IDS,manifest['selected']):
        record=full.case_record(ID)
        require(record['input']['source_sha256']==item['sha256'] and Path(record['input']['source_path']).resolve()==(SOURCES/f'{ID}.mp4').resolve(),'Source manifest mismatch')
        records.append(record)
    paths=[Path(__file__).resolve(),PROTOCOL,CANDIDATE,FULL_RUNNER,PRIOR/'evaluation.json',PRIOR/'freeze.json',PRIOR/'report.json',
           SOURCES/'manifest.json',SOURCES/'selection_policy_frozen.json',SOURCES/'download_plan.json',SOURCES/'overlap_report.json',SOURCES/'LICENSE',SOURCES/'ATTRIBUTION.json']
    files={str(p):sha(p) for p in paths}
    files[previous['evaluator_path']]=previous['evaluator_sha256']
    files.update(previous['imported_evaluator_sha256s'])
    output.mkdir()
    put(output/'freeze.json',dict(created_utc=datetime.now(timezone.utc).isoformat(),phase='round2_fresh_validation',ids=IDS,
        files=files,package_binding=package,videos=records,max_calls=12,tokens_per_file=TOKENS,model_loads=1,
        gt_read=False,human_binding_at_source_freeze=None,source_selection_only_no_independence_claim=True))
    print(json.dumps(dict(status='source_frozen_human_binding_required',ids=IDS)))


def verify(output):
    frozen=read(output/'freeze.json')
    require(frozen['ids']==IDS and [x['ID'] for x in frozen['videos']]==IDS and frozen['max_calls']==12,'Frozen cohort differs')
    for p,h in frozen['files'].items():require(sha(p)==h,'Frozen dependency changed')
    require(sha(CANDIDATE)==CANDIDATE_SHA,'Fixed C changed')
    full=load('_round2_full_input_helpers',FULL_RUNNER)
    require(full.package_binding()==frozen['package_binding'],'Package changed')
    for record in frozen['videos']:require(full.case_record(record['ID'])==record,'Full source/case/mapping/PNG changed')
    return frozen


def review_guard(output,frozen):
    binding_path=output/'review_binding.json'
    require(binding_path.is_file(),'RUN BLOCKED: actual human review_binding.json not imported')
    b=read(binding_path)
    require(b.get('status')=='VALIDATED_CONTACT_BINDING' and b.get('ground_truth_promotion') is False,'Actual importer-approved draft binding required')
    require(b['freeze_sha256']==sha(output/'freeze.json') and b['protocol_sha256']==sha(PROTOCOL),'Human binding freeze/protocol mismatch')
    require(b['predictions_seen_before_binding'] is False and [x['ID'] for x in b['reviews']]==IDS,'Human binding is not the fixed prereview cohort')
    validator=b['validator'];vp=Path(validator['path']).resolve()
    require(vp.is_relative_to(ROOT.resolve()) and sha(vp)==validator['sha256'],'Importer source hash mismatch')
    audit=b['integrity_report'];require(sha(audit['path'])==audit['sha256'],'Imported integrity report changed')
    for item,record in zip(b['reviews'],frozen['videos']):
        require(set(item)=={'ID','review_path','review_sha256','source_sha256','frame_mapping_sha256','contact_status','native_pts_validated','selected_png_validated'},'Binding must contain metadata only, no answers/descriptions')
        require(item['contact_status']=='observed' and item['native_pts_validated'] is True and item['selected_png_validated'] is True,'Actual observed contact/PTS/pixels not validated')
        require(sha(item['review_path'])==item['review_sha256'] and item['source_sha256']==record['input']['source_sha256'],'Actual review/source bytes changed')
        # Mapping digest is certified by the importer; match full frozen case via Node JSON.stringify.
        import subprocess
        script="const f=require('fs'),c=require('crypto'),x=JSON.parse(f.readFileSync(process.argv[1],'utf8'));process.stdout.write(c.createHash('sha256').update(JSON.stringify(x.frames),'utf8').digest('hex'));"
        digest=subprocess.run(['node','-e',script,record['case_path']],capture_output=True,text=True,check=True).stdout.strip()
        require(digest==item['frame_mapping_sha256'],'Human UI mapping SHA mismatch')
    return dict(path=str(binding_path),sha256=sha(binding_path),validator=validator,integrity_report=audit,
                review_file_sha256s={r['review_path']:r['review_sha256'] for r in b['reviews']})


class QuietRecorder:
    def __init__(self,model,folder,report):self.model=model;self.folder=folder;self.report=report;self.calls=[]
    def ask(self,images,prompt,max_new_tokens=128):
        slot=len(self.calls)
        require(slot<4 and self.report['call_count']<12 and max_new_tokens==TOKENS[slot],'Four-call budget')
        folder=self.folder/f'call_{slot+1}';folder.mkdir()
        for i,image in enumerate(images):image.save(folder/f'input_{i}.png')
        row=dict(prompt=prompt,max_new_tokens=max_new_tokens,status='pending')
        self.calls.append(row);self.report['call_count']+=1
        self.model.torch.cuda.synchronize();start=time.perf_counter()
        try:
            answer=self.model.ask(images,prompt,max_new_tokens=max_new_tokens)
            self.model.torch.cuda.synchronize();row.update(status='complete',raw=answer);return answer
        except BaseException as error:row.update(status='failed',error_type=type(error).__name__);raise
        finally:
            row['seconds']=time.perf_counter()-start;put(folder/'record.json',row)
            print(json.dumps(dict(ID=self.folder.name,status=f'call_{slot+1}_{row["status"]}')),flush=True)


def run(output):
    require(sys.flags.isolated,'Use Python -I')
    require(not (output/'report.json').exists() and not (output/'traces').exists(),'No retry or overwrite')
    # Missing reviews block before heavy package hashing and before any model import.
    require((output/'review_binding.json').is_file(),'RUN BLOCKED: actual human reviews not imported/bound')
    frozen=verify(output);human=review_guard(output,frozen)
    report=dict(status='running',phase='round2_fresh_validation',freeze_sha256=sha(output/'freeze.json'),
                review_binding=human,max_calls=12,call_count=0,network_attempts=0,model_loads=0,human_answer_contents_read=False,videos=[])
    def deny(*a,**kw):report['network_attempts']+=1;raise RuntimeError('Offline validation')
    socket.socket.connect=deny;socket.socket.connect_ex=deny;socket.create_connection=deny
    code=(Path(frozen['package_binding']['package'])/'model/stage2/code').resolve()
    require(not any(k=='solution' or k.startswith('solution.') for k in sys.modules),'Preloaded solution')
    sys.path.insert(0,str(code))
    try:
        import numpy as np
        candidate=load('solution.stage2_uncapped_jerk_v6c',CANDIDATE)
        base=candidate.baseline;candidate.primitives.cv2.setNumThreads(2)
        for name,m in list(sys.modules.items()):
            if name=='solution' or name.startswith('solution.'):
                p=Path(m.__file__).resolve();require(p==CANDIDATE.resolve() or p.is_relative_to(code),'Wrong runtime module')
        with base.CandidateVLM(Path(frozen['package_binding']['package'])/'model/stage2/vlm',precision='nf4') as model:
            report['model_loads']+=1;model.torch.set_num_threads(2);report['runtime']=model.candidate_metadata
            for record in frozen['videos']:
                ID=record['ID'];v=record['input'];paths=[Path(record['input_root'])/x['path'] for x in v['input_images']]
                start=time.perf_counter();valid,bs,ns,features=candidate._dual_motion_scan(paths);elapsed=time.perf_counter()-start
                expected=(candidate.primitives._robust_scale(features[:,0])+.6*candidate.primitives._robust_scale(features[:,1])+.25*candidate.primitives._robust_scale(features[:,2]));expected[0]=0
                require(bs.dtype==np.dtype('float32') and bs.tobytes()==expected.tobytes(),'Original V5 score expression not exact')
                folder=output/'traces'/ID;folder.mkdir(parents=True);rec=QuietRecorder(model,folder,report)
                old,diag=base._predict_file(valid,bs,rec)
                new,newdiag=candidate.apply_collision(valid,deepcopy(old),deepcopy(diag),ns,bs)
                require(len(rec.calls)==4 and all(old[k]==new[k] for k in ('entry_frame','entry_side','evasion_space')),'Policy/other-field mismatch')
                row=dict(ID=ID,baseline=old,candidate=new,baseline_diagnostics=diag,diagnostics=newdiag,
                         frame_numbers=[candidate.primitives._frame_number(p) for p in valid],base_scores=bs.tolist(),new_scores=ns.tolist(),features=features.tolist(),
                         base_scores_exact_original_expression=True,base_scores_equal_separate_original_scan=None,scan_seconds=elapsed,calls=rec.calls)
                put(folder/'trace.json',row);report['videos'].append(row)
                print(json.dumps(dict(ID=ID,status='complete')),flush=True)
        require(report['call_count']==12 and report['model_loads']==1 and report['network_attempts']==0,'Offline/model/call contract')
        verify(output);require(review_guard(output,frozen)==human,'Human binding changed during run')
        report['status']='complete'
    except BaseException as error:report.update(status='failed',error_type=type(error).__name__);raise
    finally:put(output/'report.json',report)
    print(json.dumps(dict(status='complete',ids=IDS)))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=('freeze','run'));p.add_argument('--output',required=True,type=Path);a=p.parse_args()
    if a.command=='freeze':freeze(a.output.resolve())
    else:run(a.output.resolve())
