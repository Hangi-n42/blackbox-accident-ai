"""Development-only human-target conditional diagnosis, never a submission model.

Replays the exact V6A control mosaics, candidates and question with the user's
target description prepended. No new time window, model or acceptance gate.
"""
from datetime import datetime, timezone
from pathlib import Path
import argparse
import hashlib
import importlib.util
import json
import os
import socket
import sys
import time

sys.dont_write_bytecode=True
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='2'
os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
HERE=Path(__file__).resolve().parent
CONTROL=HERE/'contact_verify_v6a_development'
OUTPUT=HERE/'human_target_diagnostic'
IDS=['00000','00003','00004']
DESCRIPTIONS={
    '00000':('NEXAR_REVIEW_00000_review_1789400071868.json',
        'The collision counterpart is the white vehicle that approaches from the left and suddenly enters toward the right.'),
    '00003':('NEXAR_REVIEW_00003_review_1789444125537.json',
        'The collision counterpart is the vehicle directly ahead that appears black.'),
    '00004':('NEXAR_REVIEW_00004_review_1789444066492.json',
        'The collision counterpart is the dark vehicle that suddenly comes from the right.'),
}

def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def put(p,data):
    with Path(p).open('x',encoding='utf-8') as f:json.dump(data,f,ensure_ascii=False,indent=2,allow_nan=False)
def require(ok,message):
    if not ok:raise ValueError(message)
def module(name,p):
    spec=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m

def freeze():
    require(not OUTPUT.exists(),'Refuse overwrite')
    runner=module('diagnostic_control_runner',HERE/'run_contact_verify_v6a.py')
    old=runner.verify(CONTROL);report=read(CONTROL/'report.json')
    require(old['phase']=='development' and report['status']=='complete','Wrong control')
    require([v['ID'] for v in report['videos']]==IDS and report['call_count']==3,'Control IDs/calls')
    require(report['freeze_sha256']==sha(CONTROL/'freeze.json'),'Control freeze mismatch')
    files={str(p):sha(p) for p in (Path(__file__),HERE/'run_contact_verify_v6a.py',CONTROL/'freeze.json',CONTROL/'report.json')}
    rows=[]
    for r in report['videos']:
        ID=r['ID'];name,description=DESCRIPTIONS[ID];rp=HERE/'user_reviews'/name;review=read(rp)
        require(review['ID']=='NEXAR_REVIEW_'+ID and review['record_type']=='human_review_draft','Wrong human review')
        files[str(rp)]=sha(rp)
        image_path=CONTROL/'traces'/ID/'call_1/input_0.png';files[str(image_path)]=sha(image_path)
        prompt=r['calls'][0]['prompt']
        prefix='Use this externally supplied description to identify the same counterpart throughout the images: '+description+' '
        rows.append(dict(ID=ID,review_path=str(rp),human_target_original=review['review']['target'],
            english_translation=description,image_path=str(image_path),control_prompt=prompt,
            diagnostic_prompt=prefix+prompt,prompt_prefix=prefix,
            max_new_tokens=r['calls'][0]['max_new_tokens'],control=r))
    OUTPUT.mkdir()
    put(OUTPUT/'freeze.json',dict(created_utc=datetime.now(timezone.utc).isoformat(),
        scope='Human-target conditional development diagnosis only; forbidden for candidate adoption or submission',
        phase='development_diagnostic',candidate=False,submission_eligible=False,
        injected_human_information='review.target only, conservative English translation recorded verbatim; no human frame/time/bbox supplied to model',
        hypothesis='Determine whether providing the reviewed target description alone changes V6A contact selection on identical image evidence.',
        limits='Language hint is not verified visual grounding. Failure does not prove identity is irrelevant; improvement would not establish automatic target identification or deployable accuracy.',
        budget=dict(calls=3,max_new_tokens=48,retries=0,followup_prompt_search=False),
        files=files,control_files=old['files'],package_binding=old['package_binding'],videos=rows))
    print(json.dumps(dict(status='frozen',sha256=sha(OUTPUT/'freeze.json'),submission_eligible=False)))

def verify():
    f=read(OUTPUT/'freeze.json')
    require(f['candidate'] is False and f['submission_eligible'] is False,'Diagnostic cannot be promoted')
    require([r['ID'] for r in f['videos']]==IDS,'Reserved data forbidden')
    for p,d in {**f['control_files'],**f['files']}.items():require(sha(p)==d,'Bound file changed: '+p)
    package=HERE.parents[1]/'artifacts/submissions/verify_v5'
    for p,d in f['package_binding']['files'].items():require(sha(package/p)==d,'Package changed: '+p)
    return f,package

def run():
    require(sys.flags.isolated,'Use -I')
    f,package=verify()
    require(not (OUTPUT/'report.json').exists() and not (OUTPUT/'traces').exists(),'No retry/overwrite')
    report=dict(status='running',scope=f['scope'],candidate=False,submission_eligible=False,
        freeze_sha256=sha(OUTPUT/'freeze.json'),call_count=0,network_attempts=0,videos=[])
    def deny(*a,**kw):report['network_attempts']+=1;raise RuntimeError('Offline only')
    socket.socket.connect=deny;socket.socket.connect_ex=deny;socket.create_connection=deny
    sys.path.insert(0,str(package/'model/stage2/code'))
    try:
        from PIL import Image
        from solution.vlm_candidate import CandidateVLM
        with CandidateVLM(package/'model/stage2/vlm',precision='nf4') as model:
            model.torch.set_num_threads(2);report['runtime']=model.candidate_metadata
            for row in f['videos']:
                require(row['max_new_tokens']==48 and row['diagnostic_prompt']==row['prompt_prefix']+row['control_prompt'],'One-variable question contract')
                with Image.open(row['image_path']) as im:image=im.convert('RGB')
                folder=OUTPUT/'traces'/row['ID'];folder.mkdir(parents=True)
                model.torch.cuda.synchronize();start=time.perf_counter();report['call_count']+=1
                raw=model.ask([image],row['diagnostic_prompt'],max_new_tokens=48)
                model.torch.cuda.synchronize()
                rec=dict(ID=row['ID'],prompt=row['diagnostic_prompt'],max_new_tokens=48,raw=raw,
                    seconds=time.perf_counter()-start,image_path=row['image_path'],image_sha256=sha(row['image_path']),
                    image_rgb_sha256=hashlib.sha256(image.tobytes()).hexdigest(),
                    candidate=False,submission_eligible=False)
                put(folder/'record.json',rec);report['videos'].append(rec)
                print(json.dumps(rec,ensure_ascii=False),flush=True)
        require(report['call_count']==3 and report['network_attempts']==0,'Budget/offline failure')
        verify();report['status']='complete'
    except BaseException as e:report.update(status='failed',error=repr(e));raise
    finally:put(OUTPUT/'report.json',report)
    print(json.dumps(dict(status=report['status'],call_count=report['call_count'],submission_eligible=False)))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['freeze','run']);a=p.parse_args()
    freeze() if a.command=='freeze' else run()
