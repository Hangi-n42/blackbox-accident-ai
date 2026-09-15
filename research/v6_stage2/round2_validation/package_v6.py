"""Package unchanged C only after the actual fresh three-source gate passes.

No GPU, inference, upload or human-reference creation. Missing gate writes nothing.
python -I -B package_v6.py --validation-run round2_validation/run
"""
import sys
sys.dont_write_bytecode=True
import argparse,hashlib,importlib.util,json,math,shutil,zipfile
from pathlib import Path
from datetime import datetime,timezone

HERE=Path(__file__).resolve().parent
STAGE=HERE.parent
ROOT=HERE.parents[2]
ARTIFACTS=ROOT/'artifacts/submissions'
PACKAGE=ARTIFACTS/'verify_v5'
SELECTION=ARTIFACTS/'v5_selection_frozen.json'
PROTOCOL=HERE/'protocol.json'
RUNNER=HERE/'run_validation.py'
EVALUATOR=HERE/'evaluate_validation.py'
HELPER=STAGE/'package_uncapped_jerk_v6c.py'
HELPER_SHA='d87242a87c48a9792e19a23eaa3367ef45d76d14d6211468624791993c01860f'
PROTOCOL_SHA='df71f0933b1b3dacfedec376472af9d4d897c7aa8498561aba8bb0c9544164a5'
CANDIDATE=STAGE/'candidates/stage2_uncapped_jerk_v6c.py'
C_SHA='3e86e5117230fb4680c1d4af630cb02dd7e9aca99b7c4184d7baed13f479d64e'
MEMBER='model/stage2/code/solution/stage2_uncapped_jerk_v6c.py'
IDS=['00008','00010','00013']
PHASE='round2_fresh_validation'
EPS=1e-12
CONDITION_KEYS={'additional_correct_at_least_one','previous_success_losses_zero','mean_absolute_error_nonincrease',
                'other_three_fields_identical','exact_native_base_and_offline_contracts'}
GATE=dict(all_three_known_observed_contact_references_required=True,minimum_additional_correct_within_0_3_seconds=1,
          maximum_previously_correct_cases_lost=0,mean_absolute_contact_time_error_must_not_increase=True,
          other_three_outputs_identical=True,exact_base_score_expression_and_offline_original_frame_contracts_required=True,
          roundoff_epsilon_seconds=EPS)

def require(ok,message):
    if not ok:raise ValueError(message)
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def add(bindings,p,digest):
    key=str(Path(p).resolve())
    require(isinstance(digest,str) and len(digest)==64,'Invalid SHA binding')
    require(key not in bindings or bindings[key]==digest,'Conflicting evidence bindings: '+key)
    bindings[key]=digest
def verify(bindings):
    for p,h in bindings.items():require(sha(p)==h,'Bound bytes changed: '+p)
def bind_existing(bindings,complete,p,digest):
    key=str(Path(p).resolve())
    require(complete.get(key)==digest,'Evaluator omitted/differs on required binding: '+key)
    add(bindings,p,digest)

def fresh_gate(run_dir):
    ep,fp,rp,bp=[run_dir/n for n in ('evaluation.json','freeze.json','report.json','review_binding.json')]
    require(all(p.is_file() for p in (ep,fp,rp,bp)),'Actual fresh evaluation/freeze/report/review_binding required; no output written')
    e,f,r,b=map(read,(ep,fp,rp,bp))
    require(e['phase']==f['phase']==r['phase']==PHASE and e['gate_passed'] is True,'Fresh phase/gate failed')
    require(set(e['conditions'])==CONDITION_KEYS and all(x is True for x in e['conditions'].values()),'Fresh gate conditions failed')
    require(e['freeze_sha256']==r['freeze_sha256']==sha(fp) and e['report_sha256']==sha(rp) and e['review_binding_sha256']==sha(bp),'Evaluation/report/freeze binding mismatch')
    require(Path(e['evaluator_path']).resolve()==EVALUATOR.resolve() and sha(EVALUATOR)==e['evaluator_sha256'],'Wrong/changed evaluator')
    require(e['candidate_sha256']==sha(CANDIDATE)==C_SHA and e['protocol_sha256']==sha(PROTOCOL)==PROTOCOL_SHA,'Candidate/protocol changed')
    protocol=read(PROTOCOL)
    require(protocol['ids']==IDS and protocol['gate']==GATE and protocol['candidate_sha256']==C_SHA,'Fresh protocol differs')
    require([v['ID'] for v in e['videos']]==[v['ID'] for v in f['videos']]==[v['ID'] for v in r['videos']]==f['ids']==IDS,'Wrong fresh cohort')
    require(r['status']=='complete' and r['max_calls']==r['call_count']==f['max_calls']==12 and r['model_loads']==f['model_loads']==1 and r['network_attempts']==0,'One model/12 calls/offline contract failed')
    require(f['tokens_per_file']==[64,48,40,40] and f['gt_read'] is False and r['human_answer_contents_read'] is False,'Inference answer-separation contract failed')
    rt=r['runtime']
    require(rt['precision']=='nf4' and rt['compute_dtype']=='float16' and rt['prequantized_checkpoint'] is True and rt['double_quant'] is False,'Actual saved NF4 runtime differs')
    require(b['status']=='VALIDATED_CONTACT_BINDING' and b['ground_truth_promotion'] is False and b['predictions_seen_before_binding'] is False,'Missing actual prereview binding')
    require(b['freeze_sha256']==sha(fp) and b['protocol_sha256']==PROTOCOL_SHA and [v['ID'] for v in b['reviews']]==IDS,'Review binding differs')
    expected_review_binding=dict(path=str(bp),sha256=sha(bp),validator=b['validator'],integrity_report=b['integrity_report'],
      review_file_sha256s={v['review_path']:v['review_sha256'] for v in b['reviews']})
    require(r['review_binding']==expected_review_binding,'Run used different reviews')
    complete={}
    for p,h in e['all_bound_file_sha256s'].items():add(complete,p,h)
    require(complete,'No complete evidence hashes')
    bindings=dict(complete)
    for p,h in f['files'].items():bind_existing(bindings,complete,p,h)
    for p,h in [(fp,sha(fp)),(rp,sha(rp)),(bp,sha(bp)),(EVALUATOR,e['evaluator_sha256']),(PROTOCOL,PROTOCOL_SHA),(CANDIDATE,C_SHA)]:bind_existing(bindings,complete,p,h)
    require(f['files'][str(RUNNER)]==sha(RUNNER),'Actual runner not frozen')
    for p,h in e['imported_evaluator_sha256s'].items():bind_existing(bindings,complete,p,h)
    for name in ('validator','integrity_report'):bind_existing(bindings,complete,b[name]['path'],b[name]['sha256'])
    for p,h in e['review_file_sha256s'].items():bind_existing(bindings,complete,p,h)
    add(bindings,ep,sha(ep));add(bindings,Path(__file__),sha(__file__));add(bindings,HELPER,HELPER_SHA)
    selection=read(SELECTION);expected=selection['expected_archive_sha256']
    require(len(expected)==46 and selection['stage1_module']=='stage1_v4' and selection['stage2_module']=='stage2_motion_collision' and selection['stage3_module']=='stage3_v5_compatible','Wrong actual V5 selection')
    stage2_expected={p:h for p,h in expected.items() if p.startswith('model/stage2/') or p=='inference.py'}
    package_binding=f['package_binding']
    require(Path(package_binding['package']).resolve()==PACKAGE.resolve() and package_binding['files']==stage2_expected and package_binding['selection_sha256']==sha(SELECTION),'Evaluated V5 package differs')
    for name,h in stage2_expected.items():bind_existing(bindings,complete,PACKAGE/name,h)
    add(bindings,SELECTION,sha(SELECTION))
    old_errors=[];new_errors=[]
    for ev,record,trace,item in zip(e['videos'],f['videos'],r['videos'],b['reviews']):
        ID=record['ID'];source=record['input'];mapping=source['source_frame_pts'];numbers=trace['frame_numbers']
        require(numbers==list(range(len(mapping))) and all(type(n)is int for n in numbers) and [x['frame'] for x in mapping]==numbers,'Not full original native frame numbers')
        times={x['frame']:x['pts_seconds'] for x in mapping};ts=list(times.values())
        require(ts and all(type(t)in(int,float) and math.isfinite(t) for t in ts) and all(v>u for u,v in zip(ts,ts[1:])),'Invalid native PTS')
        require([x['frame'] for x in source['input_images']]==numbers,'Full input images mismatch')
        require(hashlib.sha256(json.dumps(source['input_images'],sort_keys=True).encode()).hexdigest()==source['input_manifest_sha256'],'Input manifest differs')
        for image in source['input_images']:
            p=(Path(record['input_root'])/image['path']).resolve()
            require(p.is_relative_to(Path(record['input_root']).resolve()) and p.name==f"frame_{image['frame']:06d}.png" and image['pts_seconds']==times[image['frame']],'PNG/original PTS differs')
            bind_existing(bindings,complete,p,image['file_sha256'])
        for k in ('case','mapping'):bind_existing(bindings,complete,record[k+'_path'],record[k+'_sha256'])
        bind_existing(bindings,complete,source['source_path'],source['source_sha256'])
        require(item['ID']==ID and item['contact_status']=='observed' and item['native_pts_validated'] is True and item['selected_png_validated'] is True,'Observed human reference missing')
        require(ev['review_path']==item['review_path'] and ev['review_sha256']==item['review_sha256']==e['review_file_sha256s'][item['review_path']],'Per-video review binding differs')
        bind_existing(bindings,complete,item['review_path'],item['review_sha256'])
        review=read(item['review_path']);contact=review['review']['contact']
        require(review['ID']=='NEXAR_REVIEW_'+ID and review['record_type']=='human_review_draft' and review['evaluation_eligible'] is False,'Wrong original human draft')
        require(review['source_video_sha256']==source['source_sha256']==item['source_sha256'] and review['frame_mapping_sha256']==item['frame_mapping_sha256'],'Review/source mapping differs')
        require(contact['status']=='observed' and type(contact['frame'])is int and contact['frame'] in times and contact['pts_seconds']==times[contact['frame']],'Known native contact required')
        require(ev['human_contact_draft']==contact and ev['native_verification']['all_native_pts_independently_verified'] is True and ev['native_verification']['selected_contact_png_rgb_verified'] is True,'Native audit incomplete')
        selected=ev['native_verification']['selected']
        require(selected['frame']==contact['frame'] and selected['pts_seconds']==contact['pts_seconds'],'Native audited contact differs')
        require(len(trace['calls'])==4 and [c['max_new_tokens'] for c in trace['calls']]==[64,48,40,40] and all(c['status']=='complete' and isinstance(c.get('raw'),str) for c in trace['calls']),'Original four-call record required')
        require(trace['base_scores_exact_original_expression'] is True and trace['base_scores_equal_separate_original_scan'] is None,'False base score provenance')
        before,after=trace['baseline'],trace['candidate']
        require(ev['baseline']==before and ev['candidate']==after and ev['other_fields_identical'] is True,'Evaluated predictions differ')
        for pred,key in ((before,'base_scores'),(after,'new_scores')):
            require(set(pred)=={'collision_frame','entry_frame','entry_side','evasion_space'},'Prediction field contract differs')
            require(all(type(pred[k])is int and pred[k] in times for k in ('collision_frame','entry_frame')),'Invalid output original frame')
            require(pred['entry_side'] in ('LEFT','RIGHT') and type(pred['evasion_space'])is int and pred['evasion_space'] in (0,1),'Invalid output class')
            scores=trace[key]
            require(len(scores)==len(numbers) and all(type(s)in(int,float) and math.isfinite(s) for s in scores),'Invalid recorded motion scores')
            require(pred['collision_frame']==numbers[max(range(len(scores)),key=lambda i:scores[i])],'Prediction differs from actual score argmax')
        require(all(before[k]==after[k] for k in ('entry_frame','entry_side','evasion_space')),'Other three fields changed')
        a=abs(times[before['collision_frame']]-contact['pts_seconds']);c=abs(times[after['collision_frame']]-contact['pts_seconds'])
        require(abs(a-ev['baseline_abs_error_seconds'])<=EPS and abs(c-ev['candidate_abs_error_seconds'])<=EPS,'Raw native-error recomputation differs')
        old_errors.append(a);new_errors.append(c)
    prev=[v<=.3+EPS for v in old_errors];new=[v<=.3+EPS for v in new_errors]
    old_mean=sum(old_errors)/3;new_mean=sum(new_errors)/3;lost=sum(a and not b for a,b in zip(prev,new))
    checks=dict(additional_correct_at_least_one=sum(new)-sum(prev)>=1,previous_success_losses_zero=lost==0,
      mean_absolute_error_nonincrease=new_mean<=old_mean+EPS,other_three_fields_identical=True,exact_native_base_and_offline_contracts=True)
    require(all(checks.values()) and checks==e['conditions'],'Independent contact gate recomputation failed')
    require(e['baseline_correct_count']==sum(prev) and e['candidate_correct_count']==sum(new) and e['lost']==lost and abs(e['baseline_mean_abs_error_seconds']-old_mean)<=EPS and abs(e['candidate_mean_abs_error_seconds']-new_mean)<=EPS,'Aggregate arithmetic differs')
    verify(bindings) # Includes all actual review, source, PNG, model and code bytes.
    return expected,bindings,dict(evaluation_path=str(ep),evaluation_sha256=sha(ep),phase=PHASE,ids=IDS,conditions=checks,
      baseline_correct_count=sum(prev),candidate_correct_count=sum(new),lost=lost,baseline_mean_abs_error_seconds=old_mean,candidate_mean_abs_error_seconds=new_mean,
      freeze_sha256=sha(fp),report_sha256=sha(rp),review_binding_sha256=sha(bp),evaluator_sha256=sha(EVALUATOR),
      review_file_sha256s=e['review_file_sha256s'],S2=None)

def package(run_dir):
    target=ARTIFACTS/'submit_v6.zip';manifest=ARTIFACTS/'submit_v6.manifest.json';extract=ARTIFACTS/'verify_v6'
    require(not any(p.exists() for p in (target,manifest,extract)),'No overwrite of V6 output')
    expected,bindings,gate=fresh_gate(run_dir) # Missing/failed real gate exits before any writes.
    require(sha(HELPER)==HELPER_SHA,'Pure archive helper changed')
    spec=importlib.util.spec_from_file_location('approved_archive_helpers',HELPER);helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
    member_path=helper.member_path
    require({p.relative_to(PACKAGE).as_posix() for p in PACKAGE.rglob('*') if p.is_file()}==set(expected),'V5 must contain exactly its 46 approved files')
    for name,h in expected.items():
        p=member_path(PACKAGE,name);require(sha(p)==h,'V5 asset changed: '+name);add(bindings,p,h)
    modified=helper.modified_inference((PACKAGE/'inference.py').read_bytes())
    sources={name:member_path(PACKAGE,name) for name in expected if name!='inference.py'};sources[MEMBER]=CANDIDATE
    rows=[dict(path=n,bytes=p.stat().st_size,sha256=sha(p)) for n,p in sources.items()]
    rows.append(dict(path='inference.py',bytes=len(modified),sha256=hashlib.sha256(modified).hexdigest()));rows.sort(key=lambda x:x['path'])
    by_name={r['path']:r for r in rows}
    require(len(rows)==47 and set(by_name)==set(expected)|{MEMBER} and MEMBER not in expected,'Only approved 47 members allowed')
    require(all(by_name[n]['sha256']==h for n,h in expected.items() if n!='inference.py') and by_name[MEMBER]['sha256']==C_SHA,'Unapproved byte changes')
    require({n.split('/')[0] for n in by_name}=={'inference.py','requirements.txt','model'},'Invalid archive root')
    size=sum(r['bytes'] for r in rows);require(size<32_000_000_000,'32 GB uncompressed limit exceeded')
    for name in by_name:member_path(extract,name)
    verify(bindings)
    # First artifact write is below: all actual gate/source/evidence checks passed.
    with zipfile.ZipFile(target,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=1,allowZip64=True) as z:
        for name in sorted(by_name):
            if name=='inference.py':z.writestr(name,modified)
            else:z.write(sources[name],name)
    require(target.stat().st_size<10_000_000_000,'10 GB compressed limit exceeded')
    with zipfile.ZipFile(target) as z:
        require(z.testzip() is None,'ZIP CRC failed');infos=z.infolist();names=[i.filename for i in infos]
        require(len(names)==47 and len(set(names))==47 and set(names)==set(by_name) and sum(i.file_size for i in infos)==size,'ZIP inventory/size differs')
        for info in infos:
            member_path(extract,info.filename)
            require(not info.is_dir() and (info.external_attr>>16)&0o170000!=0o120000 and info.file_size==by_name[info.filename]['bytes'],'Unsafe/type/size ZIP member')
            with z.open(info) as f:require(hashlib.file_digest(f,'sha256').hexdigest()==by_name[info.filename]['sha256'],'ZIP member SHA differs')
        extract.mkdir()
        for info in infos:
            dest=member_path(extract,info.filename);dest.parent.mkdir(parents=True,exist_ok=True)
            with z.open(info) as src,dest.open('xb') as dst:shutil.copyfileobj(src,dst,1024*1024)
            require(dest.stat().st_size==by_name[info.filename]['bytes'] and sha(dest)==by_name[info.filename]['sha256'],'Extraction SHA differs')
    require({p.relative_to(extract).as_posix() for p in extract.rglob('*') if p.is_file()}==set(by_name),'Extracted inventory differs')
    require({p.name for p in extract.iterdir()}=={'model','inference.py','requirements.txt'},'Extracted root differs')
    verify(bindings)
    record=dict(zip=str(target),sha256=sha(target),zip_bytes=target.stat().st_size,uncompressed_bytes=size,files=rows,
      validation_evaluation_sha256=gate['evaluation_sha256'],validation=gate,created_utc=datetime.now(timezone.utc).isoformat(),
      packager_sha256=sha(__file__),helper_path=str(HELPER),helper_sha256=HELPER_SHA,evidence_file_sha256s=bindings,
      v5_selection_sha256=sha(SELECTION),extraction=dict(path=str(extract),all_file_hashes_verified=True,zip_crc_passed=True),
      allowed_changes=['inference.py docstring and Stage2 import',MEMBER],models_requirements_licenses_and_other_stage_bytes_unchanged=True,
      submission_allowed=False,offline_inference_validated=False,
      limitations='Integrity packaging only. Actual extracted ZIP offline/resource/source-overlap review and explicit submission process remain separate. No official S2 claim.')
    with manifest.open('x',encoding='utf8') as f:json.dump(record,f,ensure_ascii=False,indent=2,allow_nan=False)
    print(json.dumps(dict(status='packaged_fresh_gate_verified',zip=str(target),manifest=str(manifest),files=47)))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--validation-run',type=Path,default=HERE/'run');a=p.parse_args();package(a.validation_run.resolve())
