"""CPU evaluator for the fixed three-source test; never creates reviews or predictions."""
import argparse
from datetime import datetime,timezone
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import subprocess
import tempfile

for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='2'
HERE=Path(__file__).resolve().parent
STAGE=HERE.parent
IDS=['00008','00010','00013']
PROTOCOL=HERE/'protocol.json'
PROTOCOL_SHA='DF71F0933B1B3DACFEDEC376472AF9D4D897C7AA8498561ABA8BB0C9544164A5'.lower()
CANDIDATE=STAGE/'candidates/stage2_uncapped_jerk_v6c.py'
CANDIDATE_SHA='3e86e5117230fb4680c1d4af630cb02dd7e9aca99b7c4184d7baed13f479d64e'
HELPER=STAGE/'evaluate_uncapped_jerk_v6c.py'
HELPER_SHA='7896bb095be2d3825f31d02a4423013f5b59db8269d95bca24dfe3a95a754b58'
GATE=dict(all_three_known_observed_contact_references_required=True,minimum_additional_correct_within_0_3_seconds=1,
          maximum_previously_correct_cases_lost=0,mean_absolute_contact_time_error_must_not_increase=True,
          other_three_outputs_identical=True,exact_base_score_expression_and_offline_original_frame_contracts_required=True,
          roundoff_epsilon_seconds=1e-12)


def require(ok,message):
    if not ok:raise ValueError(message)


def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def verify_hashmap(bindings):
    for path,digest in bindings.items():require(sha(path)==digest,'Bound file changed: '+str(path))


def load_score_helper():
    require(sha(HELPER)==HELPER_SHA,'Independent score evaluator changed')
    spec=importlib.util.spec_from_file_location('_round2_score_evaluator',HELPER)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def judge(rows,contracts):
    require([r['ID'] for r in rows]==IDS,'Exactly the fixed three cases required')
    require(all(type(r[k]) in (int,float) and math.isfinite(r[k]) and r[k]>=0 for r in rows for k in ('baseline_abs_error_seconds','candidate_abs_error_seconds')),'Invalid contact errors')
    a=[r['baseline_abs_error_seconds']<=.3+1e-12 for r in rows]
    b=[r['candidate_abs_error_seconds']<=.3+1e-12 for r in rows]
    lost=sum(x and not y for x,y in zip(a,b))
    ma=sum(r['baseline_abs_error_seconds'] for r in rows)/3
    mb=sum(r['candidate_abs_error_seconds'] for r in rows)/3
    checks=dict(additional_correct_at_least_one=sum(b)-sum(a)>=1,previous_success_losses_zero=lost==0,
                mean_absolute_error_nonincrease=mb<=ma+1e-12,
                other_three_fields_identical=all(r['other_fields_identical'] for r in rows),
                exact_native_base_and_offline_contracts=bool(contracts))
    return dict(gate_passed=all(checks.values()),conditions=checks,baseline_correct_count=sum(a),candidate_correct_count=sum(b),
                additional_correct_count=sum(b)-sum(a),gained=sum(not x and y for x,y in zip(a,b)),lost=lost,
                baseline_mean_abs_error_seconds=ma,candidate_mean_abs_error_seconds=mb)


def validate_numbers(numbers,mapping):
    require(numbers and all(type(n) is int for n in numbers) and numbers==list(range(len(mapping))),'Full native 0..N-1 frame contract failed')
    require([x['frame'] for x in mapping]==numbers,'Native mapping numbering differs')
    times=[x['pts_seconds'] for x in mapping]
    require(all(type(t) in (int,float) and math.isfinite(t) for t in times) and all(b>a for a,b in zip(times,times[1:])),'Invalid native PTS')
    return dict(zip(numbers,times))


def native_check(record,integrity,contact):
    """Independently decode original PTS and the contact PNG; two codec threads."""
    import av
    import numpy as np
    from PIL import Image
    source=record['input'];mapping=source['source_frame_pts']
    audited=integrity['full_native_mapping']
    require(len(mapping)==len(audited),'Importer native mapping count differs')
    index=-1;selected=None
    with av.open(source['source_path']) as container:
        container.streams.video[0].thread_count=2
        for index,frame in enumerate(container.decode(video=0)):
            require(index<len(mapping) and frame.pts is not None and frame.time_base is not None,'Missing/excess native frames')
            native=dict(frame=index,native_pts=frame.pts,time_base_numerator=frame.time_base.numerator,
                        time_base_denominator=frame.time_base.denominator,pts_seconds=float(frame.pts*frame.time_base))
            require(all(audited[index].get(k)==v for k,v in native.items()),'Importer/native re-decode mismatch')
            require(mapping[index]['frame']==index and mapping[index]['pts_seconds']==native['pts_seconds'],'Frozen/native re-decode mismatch')
            if index==contact['frame']:
                image=source['input_images'][index]
                with Image.open(Path(record['input_root'])/image['path']) as png:
                    expected=np.asarray(png.convert('RGB'))
                actual=frame.to_ndarray(format='rgb24')
                require(np.array_equal(actual,expected),'Contact PNG differs from source decoded RGB')
                selected={**native,'rgb_sha256':hashlib.sha256(actual.tobytes()).hexdigest(),'png_sha256':image['file_sha256']}
    require(index+1==len(mapping) and selected is not None,'Incomplete native decoding/contact image')
    require(contact['pts_seconds']==selected['pts_seconds'],'Human contact/native timestamp mismatch')
    return dict(all_native_pts_independently_verified=True,selected_contact_png_rgb_verified=True,selected=selected,codec_threads=2)


def mapping_js_sha(case_path):
    script="const f=require('fs'),c=require('crypto'),x=JSON.parse(f.readFileSync(process.argv[1],'utf8'));process.stdout.write(c.createHash('sha256').update(JSON.stringify(x.frames),'utf8').digest('hex'));"
    return subprocess.run(['node','-e',script,str(case_path)],check=True,capture_output=True,text=True).stdout.strip()


def prepare_bindings(run_dir):
    fp=run_dir/'freeze.json';bp=run_dir/'review_binding.json';rp=run_dir/'report.json'
    require(all(p.is_file() for p in (fp,bp,rp)),'Actual frozen run and actual imported human binding required')
    require(sha(PROTOCOL)==PROTOCOL_SHA and sha(CANDIDATE)==CANDIDATE_SHA,'Fixed experiment policy changed')
    protocol=read(PROTOCOL);require(protocol['ids']==IDS and protocol['gate']==GATE,'Gate/cohort differs')
    frozen=read(fp);binding=read(bp)
    require(frozen['phase']=='round2_fresh_validation' and frozen['ids']==[v['ID'] for v in frozen['videos']]==IDS,'Unexpected frozen cohort')
    require(frozen['tokens_per_file']==[64,48,40,40] and frozen['model_loads']==1 and frozen['gt_read'] is False,'Frozen execution contract differs')
    require(binding['status']=='VALIDATED_CONTACT_BINDING' and binding['ground_truth_promotion'] is False and binding['predictions_seen_before_binding'] is False,'No valid prereview binding')
    require(binding['freeze_sha256']==sha(fp) and binding['protocol_sha256']==PROTOCOL_SHA,'Human binding policy mismatch')
    require([x['ID'] for x in binding['reviews']]==IDS,'Human binding cohort differs')
    hashes={str(fp):sha(fp),str(bp):sha(bp),str(rp):sha(rp),str(PROTOCOL):PROTOCOL_SHA,
            str(CANDIDATE):CANDIDATE_SHA,str(HELPER):HELPER_SHA,str(Path(__file__).resolve()):sha(__file__)}
    hashes.update(frozen['files'])
    for name in ('validator','integrity_report'):
        item=binding[name];require(sha(item['path'])==item['sha256'],'Importer/audit binding mismatch');hashes[item['path']]=item['sha256']
    package=Path(frozen['package_binding']['package']).resolve()
    for name,digest in frozen['package_binding']['files'].items():
        path=(package/name).resolve();require(path.is_relative_to(package),'Package path escapes root');hashes[str(path)]=digest
    for item,record in zip(binding['reviews'],frozen['videos']):
        require(set(item)=={'ID','review_path','review_sha256','source_sha256','frame_mapping_sha256','contact_status','native_pts_validated','selected_png_validated'},'Human binding contains unexpected fields')
        require(item['contact_status']=='observed' and item['native_pts_validated'] is True and item['selected_png_validated'] is True,'Incomplete human contact verification')
        require(item['source_sha256']==record['input']['source_sha256'],'Bound source differs')
        hashes[item['review_path']]=item['review_sha256']
        hashes[record['input']['source_path']]=record['input']['source_sha256']
        for key in ('case','mapping'):hashes[record[key+'_path']]=record[key+'_sha256']
        for image in record['input']['input_images']:
            p=(Path(record['input_root'])/image['path']).resolve()
            require(p.is_relative_to(Path(record['input_root']).resolve()) and p.name==f"frame_{image['frame']:06d}.png",'PNG path/original filename differs');hashes[str(p)]=image['file_sha256']
    verify_hashmap(hashes)
    return frozen,binding,read(rp),read(binding['integrity_report']['path']),hashes


def evaluate(run_dir):
    output=run_dir/'evaluation.json';require(not output.exists(),'Refuse evaluation overwrite')
    helper=load_score_helper()
    frozen,binding,report,integrity,hashes=prepare_bindings(run_dir)
    require(integrity['status']=='VALIDATED_ALL_CONTACTS' and integrity['freeze_sha256']==sha(run_dir/'freeze.json') and integrity['protocol_sha256']==PROTOCOL_SHA,'Incomplete importer integrity report')
    require(integrity['validator_path']==binding['validator']['path'] and integrity['validator_sha256']==binding['validator']['sha256'],'Integrity validator identity differs')
    require([x['ID'] for x in integrity['results']]==IDS,'Integrity cohort differs')
    require(report['status']=='complete' and report['phase']=='round2_fresh_validation' and report['freeze_sha256']==sha(run_dir/'freeze.json'),'Incomplete/unbound model run')
    require(report['call_count']==report['max_calls']==frozen['max_calls']==12 and report['model_loads']==frozen['model_loads']==1 and report['network_attempts']==0 and report['human_answer_contents_read'] is False,'Runtime/offline/answer-separation contract failed')
    expected_binding=dict(path=str(run_dir/'review_binding.json'),sha256=sha(run_dir/'review_binding.json'),
                          validator=binding['validator'],integrity_report=binding['integrity_report'],
                          review_file_sha256s={r['review_path']:r['review_sha256'] for r in binding['reviews']})
    require(report['review_binding']==expected_binding,'Run used a different human binding')
    require([x['ID'] for x in report['videos']]==IDS,'Prediction cohort differs')
    rows=[]
    for record,item,audit,trace in zip(frozen['videos'],binding['reviews'],integrity['results'],report['videos']):
        ID=record['ID'];source=record['input']
        require(ID==item['ID']==audit['ID']==trace['ID'],'Per-file identity mismatch')
        require(audit['integrity_passed'] is True and audit['eligible_for_contact_binding'] is True and audit['errors']==[],'Importer checks did not pass')
        for k in ('review_path','review_sha256','source_sha256','frame_mapping_sha256','contact_status','native_pts_validated','selected_png_validated'):
            require(audit[k]==item[k],'Importer result/binding mismatch: '+k)
        review=read(item['review_path'])
        require(review['ID']=='NEXAR_REVIEW_'+ID and review['record_type']=='human_review_draft' and review['evaluation_eligible'] is False,'Unexpected human reference status')
        require(review['source_video_sha256']==source['source_sha256'] and review['frame_mapping_sha256']==item['frame_mapping_sha256']==mapping_js_sha(record['case_path']),'Review source/UI mapping mismatch')
        times=validate_numbers(trace['frame_numbers'],source['source_frame_pts'])
        require([x['frame'] for x in source['input_images']]==trace['frame_numbers'],'Not all original PNGs used')
        require(all(x['pts_seconds']==times[x['frame']] for x in source['input_images']),'PNG/native PTS mismatch')
        require(hashlib.sha256(json.dumps(source['input_images'],sort_keys=True).encode()).hexdigest()==source['input_manifest_sha256'],'Input manifest digest mismatch')
        contact=review['review']['contact']  # No target description or other human labels are consumed.
        require(contact['status']=='observed' and type(contact['frame']) is int and contact['frame'] in times and contact['pts_seconds']==times[contact['frame']],'Missing/invalid observed contact')
        native=native_check(record,audit,contact)
        require(all(audit['selected_contact_native'][k]==native['selected'][k] for k in ('frame','native_pts','time_base_numerator','time_base_denominator','pts_seconds')) and audit['selected_contact_png']['source_rgb_equal'] is True,'Selected-contact importer evidence differs')
        require(len(trace['calls'])==4 and all(c['status']=='complete' and type(c.get('raw')) is str for c in trace['calls']) and [c['max_new_tokens'] for c in trace['calls']]==[64,48,40,40],'Per-file V5 call contract failed')
        require(trace['base_scores_exact_original_expression'] is True and trace['base_scores_equal_separate_original_scan'] is None,'False separate-scan provenance')
        # Adapter marks no separate cached run; validation still independently rebuilds both score expressions.
        score_audit=helper.validate_scores({**trace,'base_scores_equal_frozen_V5':None},{})
        before,after=trace['baseline'],trace['candidate']
        for pred in (before,after):
            require(set(pred)=={'collision_frame','entry_frame','entry_side','evasion_space'},'Unexpected prediction fields')
            require(all(type(pred[k]) is int and pred[k] in times for k in ('collision_frame','entry_frame')),'Invalid original frame prediction')
            require(pred['entry_side'] in ('LEFT','RIGHT') and type(pred['evasion_space']) is int and pred['evasion_space'] in (0,1),'Invalid categorical output')
        old_error=times[before['collision_frame']]-contact['pts_seconds'];new_error=times[after['collision_frame']]-contact['pts_seconds']
        rows.append(dict(ID=ID,review_path=item['review_path'],review_sha256=item['review_sha256'],human_contact_draft=contact,
                         baseline=before,candidate=after,baseline_signed_error_seconds=old_error,candidate_signed_error_seconds=new_error,
                         baseline_abs_error_seconds=abs(old_error),candidate_abs_error_seconds=abs(new_error),
                         other_fields_identical=all(before[k]==after[k] for k in ('entry_frame','entry_side','evasion_space')),
                         native_verification=native,score_audit=score_audit,base_scores_equal_separate_original_scan=None))
    result=dict(created_utc=datetime.now(timezone.utc).isoformat(),phase='round2_fresh_validation',**judge(rows,True),videos=rows,
                report_sha256=sha(run_dir/'report.json'),freeze_sha256=sha(run_dir/'freeze.json'),
                review_binding_sha256=sha(run_dir/'review_binding.json'),protocol_sha256=PROTOCOL_SHA,candidate_sha256=CANDIDATE_SHA,
                evaluator_path=str(Path(__file__).resolve()),evaluator_sha256=sha(__file__),imported_evaluator_sha256s={str(HELPER):HELPER_SHA},
                review_file_sha256s={r['review_path']:r['review_sha256'] for r in binding['reviews']},all_bound_file_sha256s=hashes,
                S2=None,official_accuracy=None,ground_truth_promotion=False,incident_device_independence_proven=False,
                adoption_eligibility='Conditional on contact gate PASS plus separate source-overlap, actual ZIP/offline equivalence and resource verification.',
                adoption_allowed=False,submission_allowed=False,
                limitations=['Single-human observed contact references, not official/adjudicated GT. Other human labels not scored.',
                             'Exact base expression was reproduced from the same extracted features, not a second original flow scan.',
                             'Native PNG input is not proven byte-identical to hidden JPEG preprocessing.'])
    verify_hashmap(hashes)  # Includes all reviews, source/PNG/model files, code and reports; no output before this succeeds.
    with output.open('x',encoding='utf-8') as f:json.dump(result,f,ensure_ascii=False,indent=2,allow_nan=False)
    print(json.dumps({k:result[k] for k in ('gate_passed','gained','lost','adoption_allowed','S2')}))


def self_test():
    def rows(a,b,same=True):
        return [dict(ID=ID,baseline_abs_error_seconds=x,candidate_abs_error_seconds=y,other_fields_identical=same) for ID,x,y in zip(IDS,a,b)]
    assert judge(rows([.3,.7,1.],[.3,.3,.9]),True)['gate_passed']
    assert not judge(rows([.3,.7,1.],[.3001,.3,.1]),True)['gate_passed']
    assert not judge(rows([.3,.7,1.],[.3,.3,2.]),True)['gate_passed']
    assert not judge(rows([.3,.7,1.],[.3,.3,.9],False),True)['gate_passed']
    assert not judge(rows([.3,.7,1.],[.3,.3,.9]),False)['gate_passed']
    try:judge(rows([.1],[.1]),True)
    except ValueError:pass
    else:raise AssertionError('Incomplete cohort accepted')
    assert validate_numbers([0,1],[{'frame':0,'pts_seconds':0.},{'frame':1,'pts_seconds':.07}])[1]==.07
    try:validate_numbers([0,2],[{'frame':0,'pts_seconds':0.},{'frame':2,'pts_seconds':.07}])
    except ValueError:pass
    else:raise AssertionError('Sparse input accepted')
    helper=load_score_helper();helper.self_test()
    with tempfile.TemporaryDirectory(prefix='round2_evaluator_fixture_') as folder:
        p=Path(folder)/'synthetic.txt';p.write_text('synthetic only',encoding='utf-8');binding={str(p):sha(p)};verify_hashmap(binding)
        p.write_text('tampered synthetic',encoding='utf-8')
        try:verify_hashmap(binding)
        except ValueError:pass
        else:raise AssertionError('Bound-file tamper accepted')
    print('Round2 synthetic gate/frame/hash contracts PASS; no actual reviews, report, source or GPU accessed')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run-dir',type=Path);parser.add_argument('--self-test',action='store_true');args=parser.parse_args()
    if args.self_test:self_test()
    else:require(args.run_dir is not None,'--run-dir required');evaluate(args.run_dir.resolve())
