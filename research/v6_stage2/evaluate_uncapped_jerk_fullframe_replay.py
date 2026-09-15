"""CPU-only exposed-six C replay audit; a pass permits fresh validation only."""
import argparse
from datetime import datetime,timezone
import hashlib
import importlib.util
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
DEPENDENCIES={
    'evaluate_nexar_fullframe_v5.py':'073d339e5a221f5baa86224a9f084dfea2110513b493edb18b1db7a7cece2709',
    'evaluate_uncapped_jerk_v6c.py':'7896bb095be2d3825f31d02a4423013f5b59db8269d95bca24dfe3a95a754b58'}
PROTOCOL=HERE/'uncapped_jerk_fullframe_replay_protocol.json'
PROTOCOL_SHA='10d20bfc841b47ec0f4d1a3ef18019f2efd93541086fe7b8cdf94a3a65fbc6e7'
CANDIDATE=HERE/'candidates/stage2_uncapped_jerk_v6c.py'
CANDIDATE_SHA='3e86e5117230fb4680c1d4af630cb02dd7e9aca99b7c4184d7baed13f479d64e'
IDS=['00000','00003','00004','00005','00006','00007']
GATE=dict(all_six_known_contact_references_required=True,minimum_additional_correct_within_0_3_seconds=1,
          maximum_previously_correct_cases_lost=0,mean_absolute_contact_time_error_must_not_increase=True,
          noncollision_prediction_fields_must_be_identical=True,
          native_frame_validity_and_exact_base_score_reproduction_required=True,roundoff_epsilon_seconds=1e-12)


def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def require(ok,message):
    if not ok:raise ValueError(message)


def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def helper(filename,name):
    path=HERE/filename
    require(sha(path)==DEPENDENCIES[filename],'Evaluation helper changed')
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def decision(rows,contracts):
    require(len(rows)==6 and [r['ID'] for r in rows]==IDS,'Exactly six ordered diagnostic cases required')
    require(all(r['full']['contact'] is not None and r['fullC']['contact'] is not None for r in rows),'Missing contact reference')
    before=[r['full']['contact'] for r in rows];after=[r['fullC']['contact'] for r in rows]
    correct_old=sum(x['correct'] for x in before);correct_new=sum(x['correct'] for x in after)
    lost=sum(a['correct'] and not b['correct'] for a,b in zip(before,after))
    mean_old=sum(x['abs_error_seconds'] for x in before)/6;mean_new=sum(x['abs_error_seconds'] for x in after)/6
    criteria=dict(additional_correct_at_least_one=correct_new-correct_old>=1,previous_success_losses_zero=lost==0,
                  mean_absolute_error_nonincrease=mean_new<=mean_old+1e-12,
                  other_three_fields_identical=all(r['other_fields_identical'] for r in rows),
                  exact_base_native_and_offline_contracts=bool(contracts))
    return dict(gate_advance_to_new_validation_only=all(criteria.values()),criteria=criteria,
                baseline_correct_count=correct_old,candidate_correct_count=correct_new,additional_correct_count=correct_new-correct_old,
                gained=sum(not a['correct'] and b['correct'] for a,b in zip(before,after)),lost=lost,
                baseline_mean_abs_error_seconds=mean_old,candidate_mean_abs_error_seconds=mean_new,
                adoption_allowed=False,submission_allowed=False)


def historical_C(c):
    records={};bindings={}
    for dirname in ('uncapped_jerk_v6c_development','uncapped_jerk_v6c_reserved'):
        folder=HERE/dirname;fp=folder/'freeze.json';rp=folder/'report.json'
        frozen=read(fp);report=read(rp)
        require(report['status']=='complete' and report['network_attempts']==0 and report['freeze_sha256']==sha(fp),'Historical C run incomplete/unbound')
        require(frozen['files'][str(CANDIDATE)]==CANDIDATE_SHA,'Historical C differs from fixed replay candidate')
        for p in (fp,rp):bindings[str(p)]=sha(p)
        for record,trace in zip(frozen['videos'],report['videos']):
            require(record['ID']==trace['ID'] and record['ID'] not in records,'Historical C ID mismatch')
            c.validate_scores(trace,record)
            records[record['ID']]=(record,trace)
    require(list(records)==IDS,'Incomplete historical C cohort')
    return records,bindings


def evaluate(run_dir):
    output=run_dir/'evaluation.json';require(not output.exists(),'Refuse overwrite')
    full=helper('evaluate_nexar_fullframe_v5.py','_full_evaluator')
    c=helper('evaluate_uncapped_jerk_v6c.py','_C_evaluator')
    require(sha(PROTOCOL)==PROTOCOL_SHA and sha(CANDIDATE)==CANDIDATE_SHA,'Fixed protocol/candidate changed')
    protocol=read(PROTOCOL);require(protocol['comparison_gate_on_fullframe_six']==GATE,'Gate changed')
    fp=run_dir/'freeze.json';frozen=read(fp)
    require(frozen['phase']=='exposed_six_fullframe_diagnostic' and [v['ID'] for v in frozen['videos']]==IDS,'Unexpected replay phase/cohort')
    for p,h in frozen['files'].items():require(sha(p)==h,'Frozen replay binding changed: '+p)
    require(frozen['protocol_sha256']==PROTOCOL_SHA and frozen['candidate_sha256']==CANDIDATE_SHA,'Replay policy SHA mismatch')
    base_dir=Path(frozen['base_run']);bf=base_dir/'freeze.json';br=base_dir/'report.json'
    require(frozen['files'][str(bf)]==sha(bf) and frozen['files'][str(br)]==sha(br),'Full V5 report binding missing')
    basefreeze=read(bf);basereport=read(br)
    require(basereport['status']=='complete' and basereport['call_count']==24 and basereport['network_attempts']==0 and basereport['freeze_sha256']==sha(bf),'Full V5 incomplete')
    require([v['ID'] for v in basefreeze['videos']]==[v['ID'] for v in basereport['videos']]==IDS,'Full baseline cohort mismatch')
    require(frozen['package_binding']==basefreeze['package_binding'],'Package differs from full V5')
    for p,h in basefreeze['files'].items():require(sha(p)==h,'Full baseline binding changed')
    require(frozen['previous_density_refs']==basefreeze['baseline10hz_refs'],'Old density reference map changed')
    rp=run_dir/'report.json';report=read(rp)
    require(report['status']=='complete' and report['phase']==frozen['phase'] and report['freeze_sha256']==sha(fp),'Replay report mismatch')
    require(report['max_calls']==report['call_count']==frozen['max_calls']==report['model_loads']==report['network_attempts']==0 and report['gt_read'] is False,'CPU/no-inference contract failed')
    require([v['ID'] for v in report['videos']]==IDS,'Replay result cohort mismatch')
    history,history_binding=historical_C(c);rows=[]
    for record,base_record,base_trace,trace in zip(frozen['videos'],basefreeze['videos'],basereport['videos'],report['videos']):
        require({k:v for k,v in record.items() if k!='baseline_trace'}==base_record and record['baseline_trace']==base_trace,'Cached full V5 differs')
        require(trace['calls']==[],'Unexpected replay VLM call')
        require(len(base_trace['calls'])==4 and all(x['status']=='complete' for x in base_trace['calls']) and [x['max_new_tokens'] for x in base_trace['calls']]==[64,48,40,40],'Baseline four-call policy changed')
        require(trace['input_manifest_sha256']==record['input']['input_manifest_sha256'] and trace['source_frame_pts']==record['input']['source_frame_pts'],'Replay input binding mismatch')
        score_audit=c.validate_scores(trace,record)
        row=full.evaluate_video(base_record,base_trace,basefreeze)
        times={x['frame']:x['pts_seconds'] for x in record['input']['source_frame_pts']}
        full.validate_prediction(trace['candidate'],trace['frame_numbers'])
        row['fullC']=full.measure(trace['candidate'],row['labels'],times);row['fullC_prediction']=trace['candidate']
        row['other_fields_identical']=all(trace['candidate'][k]==trace['baseline'][k] for k in ('entry_frame','entry_side','evasion_space'))
        row['score_audit']=score_audit
        old_record,old_trace=history[row['ID']]
        require(old_record['input']['source_sha256']==record['input']['source_sha256'],'Historical C source differs')
        require({x['frame']:x['pts_seconds'] for x in old_record['input']['source_frame_pts']}==times,'Historical C native PTS differs')
        require(old_trace['baseline']==row['old10hz_prediction'],'Historical 10Hz baseline differs')
        full.validate_prediction(old_trace['candidate'],old_trace['frame_numbers'])
        row['old10hzC']=full.measure(old_trace['candidate'],row['labels'],times)
        row['old10hzC_prediction']=old_trace['candidate'];rows.append(row)
    result=dict(created_utc=datetime.now(timezone.utc).isoformat(),phase=frozen['phase'],**decision(rows,True),videos=rows,
                comparison_2x2={key:full.summarize(rows,key) for key in ('old10hz','old10hzC','full','fullC')},
                historical_10hz_C_file_sha256s=history_binding,report_sha256=sha(rp),freeze_sha256=sha(fp),
                protocol_sha256=PROTOCOL_SHA,candidate_sha256=CANDIDATE_SHA,
                evaluator_path=str(Path(__file__).resolve()),evaluator_sha256=sha(__file__),
                imported_evaluator_sha256s={str(HERE/name):digest for name,digest in DEPENDENCIES.items()},
                review_file_sha256s={r['review_path']:r['review_sha256'] for r in rows},
                S2=None,official_S2=None,independent_validation=False,ground_truth_promotion=False,
                limitations=['All six sources are exposed single-human diagnostic references. Gate only permits separately frozen fresh validation.',
                             'Unknown labels remain null; no full S2. Prior C rejection is preserved.',
                             'Density changes are local PNG comparisons, not exact hidden JPEG/FPS replication.'])
    with output.open('x',encoding='utf-8') as stream:json.dump(result,stream,ensure_ascii=False,indent=2,allow_nan=False)
    print(json.dumps({k:result[k] for k in ('gate_advance_to_new_validation_only','gained','lost','adoption_allowed')}))


def self_test():
    rows=[]
    for i,ID in enumerate(IDS):
        a={'correct':i==0,'abs_error_seconds':.1 if i==0 else 1.}
        b={'correct':i<2,'abs_error_seconds':.1 if i<2 else 1.}
        rows.append(dict(ID=ID,full={'contact':a},fullC={'contact':b},other_fields_identical=True))
    assert decision(rows,True)['gate_advance_to_new_validation_only']
    assert not decision(rows,False)['gate_advance_to_new_validation_only']
    rows[0]['fullC']['contact']={'correct':False,'abs_error_seconds':.5}
    assert not decision(rows,True)['gate_advance_to_new_validation_only']
    assert decision(rows,True)['adoption_allowed'] is False
    print('Synthetic six-case advance-only gate PASS; no run reports or reviews read')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run-dir',type=Path);parser.add_argument('--self-test',action='store_true');args=parser.parse_args()
    if args.self_test:self_test()
    else:require(args.run_dir is not None,'--run-dir required');evaluate(args.run_dir.resolve())
