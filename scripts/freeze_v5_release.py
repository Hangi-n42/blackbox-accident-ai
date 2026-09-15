"""Freeze the reviewed safe V5 combination after all required candidate gates."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib, json

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'artifacts/submissions'

def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def main():
    target=OUT/'v5_selection_frozen.json'
    if target.exists():raise FileExistsError('Selection already frozen')
    audit_path=ROOT/'research/v5_stage3/compatible_864/audit_report.json'
    audit=json.loads(audit_path.read_text(encoding='utf-8'))
    assert audit['status']=='compatible864_contract_and_runtime_passed_root_review'
    assert audit['protected_unchanged'] and audit['prior_negative_reports_preserved']
    plan=json.loads((audit_path.parent/'plan_frozen.json').read_text(encoding='utf-8'))
    assert sha(ROOT/'solution/stage3_v5_compatible.py')==plan['source_sha256']
    s2_path=ROOT/'research/v5_stage2/decision.json'
    s2=json.loads(s2_path.read_text(encoding='utf-8'))
    assert not s2['ROI']['adopt'] and not s2['SIMPLE']['adopt']
    assert s2['additional_search_stopped'] and not s2['external_DADA_inference_performed']
    assert sha(ROOT/s2['restoration_selected_source'])==s2['restoration_selected_source_sha256']
    assert sha(ROOT/'solution/stage1_v4.py')=='624fc836bc83b09cd273204206ae5b5f0de57ab19bc2b0a428e2957ef08d5c54'
    baseline_path=OUT/'submit_v3_motion_fast.manifest.json'
    old=json.loads(baseline_path.read_text(encoding='utf-8'))
    assert old['sha256']=='2a955dc4681d5817b33835af5136a1551b406a1ab11a668e804bc88677513c4c'
    expected={r['path']:r['sha256'] for r in old['files']}
    expected['inference.py']=sha(ROOT/'inference_v5.py')
    added=['model/stage2/code/solution/stage1_v4.py','model/stage2/code/solution/stage3_v5_compatible.py']
    for name in added:
        assert name not in expected
        expected[name]=sha(ROOT/'solution'/Path(name).name)
    sources=['inference_v5.py','requirements_nf4.txt','scripts/build_submission.py',
             'scripts/verify_v5_package.py','scripts/verify_submission.py','scripts/freeze_v5_release.py']
    sources+=['solution/'+Path(name).name for name in expected if name.startswith('model/stage2/code/solution/')]
    selection={
        'status':'selected_for_packaging','created_utc':datetime.now(timezone.utc).isoformat(),
        'stage1_module':'stage1_v4','stage2_module':'stage2_motion_collision','stage3_module':'stage3_v5_compatible',
        'reason':'Reject unsupported new accuracy candidates, restore V3 Stage2, retain guarded decode and adopt verified output-equivalent864 compute optimization.',
        'accuracy_claim':'No new best-score improvement established. V4 regression removal and output-equivalent speed optimization only; label scarcity remains.',
        'no_new_weights':True,'external_images_in_package':False,'external_stage2_inference_performed':False,
        'baseline_manifest_sha256':sha(baseline_path),'expected_archive_sha256':expected,
        'allowed_added_paths':added,'allowed_changed_paths':['inference.py'],'allowed_removed_paths':[],
        'unchanged_output_stages':[1,2,3],
        'source_sha256':{p:sha(ROOT/p) for p in sorted(set(sources))},
        'evidence_sha256':{str(p.relative_to(ROOT)).replace('\\','/'):sha(p) for p in [audit_path,s2_path,
            ROOT/'research/v5_stage1/experiment_report.json',
            ROOT/'research/v5_external/dada/blind_review/decision_external_not_run.json']}}
    target.write_text(json.dumps(selection,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'selection':str(target),'sha256':sha(target),'archive_files':len(expected)},ensure_ascii=False))

if __name__=='__main__':main()
