"""Conditional V6C packager. Parent executes only after both frozen gates pass.

Creates submit_v6.zip, submit_v6.manifest.json, and a new verify_v6 extraction.
No model inference, upload, existing-file overwrite, or cleanup is performed.
"""
import argparse
import ast
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import shutil
import zipfile

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
ARTIFACTS = ROOT/'artifacts/submissions'
PACKAGE = ARTIFACTS/'verify_v5'
SELECTION = ARTIFACTS/'v5_selection_frozen.json'
PROTOCOL = HERE/'uncapped_jerk_v6c_protocol.json'
RUNNER = HERE/'run_uncapped_jerk_v6c.py'
CANDIDATE = HERE/'candidates/stage2_uncapped_jerk_v6c.py'
CANDIDATE_MEMBER = 'model/stage2/code/solution/stage2_uncapped_jerk_v6c.py'
IDS = {'development':['00000','00003','00004'], 'reserved':['00005','00006','00007']}
EPS = 1e-12


def require(value, message):
    if not value:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def member_path(root, name):
    require(isinstance(name,str) and '\\' not in name, 'Noncanonical ZIP member name')
    posix=PurePosixPath(name)
    require(not posix.is_absolute() and posix.as_posix()==name and all(x not in ('','..','.') and ':' not in x for x in posix.parts), 'Unsafe ZIP member path')
    target=(root/Path(*posix.parts)).resolve()
    require(target.is_relative_to(root.resolve()) and target!=root.resolve(), 'ZIP member escapes destination')
    return target


def gate_binding(phase, verified_files):
    folder=HERE/f'uncapped_jerk_v6c_{phase}'
    ep,fp,rp=folder/'evaluation.json',folder/'freeze.json',folder/'report.json'
    require(all(p.is_file() for p in (ep,fp,rp)), f'{phase}: real gate/freeze/report required before packaging')
    e,f,r=read(ep),read(fp),read(rp)
    require(e['gate_passed'] is True and e['phase']==f['phase']==r['phase']==phase, f'{phase}: gate not passed')
    require(e['freeze_sha256']==sha(fp)==r['freeze_sha256'] and e['report_sha256']==sha(rp), f'{phase}: result binding changed')
    evaluator=Path(e['evaluator_path']).resolve()
    require(evaluator.is_relative_to(ROOT.resolve()) and sha(evaluator)==e['evaluator_sha256'], 'Evaluator binding changed')
    require(e['protocol_sha256']==sha(PROTOCOL), 'Evaluator protocol changed')
    require([x['ID'] for x in e['videos']]==[x['ID'] for x in f['videos']]==[x['ID'] for x in r['videos']]==IDS[phase], 'Fixed cohort mismatch')
    expected_calls=0 if phase=='development' else 12
    require(r['status']=='complete' and r['network_attempts']==0 and r['call_count']==r['max_calls']==f['max_calls']==expected_calls, 'Incomplete/offline/call contract failed')
    for path,digest in f['files'].items():
        # Cache only confirmed bytes during this packaging invocation.
        actual=verified_files.setdefault(path,sha(path)) if path not in verified_files else verified_files[path]
        require(actual==digest, f'Frozen dependency changed: {path}')
    for path in (RUNNER,CANDIDATE,PROTOCOL):
        require(f['files'][str(path)]==sha(path), f'{phase}: candidate/runner/protocol not bound')
    require(e['review_file_sha256s'], 'Evaluation lacks review bindings')
    for path,digest in e['review_file_sha256s'].items():
        require(sha(path)==digest, 'Human review changed')
    old_errors,new_errors=[],[]
    for evaluated,record,trace in zip(e['videos'],f['videos'],r['videos']):
        require(evaluated['baseline']==trace['baseline'] and evaluated['candidate']==trace['candidate'], 'Evaluation predictions differ from trace')
        require(evaluated['other_fields_identical'] is True, 'Evaluation reports noncollision changes')
        require(len(trace['calls'])==(0 if phase=='development' else 4), 'Per-file calls differ')
        if phase=='development':require(trace['base_scores_equal_frozen_V5'] is True, 'V5 scores were not exactly reproduced')
        else:
            require(all(x['status']=='complete' for x in trace['calls']) and [x['max_new_tokens'] for x in trace['calls']]==[64,48,40,40], 'Four-call token contract differs')
        valid=trace['frame_numbers'];source=record['input']
        times={x['frame']:x['pts_seconds'] for x in source['source_frame_pts']}
        require(valid and len(valid)==len(set(valid)) and all(type(n) is int and n in times for n in valid), 'Invalid original frame numbers')
        review_path=Path(evaluated['review_path'])
        require(str(review_path) in e['review_file_sha256s'] and sha(review_path)==evaluated['review_sha256'], 'Per-video review binding missing')
        review=read(review_path);contact=review['review']['contact']
        require(contact['status']=='observed' and type(contact['frame']) is int and contact['frame'] in times and contact['pts_seconds']==times[contact['frame']], 'Known native-PTS contact reference required')
        require(sha(source['source_path'])==source['source_sha256']==evaluated['source_sha256']==review['source_video_sha256'], 'Original source changed')
        old,new=trace['baseline'],trace['candidate']
        for prediction in (old,new):
            require(all(type(prediction[k]) is int and prediction[k] in valid for k in ('collision_frame','entry_frame')), 'Output frame not in valid originals')
            require(prediction['entry_side'] in ('LEFT','RIGHT') and type(prediction['evasion_space']) is int and prediction['evasion_space'] in (0,1), 'Invalid category')
        require(all(old[k]==new[k] for k in ('entry_frame','entry_side','evasion_space')), 'Other output fields changed')
        a=abs(times[old['collision_frame']]-contact['pts_seconds']);b=abs(times[new['collision_frame']]-contact['pts_seconds'])
        require(math.isfinite(a) and math.isfinite(b), 'Nonfinite timing error')
        require(abs(a-evaluated['baseline_abs_error_seconds'])<=EPS and abs(b-evaluated['candidate_abs_error_seconds'])<=EPS,'Evaluation error arithmetic differs')
        old_errors.append(a);new_errors.append(b)
    previous=[x<=.3+EPS for x in old_errors];current=[x<=.3+EPS for x in new_errors]
    require(sum(current)-sum(previous)>=1 and not any(a and not b for a,b in zip(previous,current)), 'Correct-count/loss gate recomputation failed')
    require(sum(new_errors)/3<=sum(old_errors)/3+EPS, 'Mean-error gate recomputation failed')
    return dict(folder=str(folder),evaluation_sha256=sha(ep),freeze_sha256=sha(fp),report_sha256=sha(rp),
                evaluator_path=str(evaluator),evaluator_sha256=sha(evaluator),protocol_sha256=sha(PROTOCOL),
                review_file_sha256s=e['review_file_sha256s'],candidate_sha256=sha(CANDIDATE),runner_sha256=sha(RUNNER),
                baseline_correct_count=sum(previous),candidate_correct_count=sum(current),
                baseline_mean_abs_error_seconds=sum(old_errors)/3,candidate_mean_abs_error_seconds=sum(new_errors)/3)


def modified_inference(original):
    before=ast.parse(original.decode('utf-8'))
    text=original
    old=b'from solution.stage2_motion_collision import predict_stage2 as predict'
    new=b'from solution.stage2_uncapped_jerk_v6c import predict_stage2 as predict'
    require(text.count(old)==1,'Expected exactly one original Stage2 import')
    text=text.replace(old,new)
    old_doc=b'"""DACON236753 V5: V3 Stage2 policy, guarded decode, equivalent Stage3 arithmetic."""'
    new_doc=b'"""DACON236753 V6: uncapped jerk collision, guarded decode, equivalent Stage3 arithmetic."""'
    require(text.count(old_doc)==1,'Unexpected original inference docstring')
    text=text.replace(old_doc,new_doc)
    after=ast.parse(text.decode('utf-8'))
    after.body[0]=before.body[0]
    for node in ast.walk(after):
        if isinstance(node,ast.ImportFrom) and node.module=='solution.stage2_uncapped_jerk_v6c':node.module='solution.stage2_motion_collision'
    require(ast.dump(before)==ast.dump(after),'Unexpected inference change beyond docstring and Stage2 import')
    return text


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    target=ARTIFACTS/'submit_v6.zip';manifest_path=ARTIFACTS/'submit_v6.manifest.json';extract=ARTIFACTS/'verify_v6'
    require(all(not p.exists() for p in (target,manifest_path,extract)), 'V6 output exists: no overwrite or merge')
    require(extract.resolve().is_relative_to(ROOT.resolve()), 'Extraction target outside workspace')
    verified={}
    dev=gate_binding('development',verified)
    reserved=gate_binding('reserved',verified)
    require(dev['evaluator_sha256']==reserved['evaluator_sha256'], 'Different evaluators across phases')
    rf=read(HERE/'uncapped_jerk_v6c_reserved/freeze.json')
    for filename,field in [('evaluation.json','evaluation_sha256'),('freeze.json','freeze_sha256'),('report.json','report_sha256')]:
        require(rf['files'][str(HERE/'uncapped_jerk_v6c_development'/filename)]==dev[field], 'Reserved phase not bound to passed development')
    require(rf['files'][dev['evaluator_path']]==dev['evaluator_sha256'], 'Reserved phase did not freeze evaluator')
    protocol=read(PROTOCOL)
    expected_gate=dict(all_three_known_contact_labels_required=True,minimum_additional_correct_within_0_3_seconds=1,
                       maximum_previously_correct_cases_lost=0,mean_absolute_contact_time_error_must_not_increase=True,
                       noncollision_prediction_fields_must_be_identical=True,no_network_attempts_and_valid_original_frame_outputs=True)
    require(protocol['gate_each_cohort']==expected_gate and protocol['development_ids']==IDS['development'] and protocol['reserved_ids']==IDS['reserved'],'Protocol differs from approved gate')
    selection=read(SELECTION);expected=selection['expected_archive_sha256']
    require(selection['stage1_module']=='stage1_v4' and selection['stage2_module']=='stage2_motion_collision' and selection['stage3_module']=='stage3_v5_compatible','Unexpected V5 baseline modules')
    stage2_expected={k:v for k,v in expected.items() if k.startswith('model/stage2/') or k=='inference.py'}
    for phase in IDS:
        binding=read(HERE/f'uncapped_jerk_v6c_{phase}/freeze.json')['package_binding']
        require(binding['files']==stage2_expected and binding['selection_sha256']==sha(SELECTION),'Evaluated package differs from V5 archive selection')
    actual_files={p.relative_to(PACKAGE).as_posix() for p in PACKAGE.rglob('*') if p.is_file()}
    require(actual_files==set(expected),'V5 extraction has missing or extra files')
    require(CANDIDATE_MEMBER not in expected, 'Candidate already in V5 archive')
    for name,digest in expected.items():require(sha(member_path(PACKAGE,name))==digest,f'V5 asset changed: {name}')
    modified=modified_inference((PACKAGE/'inference.py').read_bytes())
    sources={name:member_path(PACKAGE,name) for name in expected if name!='inference.py'}
    sources[CANDIDATE_MEMBER]=CANDIDATE
    rows=[dict(path=name,bytes=path.stat().st_size,sha256=sha(path)) for name,path in sources.items()]
    rows.append(dict(path='inference.py',bytes=len(modified),sha256=hashlib.sha256(modified).hexdigest()))
    rows.sort(key=lambda x:x['path']);by_name={r['path']:r for r in rows}
    require(set(by_name)==set(expected)|{CANDIDATE_MEMBER},'Archive allowlist differs')
    require(all(by_name[n]['sha256']==digest for n,digest in expected.items() if n!='inference.py'),'Unapproved V5 asset byte change')
    size=sum(row['bytes'] for row in rows);require(size<32_000_000_000,'Uncompressed archive exceeds 32 GB')
    require({n.split('/')[0] for n in by_name}=={'model','inference.py','requirements.txt'},'Archive root must have exactly three entries')
    # All gates and source verification above occur before the first artifact write.
    with zipfile.ZipFile(target,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=1,allowZip64=True) as archive:
        for name in sorted(by_name):
            if name=='inference.py':archive.writestr(name,modified)
            else:archive.write(sources[name],name)
    require(target.stat().st_size<10_000_000_000,'Compressed archive exceeds 10 GB')
    with zipfile.ZipFile(target) as archive:
        require(archive.testzip() is None,'ZIP CRC test failed')
        infos=archive.infolist();names=[i.filename for i in infos]
        require(len(names)==len(set(names)) and set(names)==set(by_name),'ZIP duplicate/extra/missing members')
        require({n.split('/')[0] for n in names}=={'model','inference.py','requirements.txt'},'Invalid ZIP root')
        require(sum(i.file_size for i in infos)==size,'ZIP uncompressed size mismatch')
        for info in infos:
            member_path(extract,info.filename)
            require(not info.is_dir() and (info.external_attr>>16)&0o170000!=0o120000,'ZIP directory or symlink member disallowed')
        extract.mkdir()
        for info in infos:
            destination=member_path(extract,info.filename);destination.parent.mkdir(parents=True,exist_ok=True)
            with archive.open(info) as src,destination.open('xb') as dst:shutil.copyfileobj(src,dst,1024*1024)
            require(destination.stat().st_size==by_name[info.filename]['bytes'] and sha(destination)==by_name[info.filename]['sha256'],'Extracted bytes differ')
    require({p.relative_to(extract).as_posix() for p in extract.rglob('*') if p.is_file()}==set(by_name),'Extracted file inventory differs')
    require({p.name for p in extract.iterdir()}=={'model','inference.py','requirements.txt'},'Extracted root differs')
    # Recheck evidence after potentially long compression; failure preserves artifacts for audit.
    require(sha(CANDIDATE)==dev['candidate_sha256'] and sha(PROTOCOL)==dev['protocol_sha256'],'Candidate/protocol changed during packaging')
    for phase,binding in [('development',dev),('reserved',reserved)]:
        folder=Path(binding['folder'])
        for filename,field in [('evaluation.json','evaluation_sha256'),('freeze.json','freeze_sha256'),('report.json','report_sha256')]:require(sha(folder/filename)==binding[field],'Gate artifacts changed during packaging')
    record=dict(zip=str(target),sha256=sha(target),zip_bytes=target.stat().st_size,uncompressed_bytes=size,files=rows,
                created_utc=datetime.now(timezone.utc).isoformat(),packager_sha256=sha(__file__),
                selection_basis=dict(v5_selection_sha256=sha(SELECTION),v5_expected_file_count=len(expected),development=dev,reserved=reserved),
                extraction=dict(path=str(extract),all_file_hashes_verified=True,zip_crc_passed=True,root_entries=['inference.py','model','requirements.txt']),
                allowed_changes=['inference.py docstring and Stage2 import',CANDIDATE_MEMBER],
                models_requirements_licenses_and_other_stage_bytes_unchanged=True,
                limitations='Packaging integrity only. No offline inference, runtime validation, upload, or official score claim performed by this tool.')
    with manifest_path.open('x',encoding='utf-8') as stream:json.dump(record,stream,ensure_ascii=False,indent=2,allow_nan=False)
    print(json.dumps(dict(status='packaged_and_hash_verified',zip=str(target),manifest=str(manifest_path),sha256=record['sha256'],files=len(rows))))


if __name__=='__main__':main()
