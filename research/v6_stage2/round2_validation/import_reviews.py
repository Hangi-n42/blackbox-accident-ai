"""Import three actual user drafts; no model inference or synthetic real binding.

python -I -B import_reviews.py --review FILE0 FILE1 FILE2 --output NEW_INTAKE_DIR
Uses the fixed source freeze at round2_validation/run. Uncertain contacts are
preserved with an incomplete intake and never create a run review binding.
"""
import argparse
from datetime import datetime,timezone
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import subprocess
import sys

sys.dont_write_bytecode=True
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[name]='2'
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
RUN=HERE/'run'
RUNNER=HERE/'run_validation.py'
PROTOCOL=HERE/'protocol.json'
IDS=['00008','00010','00013']


def require(ok,message):
    if not ok:raise ValueError(message)


def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def put(path,obj):
    with Path(path).open('x',encoding='utf-8') as stream:json.dump(obj,stream,ensure_ascii=False,indent=2,allow_nan=False)


def js_frames_sha(case_path):
    code="const f=require('fs'),c=require('crypto'),x=JSON.parse(f.readFileSync(process.argv[1],'utf8'));process.stdout.write(c.createHash('sha256').update(JSON.stringify(x.frames),'utf8').digest('hex'));"
    return subprocess.run(['node','-e',code,str(case_path)],capture_output=True,text=True,check=True).stdout.strip()


def preflight(run_dir,output):
    require(not output.exists(),'Intake output exists; no overwrite')
    require((run_dir/'freeze.json').is_file(),'Source freeze required')
    require(not any((run_dir/name).exists() for name in ('review_binding.json','report.json','traces')),
            'Binding or inference artifacts already exist; prereview import refused')


def select_reviews(paths):
    require(len(paths)==3 and len({p.resolve() for p in paths})==3,'Exactly three distinct actual review files required')
    selected={}
    for path in paths:
        value=read(path)
        key=value.get('ID')
        require(key in {f'NEXAR_REVIEW_{ID}' for ID in IDS},'Wrong review ID; no substitution')
        ID=key.removeprefix('NEXAR_REVIEW_')
        require(ID not in selected,'Duplicate review ID')
        selected[ID]=path
    require(set(selected)==set(IDS),'All three fixed IDs required')
    return selected


def inspect_review(path,record):
    """Independent native decode and selected-PNG audit; no writes, no GT promotion."""
    import av
    import numpy as np
    from PIL import Image
    errors=[]
    checks={}
    def check(name,ok):
        checks[name]=bool(ok)
        if not ok:errors.append(name)
    value=read(path);ID=record['ID'];data=record['input']
    cp=Path(record['case_path']);mp=Path(record['mapping_path'])
    case=read(cp);mapping=read(mp)
    check('case_sha_bound',sha(cp)==record['case_sha256'])
    check('mapping_file_sha_bound',sha(mp)==record['mapping_sha256'])
    check('review_ID_matches_fixed_case',value.get('ID')==case.get('ID')==f'NEXAR_REVIEW_{ID}')
    check('single_human_draft_not_GT',value.get('record_type')=='human_review_draft' and value.get('evaluation_eligible') is False)
    review=value.get('review',{})
    check('review_object',isinstance(review,dict))
    if not isinstance(review,dict):review={}
    check('annotator_present',isinstance(review.get('annotator'),str) and bool(review['annotator'].strip()))
    check('annotation_blinded_self_report',review.get('annotation_blinded') is True)
    source=Path(data['source_path']);source_before=sha(source)
    check('source_SHA_all_equal',source_before==data['source_sha256']==case.get('video_sha256')==mapping.get('source_video_sha256')==value.get('source_video_sha256'))
    mapping_hash=js_frames_sha(cp)
    check('full_JS_case_frames_hash',mapping_hash==value.get('frame_mapping_sha256'))
    check('native_time_origin_equal',value.get('time_origin')==case.get('time_origin')==data.get('time_origin'))
    check('source_group_matches_case',value.get('source_group_id')==case.get('source_group_id'))
    check('source_uri_matches_case',value.get('source_uri')==case.get('source_uri'))
    ui=case['frames'];times=data['source_frame_pts']
    check('mapping_equals_frozen_native_times',mapping['frame_pts']==times==[dict(frame=f['frame'],pts_seconds=f['pts_seconds']) for f in ui])
    check('zero_based_full_frames',[f['frame'] for f in ui]==list(range(len(ui))))
    expected_images=data['input_images']
    check('PNG_count_matches',len(expected_images)==len(ui))
    for item,frame in zip(expected_images,ui):
        image=(Path(record['input_root'])/item['path']).resolve()
        safe=image.is_relative_to(Path(record['input_root']).resolve())
        check(f"PNG_{item['frame']}_SHA",safe and image.is_file() and sha(image)==item['file_sha256']==frame['sha256'])
    contact=review.get('contact',{})
    if not isinstance(contact,dict):contact={}
    status=contact.get('status','missing')
    selected=contact.get('frame');observed=status=='observed'
    if observed:
        check('observed_contact_frame_valid',type(selected) is int and 0<=selected<len(ui))
        pts=contact.get('pts_seconds')
        check('observed_contact_seconds_numeric',type(pts) in (int,float) and math.isfinite(pts))
    native=[];pixel=None
    with av.open(str(source)) as container:
        container.streams.video[0].thread_count=2
        for index,frame in enumerate(container.decode(video=0)):
            if frame.pts is None or frame.time_base is None:
                check('native_PTS_available',False);continue
            seconds=float(frame.pts*frame.time_base)
            native.append(dict(frame=index,native_pts=frame.pts,time_base_numerator=frame.time_base.numerator,
                               time_base_denominator=frame.time_base.denominator,pts_seconds=seconds))
            if observed and type(selected) is int and index==selected and index<len(expected_images):
                image=(Path(record['input_root'])/expected_images[index]['path']).resolve()
                if image.is_file():
                    try:
                        with Image.open(image) as im:rgb=np.asarray(im.convert('RGB'))
                        pixel=dict(frame=index,path=str(image),sha256=sha(image),
                                   source_rgb_equal=np.array_equal(frame.to_ndarray(format='rgb24'),rgb))
                    except (OSError,ValueError):pixel=dict(frame=index,path=str(image),source_rgb_equal=False)
    native_match=len(native)==len(times) and all(n['frame']==t['frame'] and n['pts_seconds']==t['pts_seconds'] for n,t in zip(native,times))
    check('full_independent_native_PTS_match',native_match)
    selected_native=next((n for n in native if type(selected) is int and n['frame']==selected),None)
    if observed:
        check('contact_native_PTS_exact',selected_native is not None and contact.get('pts_seconds')==selected_native['pts_seconds'])
        check('selected_contact_PNG_pixels_equal',pixel is not None and pixel.get('source_rgb_equal') is True)
    check('source_bytes_unchanged',sha(source)==source_before)
    eligible=observed and not errors
    return dict(ID=ID,review_path=str(path.resolve()),review_sha256=sha(path),source_sha256=source_before,
                frame_mapping_sha256=mapping_hash,contact_status=status,native_pts_validated=eligible and native_match,
                selected_png_validated=eligible and pixel is not None and pixel.get('source_rgb_equal') is True,
                eligible_for_contact_binding=eligible,integrity_passed=not errors,checks=checks,errors=errors,
                full_native_mapping=native,selected_contact_native=selected_native,selected_contact_png=pixel,
                unresolved=[] if observed else ['Contact missing/uncertain/non-observed: preserved; no exact label inferred'],
                unknown_other_fields_allowed=True,annotation_blinded_is_self_report=True,ground_truth_promotion=False)


def import_actual(paths,output):
    preflight(RUN,output)
    require(output.resolve().is_relative_to(ROOT.resolve()),'Output must stay in workspace')
    selected=select_reviews(paths)
    spec=importlib.util.spec_from_file_location('_round2_import_binding_runner',RUNNER)
    runner=importlib.util.module_from_spec(spec);spec.loader.exec_module(runner)
    frozen=runner.verify(RUN)  # Full source/package/PNG hashes, no model load.
    require([r['ID'] for r in frozen['videos']]==IDS,'Frozen IDs differ')
    source_hashes={str(p.resolve()):sha(p) for p in paths}
    validator_sha=sha(__file__);freeze_sha=sha(RUN/'freeze.json');protocol_sha=sha(PROTOCOL)
    output.mkdir();copies=output/'user_reviews';copies.mkdir()
    results=[]
    for record in frozen['videos']:
        source=selected[record['ID']]
        # Include fixed ID directory to avoid colliding source basenames without renaming bytes.
        folder=copies/record['ID'];folder.mkdir();destination=folder/source.name
        with source.open('rb') as original,destination.open('xb') as saved:
            import shutil
            shutil.copyfileobj(original,saved)
        require(sha(destination)==source_hashes[str(source.resolve())],'Original changed while copying')
        item=inspect_review(destination,record)
        item['original_download_path']=str(source.resolve());item['original_copy_bytes_identical']=True
        results.append(item)
    require(all(sha(p)==h for p,h in source_hashes.items()),'Actual review changed during validation')
    require(runner.verify(RUN)==frozen,'Frozen inputs changed during validation')
    require(sha(__file__)==validator_sha and sha(PROTOCOL)==protocol_sha and sha(RUN/'freeze.json')==freeze_sha,'Validation code/protocol changed')
    require(not any((RUN/name).exists() for name in ('review_binding.json','report.json','traces')),'Inference/binding appeared during import')
    eligible=all(r['eligible_for_contact_binding'] for r in results)
    report_path=output/'integrity.json'
    report=dict(created_utc=datetime.now(timezone.utc).isoformat(),status='VALIDATED_ALL_CONTACTS' if eligible else 'INCOMPLETE_OR_INVALID_CONTACTS',
                freeze_sha256=freeze_sha,protocol_sha256=protocol_sha,validator_path=str(Path(__file__).resolve()),validator_sha256=validator_sha,
                results=results,actual_files_only=True,ground_truth_promotion=False,predictions_seen_before_binding=False,
                scope='File/PTS/selected-pixel integrity of actual single-human drafts; event semantics not independently adjudicated',
                binding_will_be_written=eligible)
    put(report_path,report)
    if eligible:
        keys=('ID','review_path','review_sha256','source_sha256','frame_mapping_sha256','contact_status','native_pts_validated','selected_png_validated')
        binding=dict(status='VALIDATED_CONTACT_BINDING',ground_truth_promotion=False,predictions_seen_before_binding=False,
                     created_utc=datetime.now(timezone.utc).isoformat(),freeze_sha256=freeze_sha,protocol_sha256=protocol_sha,
                     validator=dict(path=str(Path(__file__).resolve()),sha256=validator_sha),
                     integrity_report=dict(path=str(report_path.resolve()),sha256=sha(report_path)),
                     reviews=[{k:r[k] for k in keys} for r in results])
        # Final artifact write only after all actual files validate; no composite GT is created.
        put(RUN/'review_binding.json',binding)
    print(json.dumps(dict(status=report['status'],binding_written=eligible,ids=IDS)))
    return 0 if eligible else 2


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--review',nargs=3,type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    raise SystemExit(import_actual(a.review,a.output.resolve()))
