"""CPU-only frozen V5 trace versus two provisional human development drafts.

No model imports/inference; refuses every prediction cohort except 00000/00003.
Run after parent finishes baseline: python -I -B compare_nexar_dev.py
  --run-dir <baseline output> --output <new diagnostic.json>
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
IDS = ('00000', '00003')
INTAKE = ROOT/'research/v6_stage2/nexar_review_intake_integrity.json'
INTAKE_SHA = '23c9a7d4413e184eebd8550dd8b2c1fb1462b1d6a4916b95963485f64b90f1fe'
RUNNER = ROOT/'research/v6_stage2/run_nexar_baseline.py'


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def numbers(values, time_map, context):
    require(isinstance(values, list), f'{context}: frame list missing')
    require(all(type(n) is int and n in time_map for n in values), f'{context}: invalid original frame')
    require(len(values) == len(set(values)), f'{context}: duplicate frame')
    return values


def coverage(values, gt_frame, gt_seconds, time_map, applicable=True, reason=None):
    """Distance is measured against this human draft, not confirmed GT accuracy."""
    values = numbers(values, time_map, 'coverage')
    nearest = min(values, key=lambda n: (abs(time_map[n]-gt_seconds), time_map[n], n)) if values else None
    error = None if nearest is None else abs(time_map[nearest]-gt_seconds)
    return dict(policy_applicable=applicable, nonapplicable_reason=reason, count=len(values), frames=values,
                exact_draft_frame_included=gt_frame in values, nearest_frame=nearest,
                nearest_pts_seconds=None if nearest is None else time_map[nearest],
                nearest_abs_error_seconds=error,
                within_0_3_seconds=False if error is None else error <= .3 + 1e-12,
                tolerance_roundoff_epsilon_seconds=1e-12)


def singleton(value, time_map):
    require(type(value) is int and value in time_map, 'Selected frame absent from bound source')
    return [value]


def compare_video(video, trace, intake, run_dir):
    ID = video['ID']
    review_path = Path(intake['review_path'])
    require(sha(review_path) == intake['sha256']['review'], f'{ID}: reviewed draft changed')
    review = read(review_path)
    require(review['ID'] == f'NEXAR_REVIEW_{ID}' and review['record_type'] == 'human_review_draft', 'Wrong draft')
    require(review['evaluation_eligible'] is False, 'Draft promotion needs a different protocol')
    source = Path(video['source_path'])
    source_sha = sha(source)
    require(source_sha == video['source_sha256'] == intake['sha256']['source'] == review['source_video_sha256'], 'Source SHA mismatch')
    case = Path(intake['case_path'])
    require(sha(case) == intake['sha256']['case'], 'Review case changed after integrity audit')
    require(review['frame_mapping_sha256'] == intake['sha256']['JS_JSON_stringify_full_case_frames'], 'UI mapping binding mismatch')
    time_map = {r['frame']: r['pts_seconds'] for r in video['source_frame_pts']}
    require(len(time_map) == len(video['source_frame_pts']) == video['source_frame_count'], 'Duplicate source frames')
    require(all(type(n) is int and math.isfinite(t) for n,t in time_map.items()), 'Invalid native mapping')
    require(list(time_map) == list(range(len(time_map))), 'Source decode numbering is not zero based contiguous')
    require(all(b > a for a,b in zip(time_map.values(),list(time_map.values())[1:])), 'Native PTS not increasing')
    audited = intake['full_native_mapping']
    require(len(audited) == len(time_map), 'Intake/source length mismatch')
    for row, audit_row in zip(video['source_frame_pts'], audited):
        require(row['frame'] == audit_row['frame'] and row['pts_seconds'] == audit_row['pts_seconds']
                and row['native_pts'] == audit_row['native_pts']
                and row['time_base'] == [audit_row['time_base_numerator'], audit_row['time_base_denominator']], 'Native PTS binding differs')
    images = video['input_images']
    encoded_sha = hashlib.sha256(json.dumps(images, sort_keys=True).encode()).hexdigest()
    require(encoded_sha == video['input_manifest_sha256'] == trace['input_manifest_sha256'], 'Input manifest binding mismatch')
    for image in images:
        path = (run_dir/image['path']).resolve()
        require(path.is_relative_to(run_dir.resolve()), 'Input path escapes run directory')
        require(sha(path) == image['file_sha256'], 'Sampled PNG changed')
        require(image['pts_seconds'] == time_map[image['frame']], 'Sample PTS mismatch')
    sampled = numbers([r['frame'] for r in images], time_map, 'sampled')
    require(sampled == trace['original_frame_numbers'], 'Trace input ordering mismatch')
    expected_pts = [dict(frame=n, pts_seconds=time_map[n]) for n in sampled]
    require(expected_pts == video['selected_frame_pts'] == trace['selected_frame_pts'], 'Selected PTS mismatch')
    valid = numbers(trace['frame_numbers'], time_map, 'decodevalid')
    require(valid and valid == [n for n in sampled if n in set(valid)], 'Decodevalid not ordered subset')
    calls = trace['calls']
    require(len(calls) == 4 and all(c['status'] == 'complete' for c in calls), 'Incomplete four-call trace')
    require([c['number'] for c in calls] == [1,2,3,4], 'Unexpected call ordering')
    offered = [numbers(c['offered_original_numbers'], time_map, 'offered') for c in calls]
    require(all(set(o).issubset(valid) for o in offered), 'Offered candidates not decodevalid')
    internal = trace['internal_collision_frame']; final = trace['final_motion_collision_frame']
    prediction = trace['prediction']; diag = trace['diagnostics']
    require(final == prediction['collision_frame'] == diag['collision_replacement']['collision_frame'], 'Final collision mismatch')
    require(internal == diag['collision_replacement']['base_collision_frame'] and internal in valid, 'Internal collision mismatch')
    scores = trace['motion_scores']
    require(len(scores) == len(valid) and all(math.isfinite(x) for x in scores), 'Invalid motion scores')
    require(final == valid[max(range(len(scores)),key=lambda i:scores[i])], 'Final collision is not recorded motion argmax')
    prefix = valid[:valid.index(internal)+1]
    require(prefix == trace['entry_prefix'], 'Entry prefix inconsistent with internal VLM collision')
    require(offered[1] == diag['collision_candidates'] and offered[2] == diag['entry_candidates'], 'Candidate trace/diagnostics mismatch')
    require(set(offered[2]).issubset(prefix), 'Entry candidates outside policy prefix')
    output = {}
    for field in ('contact', 'entry'):
        gt = review['review'][field]
        require(gt['status'] == 'observed' and type(gt['frame']) is int and gt['frame'] in time_map, 'Draft event not observed/bound')
        require(math.isfinite(gt['pts_seconds']) and abs(time_map[gt['frame']]-gt['pts_seconds']) < 1e-12, 'Draft event PTS mismatch')
        cov = lambda values, **kw: coverage(values, gt['frame'], gt['pts_seconds'], time_map, **kw)
        stages = dict(source_full_native_pts=cov(list(time_map)), sampled_10hz=cov(sampled), decode_valid=cov(valid),
                      internal_collision_entry_prefix=cov(prefix, applicable=field=='entry', reason=None if field=='entry' else 'Contact policy does not restrict by entry prefix; shown only as context diagnostic'))
        if field == 'contact':
            stages.update(coarse_vlm_offered=cov(offered[0]), fine_vlm_offered=cov(offered[1]),
                          internal_vlm_selected=cov(singleton(internal,time_map)),
                          final_motion_selected=cov(singleton(final,time_map)))
            chain = ['source_full_native_pts','sampled_10hz','decode_valid','fine_vlm_offered','internal_vlm_selected']
            selected = final
        else:
            stages.update(entry_candidates=cov(offered[2]), selected=cov(singleton(prediction['entry_frame'],time_map)))
            chain = ['source_full_native_pts','sampled_10hz','decode_valid','internal_collision_entry_prefix','entry_candidates','selected']
            selected = prediction['entry_frame']
        transitions = [dict(previous=a,next=b,exact_frame_lost=stages[a]['exact_draft_frame_included'] and not stages[b]['exact_draft_frame_included'],
                            tolerance_coverage_lost=stages[a]['within_0_3_seconds'] and not stages[b]['within_0_3_seconds']) for a,b in zip(chain,chain[1:])]
        output[field] = dict(human_draft=gt, stages=stages, transitions=transitions,
                             submitted_selected_frame=selected, submitted_selected_pts=time_map[selected],
                             submitted_signed_error_seconds=time_map[selected]-gt['pts_seconds'],
                             caveat='Conditional diagnostic against one unadjudicated human draft; not official Accuracy@0.3s')
    side = review['review'].get('side')
    space = review['review'].get('space')
    return dict(ID=ID,source_sha256=source_sha,review_sha256=sha(review_path),case_sha256=sha(case),
                input_manifest_sha256=encoded_sha, temporal=output,
                side=dict(human_draft=side,prediction=prediction['entry_side'],comparable=side in ('LEFT','RIGHT'),
                          agreement=None if side not in ('LEFT','RIGHT') else side == prediction['entry_side']),
                space=dict(human_draft=space,prediction=prediction['evasion_space'],comparable=space in ('0','1',0,1),
                           agreement=None if space not in ('0','1',0,1) else int(space)==prediction['evasion_space']),
                parsed_model_diagnostics=diag,
                observed_selected_vs_default_caveat='Selected frame is the policy output. Parser/default provenance is not independently recreated here; parsed raw fields are preserved for audit.',
                selected_membership=dict(internal_contact_in_fine_candidates=internal in offered[1],entry_in_candidates=prediction['entry_frame'] in offered[2]),
                final_motion_is_separate_branch=True,
                missing_sampled_due_to_decode=[n for n in sampled if n not in valid],
                source_group_as_supplied=review['source_group_id'], annotation_blinded_self_report=review['review'].get('annotation_blinded'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), 'Output exists: refuses overwrite')
    require(sha(INTAKE) == INTAKE_SHA, 'Intake integrity artifact changed')
    # Intake holds three annotations, but only the two declared development records are selected.
    # No reserved-case prediction is opened or analyzed anywhere in this tool.
    intake = {r['ID']: r for r in read(INTAKE)['results'] if r['ID'] in {f'NEXAR_REVIEW_{i}' for i in IDS}}
    freeze_path=args.run_dir/'freeze.json'; report_path=args.run_dir/'report.json'
    frozen=read(freeze_path)
    require([r['ID'] for r in frozen['videos']] == list(IDS) and frozen['policy']['ids'] == list(IDS), 'Only 00000/00003 input cohort allowed')
    report=read(report_path)
    require([r['ID'] for r in report['videos']] == list(IDS), 'Only 00000/00003 prediction cohort allowed')
    require(report['status']=='complete' and report['call_count']==8 and report['network_attempts']==0, 'Baseline incomplete/offline contract failed')
    require(report['freeze_sha256']==sha(freeze_path), 'Report/freeze mismatch')
    require(frozen['binding']['runner_sha256']==sha(RUNNER), 'Runner source changed after freeze')
    selection=ROOT/'artifacts/submissions/v5_selection_frozen.json'
    require(sha(selection)==frozen['binding']['selection_sha256'], 'V5 selection changed')
    selected=read(selection)
    require(selected['stage2_module']=='stage2_motion_collision', 'Wrong frozen policy')
    expected={k:v for k,v in selected['expected_archive_sha256'].items() if k.startswith('model/stage2/') or k=='inference.py'}
    require(expected==frozen['binding']['files'], 'Frozen package manifests differ')
    require(frozen['policy']['candidate'] is False and frozen['policy']['sampling_hz']==10, 'Unexpected candidate/sampling policy')
    if frozen.get('protocol_artifact'):
        p=frozen['protocol_artifact'];require(sha(p['path'])==p['sha256'], 'Protocol changed after freeze')
    results=[compare_video(v,t,intake[f'NEXAR_REVIEW_{v["ID"]}'],args.run_dir) for v,t in zip(frozen['videos'],report['videos'])]
    result=dict(created_utc=datetime.now(timezone.utc).isoformat(),status='COMPLETE_DEVELOPMENT_DIAGNOSTIC',
                binding=dict(script_sha256=sha(__file__),freeze_sha256=sha(freeze_path),baseline_report_sha256=sha(report_path),
                             intake_sha256=INTAKE_SHA,runner_sha256=sha(RUNNER),v5_selection_sha256=sha(selection),
                             package_weight_files_rehashed_here=False),
                videos=results,S2=None,official_accuracy=None,independent_validation=False,gt_promotion=False,
                limitations=['Two single-human drafts only; source-group independence pending and no adjudication.',
                             '00003 side is missing and remains unscored. No full-cohort S2 or model adoption claim.',
                             'Contact motion output and internal VLM output are separate branches; internal collision still limits entry candidates.',
                             'No reserved00007 prediction read; no model inference or prediction changes.'])
    with args.output.open('x',encoding='utf-8') as stream:
        json.dump(result,stream,ensure_ascii=False,indent=2,allow_nan=False)
    print(json.dumps(dict(status=result['status'],ids=list(IDS),S2=None,output=str(args.output))))


if __name__ == '__main__':
    main()
