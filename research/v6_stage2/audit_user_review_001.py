"""Compare an actual user's development draft with frozen V5/archived policy.

This is a diagnostic, not promotion to adjudicated GT or a submission gate.
"""
import csv
from datetime import datetime, timezone
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import av

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).parent
USER_FILE = Path('C:/Users/dsl/Downloads/PUBLIC_DEV_001_review_1789397594082.json')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def js_numbers(value):
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, list):
        return [js_numbers(v) for v in value]
    if isinstance(value, dict):
        return {k: js_numbers(v) for k, v in value.items()}
    return value


def main():
    review = read(USER_FILE)
    case_path = ROOT / 'research/v6_review_tool/dist/cases/PUBLIC_DEV_001/case.json'
    case = read(case_path)
    trace_path = OUT / 'archived_v3_jpeg_trace.json'
    trace = read(trace_path)
    row = next(v for v in trace['videos'] if v['ID'] == 'S2_001')
    source = Path(case['video_path'])
    assert review['ID'] == case['ID'] == 'PUBLIC_DEV_001'
    assert review['record_type'] == 'human_review_draft'
    assert review['source_video_sha256'] == case['video_sha256'] == sha(source)
    frames_json = json.dumps(js_numbers(case['frames']), ensure_ascii=False, separators=(',', ':'))
    assert hashlib.sha256(frames_json.encode()).hexdigest() == review['frame_mapping_sha256']
    with av.open(str(source)) as container:
        native = [float(f.pts * f.time_base) for f in container.decode(video=0)]
    assert len(native) == len(case['frames']) == 50
    for number, f in enumerate(case['frames']):
        assert f['frame'] == number and f['pts_seconds'] == native[number]
        assert sha(ROOT / 'research/v6_review_tool/dist' / f['image']) == f['sha256']
    marks = review['review']
    for key in ['contact', 'entry']:
        mark = marks[key]
        assert mark['status'] == 'observed' and native[mark['frame']] == mark['pts_seconds']
    package = ROOT / 'artifacts/submissions/verify_v5/model/stage2/code/solution'
    code_hashes = {}
    for name in ['stage2.py', 'stage2_v2.py', 'stage2_motion_collision.py', 'vlm.py', 'vlm_candidate.py']:
        code_hashes[name] = sha(package / name)
        assert code_hashes[name] == trace['provenance']['recorded_sources']['solution/' + name]
    csv_path = ROOT / 'artifacts/submissions/verify_v5_results/stage2.csv'
    with csv_path.open(encoding='utf-8-sig', newline='') as handle:
        actual = next(r for r in csv.DictReader(handle) if r['ID'] == 'S2_001')
    prediction = {k: v if k == 'entry_side' else int(v) for k, v in actual.items() if k != 'ID'}
    assert prediction == row['prediction']
    pts = {i: Decimal(str(t)) for i, t in enumerate(native)}
    def coverage(numbers, target):
        errors = [abs(pts[n] - pts[target]) for n in numbers]
        return {'count': len(numbers), 'exact_frame_present': target in numbers,
                'nearest_error_seconds': float(min(errors)) if errors else None,
                'within_0_3_seconds': bool(errors) and min(errors) <= Decimal('0.3')}
    def error(pred, target):
        delta = pts[pred] - pts[target]
        return {'frame': pred, 'pts_seconds': float(pts[pred]), 'signed_error_seconds': float(delta),
                'within_0_3_seconds': abs(delta) <= Decimal('0.3')}
    contact, entry = marks['contact']['frame'], marks['entry']['frame']
    public_gt_path = OUT / 'public_official_partial_gt.json'
    public = next(v for v in read(public_gt_path)['videos'] if v['ID'] == 'S2_001')
    official = public['labels']['collision_time_seconds']['value']
    result = {'created_at': datetime.now(timezone.utc).isoformat(),
        'scope': 'single_user_development_draft_diagnostic_not_adjudicated_GT_or_independent_validation',
        'evaluation_eligible': False, 'complete_gt': False, 'S2': None,
        'source_binding': {'review_sha256': sha(USER_FILE), 'case_sha256': sha(case_path),
            'video_sha256': sha(source), 'trace_sha256': sha(trace_path), 'v5_output_csv_sha256': sha(csv_path),
            'actual_v5_code_hashes_equal_archived_policy': code_hashes,
            'v5_output_matches_archived_prediction': True,
            'trace_is_actual_v5_server_run': False, 'native_pts_all_50_match': True,
            'review_frame_mapping_hash_match': True, 'all_50_rendered_frame_hashes_match': True},
        'human_review': marks, 'missing_fields': ['evasion_space'],
        'not_supplied': ['independent_second_review', 'adjudication_record'],
        'official_contact_seconds': official, 'user_contact_seconds': float(pts[contact]),
        'user_minus_official_contact_seconds': float(pts[contact] - Decimal(str(official))),
        'collision': {'all_original': coverage(row['original_frame_numbers'], contact),
            'vlm_candidates': coverage(row['diagnostics']['collision_candidates'], contact),
            'vlm_selection_vs_user': error(row['internal_collision_frame'], contact),
            'final_motion_selection_vs_user': error(prediction['collision_frame'], contact),
            'final_motion_selection_vs_official': {'signed_error_seconds': float(pts[prediction['collision_frame']] - Decimal(str(official))),
                                                  'within_0_3_seconds': abs(pts[prediction['collision_frame']] - Decimal(str(official))) <= Decimal('0.3')},
            'diagnosis': 'No missing candidate or timing failure at tolerance on this reviewed clip; final motion replacement delays VLM selection by 0.2 seconds.'},
        'entry': {'all_original': coverage(row['original_frame_numbers'], entry),
            'decoded_valid': coverage(row['valid_frame_numbers'], entry),
            'after_collision_prefix': coverage(row['entry_prefix'], entry),
            'offered_candidates': coverage(row['entry_candidates'], entry),
            'selection_vs_user': error(prediction['entry_frame'], entry),
            'diagnosis': 'No prefix truncation, sampling omission or selection failure relative to this draft.'},
        'direction': {'human': marks['side'], 'prediction': prediction['entry_side'],
            'match': marks['side'] == prediction['entry_side'],
            'coarse_raw': row['diagnostics']['coarse']['entry_side'], 'invalid_output_fallback_used': False,
            'diagnosis': 'Valid but disagreeing coarse direction is carried to final output; later calls do not reassess direction.',
            'target_identity_failure_proven': False},
        'space': {'human': None, 'prediction': prediction['evasion_space'], 'correctness': None},
        'limits': ['One exposed development clip cannot establish the dominant error rate.',
                   'Human collision differs from official label; official label remains unchanged.',
                   'A supplied annotator name or blinded checkbox is not independent identity verification.',
                   'No candidate model or new accuracy claim is produced.']}
    inbox = OUT / 'user_reviews'
    inbox.mkdir(exist_ok=True)
    archived = inbox / USER_FILE.name
    if archived.exists():
        assert sha(archived) == sha(USER_FILE)
    else:
        archived.write_bytes(USER_FILE.read_bytes())
    target = OUT / 'user_review_001_diagnostic.json'
    if target.exists():
        raise RuntimeError('Existing diagnostic; preserve previous evidence before rerun.')
    target.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'status': 'PASS', 'source_frames': len(native), 'missing': result['missing_fields'],
                      'entry_exact_match': True, 'direction_match': result['direction']['match'], 'S2': None}))


if __name__ == '__main__':
    main()
