"""Apply the predeclared contact-only advancement rule after all predictions."""
import argparse
import hashlib
import json
from pathlib import Path

def read(p):
    return json.loads(p.read_text(encoding='utf-8-sig'))

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    run = parser.parse_args().run
    report_path, evaluation_path = run / 'report.json', run / 'evaluation.json'
    report, evaluation = read(report_path), read(evaluation_path)
    assert report['status'] == 'complete' and evaluation['report_sha256'] == sha(report_path)
    reviews = {r['ID']: r for r in evaluation['videos']}
    rows = []
    for result in report['videos']:
        e = reviews[result['ID']]
        draft = read(Path(e['review_path']))
        assert sha(Path(e['review_path'])) == e['review_sha256']
        reference = draft['review']['contact']['pts_seconds']
        times = {p['frame']: p['pts_seconds'] for p in result['frame_pts']}
        offered_in_tolerance = [n for n in result['offered_frames'] if abs(times[n] - reference) <= .3 + 1e-12]
        old_error = times[result['original_internal_frame']] - reference
        errors = e['errors']['contact']
        rows.append({'ID': result['ID'], 'offered_in_tolerance': offered_in_tolerance,
                     'original_internal_error_seconds': old_error,
                     'baseline_error_seconds': errors['baseline'],
                     'candidate_error_seconds': errors['candidate'],
                     'accepted_offered_frame': result['accepted_offered_frame'],
                     'fallback': result['fallback'],
                     'candidate_correct': abs(errors['candidate']) <= .3 + 1e-12})
    transition = evaluation['transitions']['contact']
    checks = {
        'all_nine_complete': len(rows) == report['call_count'] == 9 and report['model_loads'] == 1,
        'offline_unchanged_bindings': report['network_attempts'] == 0 and report['bindings_unchanged'],
        'other_three_cached_fields_identical': all(all(r['baseline'][k] == r['candidate'][k] for k in ('entry_frame', 'entry_side', 'evasion_space')) for r in report['videos']),
        'contact_accuracy_strictly_increases': evaluation['scores']['candidate']['contact']['accuracy'] > evaluation['scores']['baseline']['contact']['accuracy'],
        'at_least_one_gained': transition['gained'] >= 1,
        'no_correct_lost': transition['lost'] == 0,
        'wall_budget': report['seconds'] <= 600,
    }
    assessment = {
        'gate_passed': all(checks.values()), 'checks': checks,
        'report_sha256': sha(report_path), 'evaluation_sha256': sha(evaluation_path),
        'source_role': 'exposed_development', 'official_accuracy': None,
        'original_internal_correct': sum(abs(r['original_internal_error_seconds']) <= .3 + 1e-12 for r in rows),
        'offered_coverage_count': sum(bool(r['offered_in_tolerance']) for r in rows),
        'fallback_count': sum(r['fallback'] for r in rows),
        'fallback_correct_count': sum(r['fallback'] and r['candidate_correct'] for r in rows),
        'direct_offered_correct_count': sum(r['accepted_offered_frame'] and r['candidate_correct'] for r in rows),
        'videos': rows, 'adopted': False, 'submission_approved': False,
        'decision': 'Advance to separately frozen full pipeline and independent evidence only' if all(checks.values()) else 'Failed frozen gate: stop this native-video hypothesis without tuning or packaging',
    }
    with (run / 'assessment.json').open('x', encoding='utf-8') as f:
        json.dump(assessment, f, ensure_ascii=False, indent=2)
    print(json.dumps(assessment, indent=2))

if __name__ == '__main__':
    main()
