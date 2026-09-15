"""Secondary inferred-interval consistency only; no accuracy/adoption gate."""
import hashlib
import json
from fractions import Fraction
from pathlib import Path

BASE = Path(__file__).resolve().parent

def read(p):
    return json.loads(p.read_text(encoding='utf-8-sig'))

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def classify(p, a, b):
    if max(abs(p-a), abs(p-b)) <= Fraction(3, 10):
        return 'within_tolerance_for_entire_conditional_interval'
    if max(a-p, p-b, Fraction(0)) > Fraction(3, 10):
        return 'outside_tolerance_of_conditional_interval'
    return 'indeterminate'

def main():
    adjudication = BASE / 'next_review_adjudication.json'
    predictions = BASE / 'next_motion_diagnostic/report.json'
    output = BASE / 'next_motion_diagnostic/conditional_consistency.json'
    assert not output.exists()
    refs = {r['ID']: r for r in read(adjudication)['records']}
    results = read(predictions)
    rows = []
    for p in results['videos']:
        r = refs[p['ID']]
        assert p['source_sha256'] == r['source_sha256']
        times = read(BASE / f"next_review_native_qa/{p['ID']}.frames.json")['frames']
        row = {k: p[k] for k in ['ID', 'v6_decoded_index', 'corrected_decoded_index', 'prediction_changed']}
        interval = r['contact_interval']
        row['conditional_reference'] = interval
        for name in ['v6', 'corrected']:
            point = times[p[f'{name}_decoded_index']]
            sec = Fraction(point['pts']) * Fraction(point['time_base'])
            assert abs(float(sec) - p[f'{name}_pts_seconds']) < 1e-10
            row[f'{name}_pts_seconds'] = float(sec)
            if interval is None:
                label = 'unknown_reference_not_scored'
            else:
                lo, hi = [times[n] for n in interval['frame_interval']]
                a = Fraction(lo['pts']) * Fraction(lo['time_base'])
                b = Fraction(hi['pts']) * Fraction(hi['time_base'])
                label = classify(sec, a, b)
            row[f'{name}_conditional_consistency'] = label
        rows.append(row)
    assert sorted(r['ID'] for r in rows) == sorted(refs)
    report = {
        'status': 'complete_secondary_diagnostic',
        'not_human_or_official_accuracy': True,
        'previous_human9_adoption_gate_still_failed': True,
        'adopt_candidate': False,
        'reason': 'Inferred intervals remain conditional; this diagnostic does not override failed accuracy gate or validate other fields.',
        'bindings': {str(p.relative_to(BASE)): sha(p) for p in [adjudication, predictions, BASE / 'next_review_adjudication_protocol.json', Path(__file__)]},
        'source_count': len(rows),
        'conditional_reference_count': sum(r['conditional_reference'] is not None for r in rows),
        'changed_prediction_count': sum(r['prediction_changed'] for r in rows),
        'records': rows,
    }
    with output.open('x', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
        f.write('\n')
    print(json.dumps(rows, ensure_ascii=True))

if __name__ == '__main__':
    main()
