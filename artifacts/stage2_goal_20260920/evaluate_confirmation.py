"""Frozen interval-aware paired evaluation, provider labels kept as a separate reference."""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
EPSILON = 0.3
ROUNDING_TOLERANCE = 1e-9


def read(path):
    return json.loads(path.read_text())


def grade(prediction, lower, upper):
    assert lower <= upper
    low = max(lower - prediction, 0.0, prediction - upper)
    high = max(abs(prediction - lower), abs(prediction - upper))
    result = 'correct' if high <= EPSILON + ROUNDING_TOLERANCE else 'wrong' if low > EPSILON + ROUNDING_TOLERANCE else 'indeterminate'
    return dict(result=result, error_lower_seconds=low, error_upper_seconds=high)


def aggregate(rows):
    n = len(rows)
    counts = {kind: sum(row['result'] == kind for row in rows) for kind in ['correct', 'wrong', 'indeterminate']}
    return dict(n=n, **counts, accuracy_bounds=None if not n else [counts['correct'] / n, (counts['correct'] + counts['indeterminate']) / n],
                mae_bounds_seconds=None if not n else [sum(row['error_lower_seconds'] for row in rows) / n,
                                                       sum(row['error_upper_seconds'] for row in rows) / n])


def category_metric(pairs, classes):
    matrix = [[sum(target == a and pred == b for target, pred in pairs) for b in classes] for a in classes]
    f1 = []
    for k in range(len(classes)):
        tp = matrix[k][k]
        denominator = sum(matrix[k]) + sum(row[k] for row in matrix)
        f1.append(2 * tp / denominator if denominator else 0.0)
    return dict(n=len(pairs), classes=classes, confusion=matrix,
                macro_f1=sum(f1) / len(f1) if pairs else None, zero_division=0)


def paired_accuracy_delta(baseline, candidate, lower, upper):
    # Both predictions share one unknown true time. Evaluate all constant pieces and their closed endpoints.
    radius = EPSILON + ROUNDING_TOLERANCE
    cuts = sorted({lower, upper, *[x for p in [baseline, candidate] for x in [p - radius, p + radius] if lower <= x <= upper]})
    samples = cuts + [(a + b) / 2 for a, b in zip(cuts, cuts[1:])]
    changes = [int(candidate - radius <= t <= candidate + radius) - int(baseline - radius <= t <= baseline + radius) for t in samples]
    return [min(changes), max(changes)]


assert grade(1.0, 0.7, 1.3)['result'] == 'correct'
assert grade(0.0, 0.301, 0.4)['result'] == 'wrong'
assert grade(0.0, 0.2, 0.4)['result'] == 'indeterminate'
assert grade(0.0, 0.3, 0.3)['result'] == 'correct'
assert aggregate([grade(0.0, 0.2, 0.4)])['accuracy_bounds'] == [0.0, 1.0]
assert paired_accuracy_delta(0.0, 0.0, 0.2, 0.4) == [0, 0]
assert paired_accuracy_delta(0.0, 1.0, 0.9, 1.1) == [1, 1]
assert paired_accuracy_delta(0.0, 1.0, 0.0, 1.0) == [-1, 1]


if __name__ == '__main__':
    report = read(HERE / 'mac_run/report.json')
    assert report['status'] == 'complete'
    annotations = {row['ID']: row for row in read(HERE / 'ccd_adjudication/records.json')['records']}
    sources = {row['ID']: row for row in read(HERE / 'ccd_intake/inputs.json')}
    provider = {row['ID']: row for row in read(HERE / 'ccd_intake/selection.json')['selected']}
    rows = []
    for record in report['videos']:
        sid = record['ID']
        source, labels = sources[sid], annotations[sid]
        times = {img['frame']: img['pts_seconds'] for img in source['images']}
        base, candidate, diagnostic = (record[k] for k in ['baseline_prediction', 'candidate_prediction', 'diagnostics'])
        row = dict(ID=sid, source_group=source['source_group'], baseline=base, candidate=candidate,
                   collision_changed=base['collision_frame'] != candidate['collision_frame'],
                   other_three_unchanged=all(base[k] == candidate[k] for k in ['entry_frame', 'entry_side', 'evasion_space']),
                   contact_evaluation=None, entry_evaluation=None, collision_candidates=diagnostic['collision_candidates'],
                   entry_candidates=diagnostic['entry_candidates'], internal_vlm_collision_frame=diagnostic['collision']['collision_frame'],
                   entry_after_final_collision=times[candidate['entry_frame']] > times[candidate['collision_frame']],
                   original_times={name: {k: times[pred[k]] for k in ['collision_frame', 'entry_frame']}
                                   for name, pred in [('baseline', base), ('candidate', candidate)]})
        for field, output_key in [('collision', 'collision_frame'), ('entry', 'entry_frame')]:
            truth = labels[field]
            if truth.get('evaluation_eligible', False):
                lo, hi = times[truth['lower_frame']], times[truth['upper_frame']]
                scores = {name: grade(times[pred[output_key]], lo, hi) for name, pred in [('baseline', base), ('candidate', candidate)]}
                scores.update(interval_seconds=[lo, hi], label_kind=truth['status'], evidence_level=truth.get('evidence_level'),
                              definite_gain=scores['baseline']['result'] == 'wrong' and scores['candidate']['result'] == 'correct',
                              definite_loss=scores['baseline']['result'] == 'correct' and scores['candidate']['result'] == 'wrong')
                scores['paired_accuracy_delta_bounds'] = paired_accuracy_delta(times[base[output_key]], times[candidate[output_key]], lo, hi)
                choices = row['collision_candidates'] if field == 'collision' else row['entry_candidates']
                scores['any_presented_definitely_correct'] = any(grade(times[f], lo, hi)['result'] == 'correct' for f in choices)
                scores['any_presented_possibly_correct'] = any(grade(times[f], lo, hi)['result'] != 'wrong' for f in choices)
                if field == 'entry':
                    scores['any_original_definitely_correct'] = any(grade(t, lo, hi)['result'] == 'correct' for t in times.values())
                    scores['any_before_internal_contact_definitely_correct'] = any(grade(t, lo, hi)['result'] == 'correct' for f,t in times.items() if f <= row['internal_vlm_collision_frame'])
                row['contact_evaluation' if field == 'collision' else 'entry_evaluation'] = scores
        row['category_labels'] = {key: labels[key].get('value') if labels[key].get('evaluation_eligible', False) else None
                                  for key in ['entry_side', 'evasion_space']}
        reference_frame = provider[sid]['labels'].index(1)
        row['provider_reference_only'] = dict(first_positive_frame=reference_frame, time_seconds=times[reference_frame],
                 baseline=grade(times[base['collision_frame']], times[reference_frame], times[reference_frame]),
                 candidate=grade(times[candidate['collision_frame']], times[reference_frame], times[reference_frame]),
                 not_DACON_ground_truth=True)
        rows.append(row)
    fields = {}
    for name in ['contact_evaluation', 'entry_evaluation']:
        eligible = [r for r in rows if r[name] is not None]
        fields[name] = dict(eligible_ids=[r['ID'] for r in eligible],
                           baseline=aggregate([r[name]['baseline'] for r in eligible]),
                           candidate=aggregate([r[name]['candidate'] for r in eligible]),
                           definite_gain_ids=[r['ID'] for r in eligible if r[name]['definite_gain']],
                           definite_loss_ids=[r['ID'] for r in eligible if r[name]['definite_loss']])
        fields[name]['transitions'] = {a + '->' + b: sum(r[name]['baseline']['result'] == a and r[name]['candidate']['result'] == b for r in eligible)
                                      for a in ['correct', 'wrong', 'indeterminate'] for b in ['correct', 'wrong', 'indeterminate']}
        fields[name]['paired_accuracy_delta_bounds'] = [sum(r[name]['paired_accuracy_delta_bounds'][k] for r in eligible) / len(eligible) for k in [0, 1]] if eligible else None
    contact = fields['contact_evaluation']
    categories = {key: {name: category_metric([(row['category_labels'][key], row[name][key]) for row in rows
                        if row['category_labels'][key] is not None], classes)
                        for name in ['baseline', 'candidate']}
                  for key, classes in [('entry_side', ['LEFT', 'RIGHT']), ('evasion_space', [0, 1])]}
    result = dict(status='complete', official_S2=None, epsilon_seconds=EPSILON, rounding_tolerance=ROUNDING_TOLERANCE,
                  source_cases=len(rows), model_calls=sum(len(r['calls']) for r in report['videos']),
                  evaluation=fields, categorical_evaluation=categories, collision_changed_ids=[r['ID'] for r in rows if r['collision_changed']],
                  other_three_unchanged=sum(r['other_three_unchanged'] for r in rows),
                  contact_confirmation_gate_pass=bool(contact['definite_gain_ids']) and not contact['definite_loss_ids'],
                  provider_reference_metrics={name: aggregate([r['provider_reference_only'][name] for r in rows]) for name in ['baseline', 'candidate']},
                  rows=rows)
    (HERE / 'evaluation.json').write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(json.dumps({k:v for k,v in result.items() if k != 'rows'}, ensure_ascii=False, indent=2))
