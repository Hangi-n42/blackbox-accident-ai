"""Compare frozen AI review intervals with saved runs; not official accuracy."""
import json
from pathlib import Path

BASE = Path(__file__).resolve().parents[2] / 'artifacts/data_pilot_20260916/nexar'

def read(path):
    return json.loads(path.read_text())

refs = {}
for name in ['expert_a', 'expert_a_round2', 'expert_b', 'expert_b_round2']:
    review = read(BASE / f'{name}.json')
    for case in review.get('cases', review.get('records', [])):
        if case['actual_ego_contact']['value'] is not True:
            continue
        if 'contact' in case:
            ref = dict(contact=case['contact']['time_interval_sec'], entry=case['entry']['time_interval_sec'], side=case['side']['value'], space=case['ego_space']['value'])
        else:
            entry = case.get('entry_uncertainty')
            ref = dict(contact=case['collision_uncertainty']['time_interval_s'], entry=entry['time_interval_s'] if entry else None, side=case['entry_side']['value'], space=case['evasion_space']['value'])
        refs[case['id']] = dict(ref, source=f'{name}.json')
rows = []
for folder in ['baseline_run_metal', 'temporal_run_metal', 'baseline_round2_metal', 'temporal_round2_metal', 'final_contact_metal']:
    path = BASE / folder / 'stage2_report.json'
    if not path.exists():
        continue
    report = read(path)
    if report['status'] != 'complete':
        raise RuntimeError(f'Run is incomplete: {folder}')
    for video in report['videos']:
        ident = video['ID']; ref = refs[ident]; pred = video['prediction']; pts = read(BASE / f'{ident}.pts.json')
        row = dict(run=folder, ID=ident, prediction=pred, reference=ref, times={})
        for field, column in [('contact', 'collision_frame'), ('entry', 'entry_frame')]:
            if ref[field] is None:
                continue
            lo, hi = ref[field]; time = pts[pred[column]]
            minimum = max(lo-time, 0, time-hi); maximum = max(abs(time-lo), abs(time-hi))
            row['times'][field] = dict(predicted_sec=time, min_error_sec=minimum, max_error_sec=maximum, tolerance_result='inside_all' if maximum<=.3+1e-9 else 'outside_all' if minimum>.3+1e-9 else 'interval_dependent')
        row['side_match'] = None if ref['side'] is None else ref['side']==pred['entry_side']
        row['space_match'] = None if ref['space'] is None else ref['space']==pred['evasion_space']
        rows.append(row)
output = dict(reference_type='AI inferred intervals; not official GT or independent test', cases=rows)
(BASE / 'six_case_comparison.json').write_text(json.dumps(output, ensure_ascii=False, indent=2))
for row in rows:
    print(row['run'], row['ID'], row['times'], 'side', row['side_match'], 'space', row['space_match'])
