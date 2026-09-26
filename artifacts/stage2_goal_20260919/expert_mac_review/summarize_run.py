"""Summarize paired outputs using original PTS; does not manufacture accuracy labels."""
from datetime import datetime
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent


def read(path):
    return json.loads(path.read_text())


report = read(HERE / 'mac_run/report.json')
lock = read(HERE / 'review_lock.json')
assert report['status'] == 'complete' and len(report['workers']) == len(report['videos']) == 6
assert report['review_lock_sha256'] == hashlib.sha256((HERE / 'review_lock.json').read_bytes()).hexdigest()
assert datetime.fromisoformat(lock['created_utc']) < datetime.fromisoformat(report['started_utc'])
inputs = {case['ID']: case for case in read(HERE / 'inputs.json')}
rows = []
for record, process in zip(report['videos'], report['workers']):
    sid = record['ID']
    assert sid == process['ID'] and process['exit_status'] == 0
    manifest = inputs[sid]
    seconds = {image['frame']: image['pts_seconds'] for image in manifest['images']}
    baseline, candidate, diagnostic = (record[key] for key in ['baseline_prediction', 'candidate_prediction', 'diagnostics'])
    worker = read(HERE / 'mac_run' / sid / 'worker_report.json')
    assert worker['status'] == 'complete' and worker['model_calls'] == len(record['calls']) == 4
    assert worker['network_attempts'] == 0
    vlm_collision = diagnostic['collision']['collision_frame']
    rows.append(dict(ID=sid, frames=record['frames'], baseline=baseline, candidate=candidate,
                     baseline_collision_seconds=seconds[baseline['collision_frame']],
                     candidate_collision_seconds=seconds[candidate['collision_frame']],
                     entry_seconds=seconds[candidate['entry_frame']],
                     internal_vlm_collision_frame=vlm_collision,
                     internal_vlm_collision_seconds=seconds[vlm_collision],
                     collision_candidates=diagnostic['collision_candidates'], entry_candidates=diagnostic['entry_candidates'],
                     entry_after_baseline_contact=seconds[baseline['entry_frame']] > seconds[baseline['collision_frame']],
                     entry_after_candidate_contact=seconds[candidate['entry_frame']] > seconds[candidate['collision_frame']],
                     final_contact_differs_from_vlm_context=candidate['collision_frame'] != vlm_collision,
                     collision_changed=baseline['collision_frame'] != candidate['collision_frame'],
                     other_three_unchanged=all(baseline[k] == candidate[k] for k in ['entry_frame', 'entry_side', 'evasion_space']),
                     parent_worker_wall_seconds=process['wall_seconds'], worker_wall_seconds=worker['wall_seconds'],
                     motion_seconds=record['motion_seconds'], model_load_seconds=record['model_load_seconds'],
                     model_generation_seconds=sum(call['seconds'] for call in record['calls']),
                     mlx_peak_memory_GB=record['max_peak_memory_GB'], process_peak_rss_bytes=worker['max_process_rss_bytes']))

summary = dict(status='complete', reviews_frozen_before_predictions=True, cases=6, original_frames=sum(r['frames'] for r in rows),
               new_real_model_calls=24, changed_collision_ids=[r['ID'] for r in rows if r['collision_changed']],
               other_three_unchanged=sum(r['other_three_unchanged'] for r in rows),
               parent_worker_wall_seconds_sum=sum(r['parent_worker_wall_seconds'] for r in rows),
               mlx_peak_memory_GB_max=max(r['mlx_peak_memory_GB'] for r in rows),
               process_peak_rss_bytes_max=max(r['process_peak_rss_bytes'] for r in rows),
               python_socket_connection_attempts=0,
               network_measurement_limit='Python socket connect hooks plus HuggingFace offline configuration; not OS-wide native-network tracing',
               shared_call_policy='One shared original four-call inference followed by two final motion-score selections; no two-model latency comparison',
               official_S2=None, accuracy_not_computed_here='Only prediction diagnostics. Eligibility comes from the separately locked expert adjudication.',
               rows=rows)
(HERE / 'prediction_analysis.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2))
print(json.dumps({key: value for key, value in summary.items() if key != 'rows'}, ensure_ascii=False, indent=2))
for r in rows:
    print(r['ID'], r['baseline'], '->', r['candidate'], 'wall', round(r['parent_worker_wall_seconds'], 3))
