"""Fourth fixed condition diagnostics, no inference or tuning."""
import json
from pathlib import Path
HERE=Path(__file__).resolve().parent
def read(p):return json.loads(p.read_text('utf8'))
run=HERE/'grounded_8b_run';r=read(run/'report.json');evaluation=read(run/'evaluation.json')
human={x['ID']:read(Path(x['review_path']))['review'] for x in evaluation['videos']}
rows=[]
for v in r['videos']:
    diag=v['diagnostics'];times={x['frame']:x['pts_seconds'] for x in v['frame_pts']};row=dict(ID=v['ID'],events={})
    for field,key,candidates in [('contact','collision_frame','collision_candidates'),('entry','entry_frame','entry_candidates')]:
        h=human[v['ID']][field]
        if h['status']!='observed':continue
        proposed=diag[candidates];target=h['pts_seconds'];best=min(abs(times[n]-target) for n in proposed);actual=abs(times[v['candidate'][key]]-target)
        row['events'][field]=dict(candidate_nearest_seconds=best,selected_error_seconds=actual,candidate_has_tolerated_frame=best<=.3+1e-12,selected_within_tolerance=actual<=.3+1e-12,window_status=diag[field+'_window_status']['status'],selection_route=diag[field+'_selection_route'] if field=='entry' else diag['contact_selection_route'])
    rows.append(row)
summary={}
for field in ('contact','entry'):
    events=[v['events'][field] for v in rows if field in v['events']]
    summary[field]=dict(n=len(events),candidate_omission=sum(not x['candidate_has_tolerated_frame'] for x in events),selection_failure_when_candidate_present=sum(x['candidate_has_tolerated_frame'] and not x['selected_within_tolerance'] for x in events),correct=sum(x['selected_within_tolerance'] for x in events))
value=dict(status='completed_fixed_condition_postmortem',summary=summary,rows=rows,overview_all_null_contact=sum(all(x is None for x in v['diagnostics']['trace'][0]['parsed'].get('contact',[])) for v in r['videos']),side_predictions=[v['candidate']['entry_side'] for v in r['videos']],space_predictions=[v['candidate']['evasion_space'] for v in r['videos']],limitation='Development postmortem; no parameter search or independent accuracy claim')
with (HERE/'grounded_8b_decomposition.json').open('x',encoding='utf8') as f:json.dump(value,f,ensure_ascii=False,indent=2)
print(json.dumps(summary,indent=2))
