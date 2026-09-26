"""Read-only inventory of existing two CCD cohorts and cached baseline runs."""
import json,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];HERE=Path(__file__).resolve().parent
read=lambda p:json.loads(p.read_text())
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
selected={'CCD_'+x for x in ['000688','000801','000196','000453','000728','000052','001237','000540']}
rows=[]
for base,intake,run in [('stage2_goal_20260920','ccd_intake','mac_run'),('stage2_first_frame_policy_20260920','intake','run')]:
    parent=ROOT/'artifacts'/base
    for src in read(parent/intake/'inputs.json'):
        sid=src['ID']; folder=parent/run/(sid if base.startswith('stage2_goal_') else sid+'_baseline')
        if not (folder/'result.json').exists():
            candidates=list((parent/run).glob(f'*{sid}*/result.json'));assert len(candidates)==1,(sid,candidates);folder=candidates[0].parent
        rows.append(dict(ID=sid,cohort=base,source=src,baseline_folder=str(folder.relative_to(ROOT)),review_selected=sid in selected,
            reason='Prior entry reference exists: revalidate before reuse' if sid in selected else 'Existing entry unknown: retained in inventory, not newly adjudicated or scored',
            baseline_sha256={n:sha(folder/n) for n in ['result.json','calls.json','worker_report.json']}))
assert len(rows)==24 and sum(r['review_selected'] for r in rows)==8
(HERE/'inventory.json').write_text(json.dumps(dict(status='fixed_before_new_candidate_calls',selection_scope='Prior-reference audit, not representative or independent held-out sampling',cases=rows),ensure_ascii=False,indent=2)+'\n')
print([(r['ID'],r['baseline_folder']) for r in rows if r['review_selected']])
for r in rows:
    if r['ID']=='CCD_000688':
        result=read(ROOT/r['baseline_folder']/'result.json');print('result keys',list(result));print('diag keys',list(result.get('baseline_diagnostics',{})));print(json.dumps({k:v for k,v in result.items() if k not in ['calls','baseline_diagnostics','candidate_diagnostics']},ensure_ascii=False)[:2000])
