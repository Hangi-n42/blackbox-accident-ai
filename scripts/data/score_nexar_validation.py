"""Score the saved baseline against pre-inference AI reference intervals."""
import hashlib,json
from pathlib import Path
BASE=Path(__file__).resolve().parents[2]/'artifacts/data_pilot_20260916/nexar_validation'
ref=json.loads((BASE/'reference_frozen.json').read_text())
for name,digest in ref['sources'].items():
    assert hashlib.sha256((BASE/name).read_bytes()).hexdigest()==digest, 'Reference changed'
run=json.loads((BASE/'baseline_metal/stage2_report.json').read_text())
assert run['status']=='complete'
rows=[]
for video in run['videos']:
    ident=video['ID']; expected=ref['cases'][ident]; pred=video['prediction'];pts=json.loads((BASE/f'{ident}.pts.json').read_text())
    row={'ID':ident,'prediction':pred,'reference':expected,'time_errors':{}}
    for name,column in [('contact','collision_frame'),('entry','entry_frame')]:
        if expected[name] is None:continue
        t=pts[pred[column]];lo,hi=expected[name];minimum=max(lo-t,0,t-hi);maximum=max(abs(t-lo),abs(t-hi))
        row['time_errors'][name]={'seconds':t,'min_error':minimum,'max_error':maximum,'result':'inside_all' if maximum<=.3+1e-9 else 'outside_all' if minimum>.3+1e-9 else 'interval_dependent'}
    for name,column in [('side','entry_side'),('space','evasion_space')]:
        row[name+'_match']=None if expected[name] is None else expected[name]==pred[column]
    rows.append(row)
report={'reference_type':ref['reference_type'],'common_four_field_cases':0,'cases':rows}
(BASE/'baseline_comparison.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
for row in rows:print(json.dumps(row,ensure_ascii=False))
