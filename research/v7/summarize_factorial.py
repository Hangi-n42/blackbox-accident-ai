"""Four fixed development conditions; descriptive comparisons, not causal proof."""
import json, hashlib
from pathlib import Path
H=Path(__file__).resolve().parent
def read(p):return json.loads(p.read_text('utf8'))
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
files={'4b_unified':H/'stage2_dev_run1/evaluation.json',
       '8b_v6':H/'capacity_8b_combined/evaluation.json',
       '8b_unified':H/'grounded_8b_run/evaluation.json'}
e={k:read(p) for k,p in files.items()}
baseline=e['4b_unified']['scores']['baseline']
assert all(v['scores']['baseline']==baseline for v in e.values())
cells={'4b_v6':baseline,**{k:v['scores']['candidate'] for k,v in e.items()}}
fields=[('contact','accuracy'),('entry','accuracy'),('side','macro_f1'),('space','macro_f1')]
effects={}
for f,m in fields:
    assert len({c[f]['n'] for c in cells.values()})==1
    a,b,c,d=[cells[k][f][m] for k in ('4b_v6','4b_unified','8b_v6','8b_unified')]
    effects[f]={'metric':m,'8b_minus_4b_under_v6':c-a,'8b_minus_4b_under_unified':d-b,
                'unified_minus_v6_under_4b':b-a,'unified_minus_v6_under_8b':d-c,
                'descriptive_difference_of_differences':(d-b)-(c-a)}
final=cells['8b_unified']
checks={f+'_nondecrease':final[f][m]>=baseline[f][m] for f,m in fields}
checks['at_least_one_improvement']=any(final[f][m]>baseline[f][m] for f,m in fields)
result={'cells':cells,'effects':effects,'8b_unified_initial_numeric_gate':all(checks.values()),'gate_checks':checks,
        'evaluation_bindings':{str(p):sha(p) for p in files.values()},'summary_script_sha256':sha(Path(__file__)),
        'independent_validation':False,'adopted':False,'submitted':False,
        'limits':['Same nine already exposed development sources and per-field unequal known-label denominators.',
                  'Different checkpoint vision architecture; not isolated language parameter count.',
                  'Same source and rendering rules; adaptive follow-up pixels can differ.',
                  'Descriptive metric interaction only, no causal or statistical-significance claim.',
                  'Weighted available-label diagnostic is not official Stage2 score.']}
with (H/'factorial_summary.json').open('x',encoding='utf8') as f:json.dump(result,f,ensure_ascii=False,indent=2,allow_nan=False)
print(json.dumps({'8b_unified_initial_numeric_gate':result['8b_unified_initial_numeric_gate'],
                  'checks':checks,'cells':{k:{f:c[f][m] for f,m in fields} for k,c in cells.items()}},indent=2))
