"""Compare common cases without mixing performance metrics with timing noise."""
import argparse,json
from pathlib import Path

def compare(left,right):
    def records(path):
        report=json.loads((path/'stage2_report.json').read_text())
        if report['status']!='complete':raise ValueError('Incomplete run')
        return {(r['group'],r['ID']):r for r in report['videos']}
    a,b=records(left),records(right);shared=sorted(a.keys() & b.keys())
    if not shared:raise ValueError('No common cases')
    details=[]
    for key in shared:
        x,y=a[key],b[key]
        calls=[]
        if len(x['calls'])!=len(y['calls']):raise ValueError('Call count differs')
        for i,(c,d) in enumerate(zip(x['calls'],y['calls'])):
            fields=['text','image_sha256','prompt_sha256','processor_input_sha256']
            if c.get('token_trace') and d.get('token_trace'):fields.append('token_trace')
            calls.append(dict(call=i+1,equal={k:c.get(k)==d.get(k) for k in fields},missing=[k for k in fields if k not in c or k not in d]))
        details.append(dict(group=key[0],ID=key[1],prediction_equal=x['prediction']==y['prediction'],calls=calls))
    return dict(equal=all(r['prediction_equal'] and all(all(c['equal'].values()) and not c['missing'] for c in r['calls']) for r in details),cases=details)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('left',type=Path);p.add_argument('right',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    r=compare(a.left,a.right)
    with a.output.open('x') as f:json.dump(r,f,indent=2)
    print(json.dumps(r,indent=2));raise SystemExit(0 if r['equal'] else 1)
