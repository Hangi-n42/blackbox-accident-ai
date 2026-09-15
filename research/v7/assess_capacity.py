"""Report the predeclared development gate, class confusions, and timing limits."""
import argparse, hashlib, json
from pathlib import Path
HERE=Path(__file__).resolve().parent; ROOT=HERE.parents[1]
def read(p): return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args()
    out=a.run/'assessment.json'
    if out.exists():raise ValueError('Preserve existing assessment')
    report=read(a.run/'report.json');e=read(a.run/'evaluation.json')
    assert report['status']=='complete' and e['report_sha256']==sha(a.run/'report.json')
    assert len(report['videos'])==9
    b=e['scores']['baseline'];c=e['scores']['candidate']
    checks={'contact_unchanged':b['contact']['accuracy']==c['contact']['accuracy'],
            'contact_predictions_unchanged':all(r['baseline']['collision_frame']==r['candidate']['collision_frame'] for r in report['videos']),
            'entry_nondecrease':c['entry']['accuracy']>=b['entry']['accuracy'],
            'side_nondecrease':c['side']['macro_f1']>=b['side']['macro_f1'],
            'space_nondecrease':c['space']['macro_f1']>=b['space']['macro_f1'],
            'changed_field_improves':any(c[f][m]>b[f][m] for f,m in [('entry','accuracy'),('side','macro_f1'),('space','macro_f1')])}
    evaluated={r['ID']:r for r in e['videos']}
    confusions={v:{f:{'classes':cl,'matrix':[[0,0],[0,0]],'correct_ids':[],
                      'prediction_counts':{str(x):0 for x in cl}}
                     for f,cl in [('side',['LEFT','RIGHT']),('space',[0,1])]}
                for v in ('baseline','candidate')}
    transitions={f:{'gained':[],'lost':[]} for f in ('contact','entry','side','space')}
    round2={r['ID']:r for r in read(ROOT/'research/v6_stage2/round2_validation/run/report.json')['videos']}
    timing=[]
    for row in report['videos']:
        ID=row['ID'];er=evaluated[ID];assert sha(er['review_path'])==er['review_sha256']
        human=read(er['review_path'])['review']
        for field in ('contact','entry'):
            if field not in er['errors']:continue
            old=abs(er['errors'][field]['baseline'])<=.3+1e-12
            new=abs(er['errors'][field]['candidate'])<=.3+1e-12
            if old and not new:transitions[field]['lost'].append(ID)
            if new and not old:transitions[field]['gained'].append(ID)
        for field,key,classes in [('side','entry_side',['LEFT','RIGHT']),('space','evasion_space',[0,1])]:
            label=human.get(field)
            if field=='space' and label in ('0','1'):label=int(label)
            for variant in confusions:
                cell=confusions[variant][field];pred=row[variant][key]
                cell['prediction_counts'][str(pred)]+=1
                if label in classes:
                    cell['matrix'][classes.index(label)][classes.index(pred)]+=1
                    if pred==label:cell['correct_ids'].append(ID)
            if label in classes:
                old=row['baseline'][key]==label;new=row['candidate'][key]==label
                if old and not new:transitions[field]['lost'].append(ID)
                if new and not old:transitions[field]['gained'].append(ID)
        previous=round2[ID] if ID in round2 else read(ROOT/f'research/v6_stage2/v5_fullframe_diagnostic/traces/{ID}/trace.json')
        assert len(previous['calls'])==len(row['calls'])==4
        assert [r['max_new_tokens'] for r in previous['calls']]==[r['max_new_tokens'] for r in row['calls']]==[64,48,40,40]
        old=sum(r['seconds'] for r in previous['calls']);new=sum(r['seconds'] for r in row['calls'])
        timing.append({'ID':ID,'archived_4b_ask_seconds':old,'current_8b_ask_seconds':new,'ratio':new/old})
    old=sum(r['archived_4b_ask_seconds'] for r in timing);new=sum(r['current_8b_ask_seconds'] for r in timing)
    result={'numeric_development_gate_passed':all(checks.values()),'checks':checks,'scores':e['scores'],
            'confusions_truth_rows_prediction_columns':confusions,'transitions':transitions,
            'timing':{'videos':timing,'sum_4b_ask_seconds':old,'sum_8b_ask_seconds':new,'sum_ratio':new/old,
                      'limitation':'Archived versus current model ask time only; no scan/load/package or hidden all-stage timing guarantee. Not contemporaneous randomized benchmarking.'},
            'report_sha256':sha(a.run/'report.json'),'evaluation_sha256':sha(a.run/'evaluation.json'),
            'assessor_sha256':sha(__file__),'source_role':'already_exposed_development',
            'independent_accuracy_verified':False,'adopted':False,'submission_approved':False,
            'limitation':'Numerical gate is necessary only. Inspect single-class collapse, true target/time validity, independent evidence, and actual package execution before adoption.'}
    with out.open('x',encoding='utf8') as f:json.dump(result,f,ensure_ascii=False,indent=2,allow_nan=False)
    print(json.dumps({k:result[k] for k in ('numeric_development_gate_passed','checks','scores','transitions')},ensure_ascii=False,indent=2))

if __name__=='__main__':main()
