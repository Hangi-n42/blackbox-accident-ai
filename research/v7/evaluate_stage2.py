"""Separate evaluation against actual human drafts; missing labels stay unknown."""
import argparse,json,hashlib
from pathlib import Path
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
def read(p): return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def macro(pairs,classes):
    scores=[]
    for c in classes:
        tp=sum(a==c and b==c for a,b in pairs);fp=sum(a!=c and b==c for a,b in pairs);fn=sum(a==c and b!=c for a,b in pairs)
        scores.append(2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.)
    return sum(scores)/len(scores) if pairs else None
def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args();out=a.run/'evaluation.json'
    assert not out.exists(),'Preserve earlier evaluation'
    report=read(a.run/'report.json');assert report['status']=='complete'
    stage=ROOT/'research/v6_stage2';reviews={}
    for f in (stage/'user_reviews').glob('NEXAR_REVIEW_*_review_*.json'):
        v=read(f);ID=v['ID'].split('_')[-1]
        if ID not in reviews or f.name>reviews[ID].name:reviews[ID]=f
    binding=read(stage/'round2_validation/run/review_binding.json')
    reviews.update({v['ID']:Path(v['review_path']) for v in binding['reviews']})
    rows=[];metric={k:{'contact':[],'entry':[],'side':[],'space':[]} for k in ('baseline','candidate')}
    for result in report['videos']:
        ID=result['ID'];file=reviews[ID];draft=read(file);human=draft['review'];times={f['frame']:f['pts_seconds'] for f in result['frame_pts']}
        assert draft['source_video_sha256']==result['source_sha256'] and draft['record_type']=='human_review_draft'
        row=dict(ID=ID,review_path=str(file),review_sha256=sha(file),errors={},unknown=[])
        for field,output in (('contact','collision_frame'),('entry','entry_frame')):
            label=human.get(field,{})
            if label.get('status')!='observed':row['unknown'].append(field);continue
            assert type(label['frame']) is int and times[label['frame']]==label['pts_seconds']
            row['errors'][field]={}
            for variant in metric:
                error=times[result[variant][output]]-label['pts_seconds'];row['errors'][field][variant]=error
                metric[variant][field].append(abs(error))
        for field,output,classes in (('side','entry_side',['LEFT','RIGHT']),('space','evasion_space',[0,1])):
            label=human.get(field)
            if field=='space' and label in ('0','1'):label=int(label)
            if label not in classes:row['unknown'].append(field);continue
            for variant in metric:metric[variant][field].append((label,result[variant][output]))
        rows.append(row)
    scores={}
    for variant,m in metric.items():
        s={}
        for field in ('contact','entry'):
            errors=m[field];s[field]=dict(n=len(errors),correct=sum(e<=.3+1e-12 for e in errors),
                accuracy=sum(e<=.3+1e-12 for e in errors)/len(errors) if errors else None,mae=sum(errors)/len(errors) if errors else None)
        for field,classes in (('side',['LEFT','RIGHT']),('space',[0,1])):
            s[field]=dict(n=len(m[field]),macro_f1=macro(m[field],classes),correct=sum(a==b for a,b in m[field]))
        vals=[s['contact']['accuracy'],s['entry']['accuracy'],s['side']['macro_f1'],s['space']['macro_f1']]
        s['available_label_weighted_diagnostic']=sum(w*v for w,v in zip((.35,.35,.15,.15),vals)) if all(v is not None for v in vals) else None
        scores[variant]=s
    transitions={}
    for field in ('contact','entry'):
        pairs=[r['errors'][field] for r in rows if field in r['errors']]
        transitions[field]=dict(gained=sum(abs(p['baseline'])>.3+1e-12 and abs(p['candidate'])<=.3+1e-12 for p in pairs),
            lost=sum(abs(p['baseline'])<=.3+1e-12 and abs(p['candidate'])>.3+1e-12 for p in pairs))
    value=dict(source_role='exposed_development',report_sha256=sha(a.run/'report.json'),scores=scores,transitions=transitions,videos=rows,
        official_S2=None,ground_truth_promotion=False,submission_approved=False,
        limitation='Per-field known human drafts have unequal denominators; weighted diagnostic is not official S2 or independent validation.')
    with out.open('x',encoding='utf8') as f:json.dump(value,f,ensure_ascii=False,indent=2)
    print(json.dumps(dict(scores=scores,transitions=transitions),indent=2))
if __name__=='__main__':main()
