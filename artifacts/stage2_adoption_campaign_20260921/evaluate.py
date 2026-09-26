"""Exact rational interval scoring. AI references are not official ground truth."""
import json
from fractions import Fraction as F
from pathlib import Path
HERE=Path(__file__).resolve().parent
read=lambda p:json.loads(p.read_text())
TAU=F(3,10)
def emit(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2))
def points(lo,hi,preds):
    ps=sorted({lo,hi}|{max(lo,min(hi,p+d)) for p in preds for d in (-TAU,F(0),TAU)})
    return ps+[(a+b)/2 for a,b in zip(ps,ps[1:])]
def measure(pred,lo,hi):
    es=[abs(pred-t) for t in points(lo,hi,[pred])]
    return dict(accuracy_lower=int(max(es)<=TAU),accuracy_upper=int(min(es)<=TAU),error_seconds_lower=float(min(es)),error_seconds_upper=float(max(es)))
def coverage(preds,lo,hi):
    ps=points(lo,hi,preds);ok=[any(abs(p-t)<=TAU for p in preds) for t in ps]
    return dict(any_possible_candidate=any(ok),all_reference_times_covered=all(ok),one_candidate_correct_for_all_times=any(max(abs(p-lo),abs(p-hi))<=TAU for p in preds))
def main():
    rows=[]
    for ref in read(HERE/'references.json')['cases']:
        sid=ref['ID'];pts=read(HERE/'intake'/sid/'pts.json')['mapping'];times={r['frame_id']:F(r['native_pts'])*F(r['time_base']) for r in pts}
        path=HERE/'baseline'/sid/'result.json';r=dict(ID=sid,primary=ref['primary'],secondary=ref['secondary'],entry=ref['entry'],diagnostic_only=not ref['eligible'],failure=not path.exists())
        if not path.exists(): rows.append(r);continue
        base=read(path);bp=base['baseline_prediction'];fs=base['diagnostics']['entry_candidates'];r.update(baseline=bp,entry_candidates=fs,q2_anchor=fs[-1])
        if ref['entry']['status']!='unknown':
            lo=times[ref['entry']['lower_frame']];hi=times[ref['entry']['upper_frame']]
            r.update(reference_seconds=[float(lo),float(hi)],coverage={name:coverage([times[f] for f in seq],lo,hi) for name,seq in [('full_input',list(times)),('q2_window',[f for f in times if f<=fs[-1]]),('candidates',fs)]},baseline_metric=measure(times[bp['entry_frame']],lo,hi))
        cp=HERE/'run'/f'{sid}_candidate/result.json';ctrl=HERE/'run'/f'{sid}_control/result.json'
        if cp.exists() and ctrl.exists():
            cand=read(cp);control=read(ctrl);new=cand['prediction'];r.update(candidate=new,baseline_valid_raw=control['valid_raw_choice'],candidate_valid_raw=cand['valid_raw_choice'],unscored_new_first_frame=(not ref['eligible'] and new['entry_frame']==0 and bp['entry_frame']!=0),unknown_new_first_frame=(ref['entry']['status']=='unknown' and new['entry_frame']==0 and bp['entry_frame']!=0),other_three_equal=all(bp[k]==new[k] for k in ('collision_frame','entry_side','evasion_space')))
            r['control_reproduced']=control['prediction']==bp and all(control['calls'][0][k]==base['calls'][2][k] for k in ('text','token_trace','processor_input_sha256','image_sizes','image_sha256','prompt_sha256'))
            r['cost']={a:dict(q3_seconds=x['calls'][0]['seconds'],wall_seconds=x['wall_seconds'],prompt_tokens=x['calls'][0]['prompt_tokens'],generation_tokens=x['calls'][0]['generation_tokens'],peak_memory_GB=x['calls'][0]['peak_memory'],process_peak_rss_bytes=x['process_peak_rss_bytes'],image_sizes=x['calls'][0]['image_sizes']) for a,x in [('control',control),('candidate',cand)]}
            if ref['entry']['status']!='unknown':
                b=times[bp['entry_frame']];c=times[new['entry_frame']];ps=points(lo,hi,[b,c]);acc=[int(abs(c-t)<=TAU)-int(abs(b-t)<=TAU) for t in ps];ds=[abs(c-t)-abs(b-t) for t in ps]
                r.update(candidate_metric=measure(c,lo,hi),paired_accuracy_delta=[min(acc),max(acc)],paired_mae_delta_seconds=[float(min(ds)),float(max(ds))],definite_recovery=min(acc)==1,possible_regression=min(acc)<0,new_false_first_frame=(new['entry_frame']==0 and bp['entry_frame']!=0 and lo>0))
        rows.append(r)
    groups={}
    for name,key in [('strict','primary'),('conditional','secondary')]:
        rr=[r for r in rows if r[key]];n=len(rr);g=dict(n=n,IDs=[r['ID'] for r in rr],before_start=sum(r['entry']['status']=='before_start' for r in rr),during_clip=sum(r['entry']['status']=='during_clip' for r in rr))
        for arm in ('baseline','candidate'):
            if n and all(arm+'_metric' in r for r in rr):g[arm]={k:sum(r[arm+'_metric'][k] for r in rr)/n for k in ('accuracy_lower','accuracy_upper','error_seconds_lower','error_seconds_upper')}
        if n and all('paired_accuracy_delta' in r for r in rr):
            g.update(paired_accuracy_delta=[sum(r['paired_accuracy_delta'][i] for r in rr)/n for i in (0,1)],paired_mae_delta_seconds=[sum(r['paired_mae_delta_seconds'][i] for r in rr)/n for i in (0,1)],definite_recoveries=sum(r['definite_recovery'] for r in rr),possible_regressions=sum(r['possible_regression'] for r in rr),new_false_first_frames=sum(r['new_false_first_frame'] for r in rr))
        groups[name]=g
    paired=[r for r in rows if 'candidate' in r]
    totals=dict(screened=len(rows),baseline_completed=sum(not r['failure'] for r in rows),paired_completed=len(paired),primary_scored=sum(r['primary'] for r in rows),conditional_scored=sum(r['secondary'] for r in rows),unscored=sum(not r['primary'] and not r['secondary'] for r in rows),control_reproduced=sum(r['control_reproduced'] for r in paired),other_three_equal=sum(r['other_three_equal'] for r in paired),baseline_invalid_raw=sum(not r['baseline_valid_raw'] for r in paired),candidate_invalid_raw=sum(not r['candidate_valid_raw'] for r in paired))
    emit(HERE/'evaluation.json',dict(status='paired_complete' if len(paired)==len(rows) else 'coverage_only_or_partial',truth='blind dual-AI references, no official/human truth; conditional and strict separate',official_stage2_score=None,official_stage2_delta=None,groups=groups,totals=totals,cases=rows))
    print(json.dumps(dict(groups=groups,totals=totals),ensure_ascii=False,indent=2))
if __name__=='__main__':main()
