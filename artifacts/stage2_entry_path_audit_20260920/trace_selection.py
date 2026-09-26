"""Conservative interval coverage along unchanged entry-selection stages."""
from fractions import Fraction as F
import json
from pathlib import Path
HERE=Path(__file__).resolve().parent
read=lambda p:json.loads(p.read_text())

def coverage(frames,times,lo,hi):
    lo,hi=F(str(lo)),F(str(hi));tol=F(3,10)
    bounds=[(F(str(times[str(f)]))-tol,F(str(times[str(f)]))+tol) for f in frames]
    segments=sorted((max(lo,a),min(hi,b)) for a,b in bounds if b>=lo and a<=hi)
    reach=lo;full=bool(segments)
    for a,b in segments:
        if a>reach:full=False;break
        reach=max(reach,b)
    full=full and reach>=hi
    robust=[f for f in frames if max(abs(F(str(times[str(f)]))-lo),abs(F(str(times[str(f)]))-hi))<=tol]
    possible=[f for f in frames if max(lo-F(str(times[str(f)])),F(0),F(str(times[str(f)]))-hi)<=tol]
    return dict(all_truths_have_some_candidate=full,any_truth_has_candidate=bool(segments),single_candidate_covers_entire_reference=robust,possibly_correct_candidates=possible)

def grade(p,lo,hi):
    p,lo,hi=map(lambda v:F(str(v)),[p,lo,hi]);tol=F(3,10)
    best=max(lo-p,F(0),p-hi);worst=max(abs(p-lo),abs(p-hi))
    return dict(result='correct' if worst<=tol else 'wrong' if best>tol else 'indeterminate',minimum_error_seconds=float(best),maximum_error_seconds=float(worst))

assert coverage([0,5],{'0':0,'5':.5},0,.5)['all_truths_have_some_candidate']
assert not coverage([0,7],{'0':0,'7':.7},0,.7)['all_truths_have_some_candidate']
assert grade(.3,0,0)['result']=='correct'

def main():
    sources={r['ID']:r for r in read(HERE/'path_replay.json')['rows']};refs=read(HERE/'references.json')['cases'];rows=[]
    for ref in refs:
        src=sources[ref['ID']];row=dict(ID=ref['ID'],reference=ref['entry'],eligible=ref['eligible'],same_counterpart_verified=ref['same_counterpart_verified'],
            vlm_contact_frame=src['vlm_contact_frame'],final_contact_frame=src['final_contact_frame'],final_entry=src['final_entry'],parser_fallback=src['choice_fallback_or_mapping'])
        if ref['eligible']:
            entry=ref['entry'];times=src['original_times'];lo,hi=[times[str(entry[k])] for k in ['lower_frame','upper_frame']]
            row['reference_seconds']=[lo,hi];row['stages']={name:coverage(src[key],times,lo,hi) for name,key in [('original','all_frames'),('precontact','precontact_frames'),('candidates','entry_candidates')]}
            row['final']=grade(times[str(src['final_entry'])],lo,hi)
            row['first_coverage_loss']=next((name for name in ['original','precontact','candidates'] if not row['stages'][name]['all_truths_have_some_candidate']),None)
            row['classification']='candidate_path_partial_or_missing' if row['first_coverage_loss'] else 'selection_definite_error' if row['final']['result']=='wrong' else 'selection_unresolved' if row['final']['result']=='indeterminate' else 'correct'
        else:row['classification']='unscored_reference_unknown'
        rows.append(row)
    eligible=[r for r in rows if r['eligible']]
    summary=dict(reviewed=len(rows),eligible=len(eligible),unknown=len(rows)-len(eligible),stage_full_coverage={s:sum(r['stages'][s]['all_truths_have_some_candidate'] for r in eligible) for s in ['original','precontact','candidates']},
        definite_correct_candidate_cases=sum(bool(r['stages']['candidates']['single_candidate_covers_entire_reference']) for r in eligible),final={g:sum(r['final']['result']==g for r in eligible) for g in ['correct','wrong','indeterminate']},
        before_start=sum(r['reference']['status']=='before_start' for r in eligible),during_clip=sum(r['reference']['status']=='during_clip' for r in eligible))
    (HERE/'decomposition.json').write_text(json.dumps(dict(status='complete',summary=summary,rows=rows,scope='Existing exposed CCD references re-reviewed; no model calls, no official score. Full interval coverage differs from one candidate valid for every time in the interval.'),ensure_ascii=False,indent=2)+'\n');print(json.dumps(summary))

if __name__=='__main__':main()
