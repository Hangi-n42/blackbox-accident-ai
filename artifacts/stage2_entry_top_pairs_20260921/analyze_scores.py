"""Arithmetic explanation of the completed fit; no additional optimization."""
import json
from fractions import Fraction as F
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
def main():
    assert not (HERE/'score_analysis.json').exists()
    result=json.loads((HERE/'result.json').read_text())
    x=np.load(ROOT/'artifacts/stage2_entry_feasibility_20260920/design.npy')
    old=dict(np.load(ROOT/'artifacts/stage2_entry_selector_20260920/model.npz'))
    new=dict(np.load(HERE/'model.npz'))
    refs={r['ID']:r for r in json.loads((ROOT/'artifacts/stage2_entry_selector_20260920/training_references.json').read_text())['cases']}
    groups={'appearance_pca':slice(0,4),'previous_difference':slice(4,8),'following_difference':slice(8,12),
            'relative_time':slice(12,13),'first_candidate_indicator':slice(13,14)}
    rows=[];mean_lo=F(0);mean_hi=F(0)
    for z,c in zip(x,result['cases']):
        s=z@new['weights']; k=int(np.argmax(s));j=max(c['possible_nearest_indices'],key=lambda i:(s[i],-i))
        terms=(z[k]-z[j])*new['weights']
        oldt,newt=map(lambda a:F(str(a)),[c['old']['selected_seconds'],c['new']['selected_seconds']])
        lo,hi=map(lambda a:F(str(a)),c['reference_seconds'])
        points={lo,hi,*[t for t in (oldt,newt) if lo<=t<=hi]}
        delta=[abs(newt-t)-abs(oldt-t) for t in points]
        bounds=[min(delta),max(delta)];mean_lo+=bounds[0]/4;mean_hi+=bounds[1]/4
        at_zero={}
        for label,pairs in [('old258',refs[c['ID']]['preference_pairs']),('new53',c['top_pairs'])]:
            p=np.array(pairs);d=z[p[:,0]]-z[p[:,1]]
            at_zero[label]={'first_indicator_gradient_contribution_to_total':float(-d[:,13].mean()/8),
                            'positive_first_indicator_pairs':int(np.sum(d[:,13]>0)),
                            'negative_first_indicator_pairs':int(np.sum(d[:,13]<0)), 'pair_count':len(p)}
        rows.append({'ID':c['ID'],'winner_frame':c['frames'][k],'best_inside_frame':c['frames'][j],
                     'winner_minus_best_inside_by_feature_group':{name:float(terms[index].sum()) for name,index in groups.items()},
                     'winner_minus_best_inside_total':float(s[k]-s[j]),
                     'paired_absolute_error_change_range':[float(v) for v in bounds],
                     'paired_absolute_error_change_exact':[str(v) for v in bounds],
                     'zero_initialization_first_indicator_gradient':at_zero})
    out={'cases':rows,'paired_mean_absolute_error_change_range':[float(mean_lo),float(mean_hi)],
         'paired_mean_absolute_error_change_exact':[str(mean_lo),str(mean_hi)],
         'first_indicator_weight':{'old':float(old['weights'][13]),'new':float(new['weights'][13])},
         'relative_time_weight':{'old':float(old['weights'][12]),'new':float(new['weights'][12])},
         'limits':'Score decomposition and objective-gradient arithmetic, not feature ablation, causal intervention, or generalization evidence.',
         'additional_fits':0,'ccd_accesses':0}
    (HERE/'score_analysis.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(out,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
