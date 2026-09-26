"""One Mac train-only fit: fixed 53 pairs, lambda 1 -> 0.1 only."""
import importlib.util
import json
import platform
import sys
from datetime import datetime, timezone
from fractions import Fraction as F
from pathlib import Path
import numpy as np
import scipy
from scipy.optimize import minimize

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
BASE=ROOT/'artifacts/stage2_entry_top_pairs_20260921'
spec=importlib.util.spec_from_file_location('fixed_top_pairs',BASE/'run.py')
helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
L2=0.1

def write(name,obj):
    (HERE/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n')

def objective(w,diffs,l2):
    loss=l2*np.dot(w,w)/2;grad=l2*w.copy()
    for d in diffs:
        margin=d@w
        loss+=np.logaddexp(0,-margin).mean()/len(diffs)
        grad-=(d.T@np.exp(-np.logaddexp(0,margin)))/len(d)/len(diffs)
    return float(loss),grad

def loss_record(w,diffs,l2):
    loss,grad=objective(w,diffs,l2)
    return {'lambda':l2,'total':loss,'l2_penalty':float(l2*(w@w)/2),
            'data_loss':float(loss-l2*(w@w)/2),'weight_l2':float(np.linalg.norm(w)),
            'gradient_l2':float(np.linalg.norm(grad)),'gradient_inf':float(np.abs(grad).max()),
            'numerical_strong_convexity_gap_bound':float(grad@grad/(2*l2)),
            'pair_accuracy_per_incident':[float(np.mean(d@w>0)) for d in diffs]}

def main():
    assert platform.system()=='Darwin' and platform.machine()=='arm64'
    assert not (HERE/'protocol.json').exists(),'One fit only; do not overwrite'
    prior=json.loads((BASE/'protocol.json').read_text())
    baseline=json.loads((BASE/'result.json').read_text())
    source_hashes=json.loads((BASE/'freeze.json').read_text())['sha256']
    for path,h in source_hashes.items():assert helper.sha(ROOT/path)==h,path
    prior_status=json.loads((BASE/'STATUS.json').read_text())['artifact_sha256']
    for name in ['model.npz','result.json','protocol.json','run.py']:
        assert helper.sha(BASE/name)==prior_status[str((BASE/name).relative_to(ROOT))]
    state=dict(np.load(BASE/'model.npz',allow_pickle=False))
    x=np.load(helper.PRIOR/'design.npy',allow_pickle=False)
    jobs={j['ID']:j for j in json.loads((helper.OLD/'training_inputs.json').read_text())['jobs']}
    refs={r['ID']:r for r in json.loads((helper.OLD/'training_references.json').read_text())['cases'] if r['eligible']}
    selector=helper.load_module('frozen_selector',helper.OLD/'selector.py')
    embeddings=np.stack([np.load(helper.OLD/'features'/f'{i}.npy',allow_pickle=False) for i in helper.IDS])
    reconstructed=selector.design(embeddings,np.array([jobs[i]['times'] for i in helper.IDS]),state)
    np.testing.assert_array_equal(x,reconstructed)
    records=prior['cases'];pairs=[np.asarray(c['top_pairs'],dtype=int) for c in records]
    assert [c['ID'] for c in records]==helper.IDS
    assert [len(p) for p in pairs]==[11,11,11,20]
    assert not any(a in (10,11) and b in (10,11) for a,b in pairs[3])
    diffs=[z[p[:,0]]-z[p[:,1]] for z,p in zip(x,pairs)]
    # Lambda=1 formula must match the previous implementation exactly.
    for w in [np.zeros(14),state['weights']]:
        a,g=objective(w,diffs,1.0);b,h=helper.objective(w,diffs)
        assert a==b;np.testing.assert_array_equal(g,h)
    protocol={**prior,'created_utc':datetime.now(timezone.utc).isoformat(),
              'sole_change':'Keep all53pairs and other conditions; lambda1 -> lambda0.1.',
              'user_wording_interpretation':'3 pairs interpreted as 53 pairs from immediately preceding proposal; stated to user before work.',
              'L2':L2,'loss':'Mean pair logistic per incident, equal mean over4 incidents + 0.1||w||²/2',
              'baseline':str(BASE.relative_to(ROOT)),'baseline_L2':1.0,
              'initialization':'zeros(14); no warm start','fits_authorized':1,
              'evaluation':'Train4 only vs53pair lambda1 baseline. Score gap, membership, error range, same-T paired MAE and ±0.3 status. No CCD.'}
    write('protocol.json',protocol)
    paths=[ROOT/p for p in source_hashes]+[BASE/'model.npz',BASE/'result.json',BASE/'freeze.json',BASE/'STATUS.json',Path(__file__),HERE/'protocol.json']
    frozen={str(p.relative_to(ROOT)):helper.sha(p) for p in paths}
    write('freeze.json',{'created_utc':datetime.now(timezone.utc).isoformat(),'sha256':frozen})
    initial=np.zeros(14)
    opt=minimize(lambda w:objective(w,diffs,L2),initial,jac=True,method='L-BFGS-B',options=helper.OPTIONS)
    fit={'success':bool(opt.success),'message':str(opt.message),'iterations':int(opt.nit),'function_evaluations':int(opt.nfev),
         'initial':loss_record(initial,diffs,L2),'final':loss_record(opt.x,diffs,L2)}
    write('fit.json',fit)
    assert opt.success and np.isfinite(opt.fun) and opt.fun<objective(initial,diffs,L2)[0]
    np.savez(HERE/'model.npz',**{**state,'weights':opt.x})
    cases=[];mean_delta=[F(0),F(0)]
    for i,(z,c,p) in enumerate(zip(x,records,pairs)):
        sid=c['ID'];old=helper.evaluate(z,state['weights'],jobs[sid],refs[sid],c['possible_nearest_indices'],p)
        assert old==baseline['cases'][i]['new']
        new=helper.evaluate(z,opt.x,jobs[sid],refs[sid],c['possible_nearest_indices'],p)
        lo,hi=map(lambda t:F(str(t)),c['reference_seconds'])
        a,b=map(lambda t:F(str(t)),[old['selected_seconds'],new['selected_seconds']])
        points={lo,hi,*[t for t in [a,b] if lo<=t<=hi]}
        changes=[abs(b-t)-abs(a-t) for t in points]
        delta=[min(changes),max(changes)]
        mean_delta=[s+d/4 for s,d in zip(mean_delta,delta)]
        # Fixed-weight score contributions, not feature ablation or another fit.
        terms={}
        for label,w,ev in [('old',state['weights'],old),('new',opt.x,new)]:
            scores=z@w;allowed=c['possible_nearest_indices'];outside=[j for j in range(12) if j not in allowed]
            inside=max(allowed,key=lambda j:(scores[j],-j));out=max(outside,key=lambda j:(scores[j],-j))
            contributions=(z[inside]-z[out])*w
            terms[label]={'appearance_pca':float(contributions[:4].sum()),'previous_difference':float(contributions[4:8].sum()),
                          'following_difference':float(contributions[8:12].sum()),'relative_time':float(contributions[12]),
                          'first_indicator':float(contributions[13]),'sum':float(contributions.sum())}
        cases.append({**c,'old':old,'new':new,'paired_error_change_range':[float(t) for t in delta],
                      'paired_error_change_exact':[str(t) for t in delta],'inside_minus_outside_score_decomposition':terms})
    summary={}
    for version in ['old','new']:
        rows=[c[version] for c in cases]
        summary[version]={'possible_nearest_top1':sum(r['in_possible_nearest_set'] for r in rows),
                          'within_0_3_counts':{s:sum(r['within_0_3_status']==s for r in rows) for s in ['correct_all_reference_times','wrong_all_reference_times','indeterminate']},
                          'mean_error_lower':float(np.mean([r['absolute_error_seconds'][0] for r in rows])),
                          'mean_error_upper':float(np.mean([r['absolute_error_seconds'][1] for r in rows]))}
    assert all(helper.sha(ROOT/p)==h for p,h in frozen.items())
    result={'completed_utc':datetime.now(timezone.utc).isoformat(),'platform':platform.platform(),'python':sys.version,
            'numpy':np.__version__,'scipy':scipy.__version__,'cases':cases,'summary':summary,'fit':fit,
            'paired_mean_error_change_range':[float(t) for t in mean_delta],
            'paired_mean_error_change_exact':[str(t) for t in mean_delta],
            'objectives':{f'{label}_weights_lambda_{lam}':loss_record(w,diffs,lam) for label,w in [('old',state['weights']),('new',opt.x)] for lam in [1.0,L2]},
            'training_runs':1,'additional_lambda_trials':0,'ccd_evaluations':0,'encoder_forwards':0,
            'submission_changed':False,'candidate_adopted':False,'frozen_files_preserved':len(frozen)}
    write('result.json',result)
    print(json.dumps({'summary':summary,'fit':fit,'paired_mean_error_change_range':result['paired_mean_error_change_range']},indent=2))
    for c in cases:print(c['ID'],{v:{k:c[v][k] for k in ['selected_frame','inside_minus_outside','absolute_error_seconds']} for v in ['old','new']})

if __name__=='__main__':main()
