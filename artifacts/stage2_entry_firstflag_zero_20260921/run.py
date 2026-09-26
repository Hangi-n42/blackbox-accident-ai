"""One fixed-first-indicator fit, keeping lambda.1,53pairs and all features."""
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
BASE=ROOT/'artifacts/stage2_entry_l2_01_20260921'
spec=importlib.util.spec_from_file_location('baseline',BASE/'run.py')
base=importlib.util.module_from_spec(spec);spec.loader.exec_module(base)
helper=base.helper
BOUNDS=[(None,None)]*13+[(0.0,0.0)]

def write(name,obj):
    (HERE/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n')

def record(w,diffs,constrained):
    loss,g=base.objective(w,diffs,.1)
    free=g[:13] if constrained else g
    return {'total':loss,'data_loss':float(loss-.05*(w@w)),'l2_penalty':float(.05*(w@w)),
            'weight_l2':float(np.linalg.norm(w)),'first_indicator_weight':float(w[13]),
            'full_gradient':g.tolist(),'full_gradient_l2':float(np.linalg.norm(g)),
            'stationarity_gradient_l2':float(np.linalg.norm(free)),
            'stationarity_gradient_inf':float(np.abs(free).max()),
            'stationarity_coordinates':'0..12 free coordinates' if constrained else '0..13 unrestricted coordinates',
            'numerical_gap_bound':float(free@free/.2),
            'pair_accuracy_per_incident':[float(np.mean(d@w>0)) for d in diffs]}

def main():
    assert platform.system()=='Darwin' and platform.machine()=='arm64'
    assert not (HERE/'protocol.json').exists(),'One fit only; preserve results'
    prior=json.loads((BASE/'protocol.json').read_text());baseline=json.loads((BASE/'result.json').read_text())
    inherited=json.loads((BASE/'freeze.json').read_text())['sha256']
    for p,h in inherited.items():assert helper.sha(ROOT/p)==h,p
    hashes=json.loads((BASE/'STATUS.json').read_text())['artifact_sha256']
    for name in ['model.npz','protocol.json','result.json','run.py']:
        assert helper.sha(BASE/name)==hashes[str((BASE/name).relative_to(ROOT))]
    state=dict(np.load(BASE/'model.npz',allow_pickle=False));x=np.load(helper.PRIOR/'design.npy',allow_pickle=False)
    jobs={j['ID']:j for j in json.loads((helper.OLD/'training_inputs.json').read_text())['jobs']}
    refs={r['ID']:r for r in json.loads((helper.OLD/'training_references.json').read_text())['cases'] if r['eligible']}
    selector=helper.load_module('frozen_selector',helper.OLD/'selector.py')
    embeddings=np.stack([np.load(helper.OLD/'features'/f'{sid}.npy',allow_pickle=False) for sid in helper.IDS])
    np.testing.assert_array_equal(x,selector.design(embeddings,np.asarray([jobs[sid]['times'] for sid in helper.IDS]),state))
    cases=prior['cases'];pairs=[np.asarray(c['top_pairs'],dtype=int) for c in cases]
    assert [c['ID'] for c in cases]==helper.IDS and [len(p) for p in pairs]==[11,11,11,20]
    assert not any(a in (10,11) and b in (10,11) for a,b in pairs[3])
    diffs=[z[p[:,0]]-z[p[:,1]] for z,p in zip(x,pairs)]
    protocol={'created_utc':datetime.now(timezone.utc).isoformat(),'train_ids':helper.IDS,'cases':cases,
              'baseline':str(BASE.relative_to(ROOT)),'L2':.1,'sole_change':'Fix first-candidate indicator coefficient w[13] to0 by equality bound.',
              'fixed':'All14features,PCA,normalization,12candidates,PTS,weak reference intervals,53pairs,incident means and equal weighting.',
              'loss':'Same mean per-incident pair logistic + .1||w||²/2','initialization':'zeros(14), no warm start',
              'optimizer':'L-BFGS-B','jac':True,'options':helper.OPTIONS,'bounds':BOUNDS,'fits_authorized':1,
              'convergence':'Gradient on free coordinates0..12; gap estimate ||g_free||²/.2. Fixed-coordinate gradient need not vanish.',
              'scope':'Only train4, no CCD reads, no encoder, no submission changes, no followup experiment.',
              'evaluation':'Old unrestricted lambda.1 vs constrained refit: top1,inside-outside max gap,full-reference errors,±.3 and same-T paired MAE. Loss alone is not success.'}
    write('protocol.json',protocol)
    paths=[ROOT/p for p in inherited]+[BASE/'model.npz',BASE/'result.json',BASE/'freeze.json',BASE/'STATUS.json',Path(__file__),HERE/'protocol.json']
    frozen={str(p.relative_to(ROOT)):helper.sha(p) for p in paths}
    write('freeze.json',{'created_utc':datetime.now(timezone.utc).isoformat(),'sha256':frozen})
    initial=np.zeros(14)
    opt=minimize(lambda w:base.objective(w,diffs,.1),initial,jac=True,method='L-BFGS-B',bounds=BOUNDS,options=helper.OPTIONS)
    fit={'success':bool(opt.success),'message':str(opt.message),'iterations':int(opt.nit),'function_evaluations':int(opt.nfev),
         'initial':record(initial,diffs,True),'final':record(opt.x,diffs,True)}
    write('fit.json',fit)
    assert opt.success and np.isfinite(opt.fun) and opt.x[13]==0 and opt.fun<base.objective(initial,diffs,.1)[0]
    np.savez(HERE/'model.npz',**{**state,'weights':opt.x})
    rows=[];mean_delta=[F(0),F(0)]
    for i,(z,c,p) in enumerate(zip(x,cases,pairs)):
        sid=c['ID'];old=helper.evaluate(z,state['weights'],jobs[sid],refs[sid],c['possible_nearest_indices'],p)
        assert old==baseline['cases'][i]['new']
        new=helper.evaluate(z,opt.x,jobs[sid],refs[sid],c['possible_nearest_indices'],p)
        lo,hi=map(lambda a:F(str(a)),c['reference_seconds']);a,b=map(lambda t:F(str(t)),[old['selected_seconds'],new['selected_seconds']])
        points={lo,hi,*[t for t in [a,b] if lo<=t<=hi]};changes=[abs(b-t)-abs(a-t) for t in points]
        delta=[min(changes),max(changes)];mean_delta=[u+v/4 for u,v in zip(mean_delta,delta)]
        scores=z@opt.x;inside=max(c['possible_nearest_indices'],key=lambda j:(scores[j],-j))
        outside=max([j for j in range(12) if j not in c['possible_nearest_indices']],key=lambda j:(scores[j],-j))
        con=(z[inside]-z[outside])*opt.x
        rows.append({**c,'old':old,'new':new,'paired_error_change_range':[float(t) for t in delta],
                     'paired_error_change_exact':[str(t) for t in delta],
                     'new_inside_minus_outside_decomposition':{'appearance_pca':float(con[:4].sum()),'previous_difference':float(con[4:8].sum()),
                     'following_difference':float(con[8:12].sum()),'relative_time':float(con[12]),'first_indicator':float(con[13]),'sum':float(con.sum())}})
    summary={}
    for version in ['old','new']:
        ev=[r[version] for r in rows]
        summary[version]={'possible_nearest_top1':sum(v['in_possible_nearest_set'] for v in ev),
                          'within_0_3_counts':{s:sum(v['within_0_3_status']==s for v in ev) for s in ['correct_all_reference_times','wrong_all_reference_times','indeterminate']},
                          'mean_error_lower':float(np.mean([v['absolute_error_seconds'][0] for v in ev])),
                          'mean_error_upper':float(np.mean([v['absolute_error_seconds'][1] for v in ev]))}
    assert all(helper.sha(ROOT/p)==h for p,h in frozen.items())
    result={'completed_utc':datetime.now(timezone.utc).isoformat(),'platform':platform.platform(),'python':sys.version,'numpy':np.__version__,'scipy':scipy.__version__,
            'cases':rows,'summary':summary,'fit':fit,'old_unrestricted_objective':record(state['weights'],diffs,False),
            'paired_mean_error_change_range':[float(t) for t in mean_delta],'paired_mean_error_change_exact':[str(t) for t in mean_delta],
            'training_runs':1,'ccd_evaluations':0,'encoder_forwards':0,'followup_runs':0,'submission_changed':False,'candidate_adopted':False,'frozen_files_preserved':len(frozen)}
    write('result.json',result)
    print(json.dumps({'summary':summary,'fit':fit,'paired_mean_error_change_range':result['paired_mean_error_change_range']},indent=2))
    for c in rows:print(c['ID'],{v:{k:c[v][k] for k in ['selected_frame','inside_minus_outside','absolute_error_seconds']} for v in ['old','new']})

if __name__=='__main__':main()
