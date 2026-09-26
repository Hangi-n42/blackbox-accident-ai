"""One train-only fit with log-mean-exp pair aggregation; no other change."""
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
from scipy.special import logsumexp

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
BASE=ROOT/'artifacts/stage2_entry_l2_01_20260921'
spec=importlib.util.spec_from_file_location('baseline',BASE/'run.py')
base=importlib.util.module_from_spec(spec);spec.loader.exec_module(base)
helper=base.helper

def write(name,obj):
    (HERE/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n')

def objective(w,diffs):
    loss=.1*np.dot(w,w)/2;grad=.1*w.copy()
    for D in diffs:
        a=-D@w-np.log(len(D))
        value=logsumexp(np.r_[0.,a])
        loss+=value/len(diffs)
        grad-=(D.T@np.exp(a-value))/len(diffs)
    return float(loss),grad

def loss_record(w,diffs,kind):
    loss,g=objective(w,diffs) if kind=='hard' else base.objective(w,diffs,.1)
    return {'aggregation':kind,'total':loss,'data_loss':float(loss-.05*(w@w)),
            'l2_penalty':float(.05*(w@w)),'weight_l2':float(np.linalg.norm(w)),
            'gradient_l2':float(np.linalg.norm(g)),'gradient_inf':float(np.abs(g).max()),
            'numerical_gap_bound':float(g@g/.2),
            'pair_accuracy_per_incident':[float(np.mean(D@w>0)) for D in diffs]}

def pair_weight_analysis(w,diffs,cases):
    rows=[]
    for D,c in zip(diffs,cases):
        m=D@w;n=len(m);a=-m-np.log(n);v=float(logsumexp(np.r_[0.,a]))
        old=np.exp(-np.logaddexp(0,m))/(4*n);new=np.exp(a-v)/4
        pairs=[]
        for (i,j),margin,b,h in zip(c['top_pairs'],m,old,new):
            pairs.append({'preferred_frame':c['frames'][i],'other_frame':c['frames'][j],'margin':float(margin),
                          'old_gradient_coefficient':float(b),'hard_gradient_coefficient':float(h),
                          'old_within_incident_share':float(b/old.sum()),'hard_within_incident_share':float(h/new.sum())})
        rows.append({'ID':c['ID'],'old_incident_loss':float(np.logaddexp(0,-m).mean()),'hard_incident_loss':v,
                     'old_coefficient_sum':float(old.sum()),'hard_coefficient_sum':float(new.sum()),'pairs':pairs})
    return rows

def main():
    assert platform.system()=='Darwin' and platform.machine()=='arm64'
    assert not (HERE/'protocol.json').exists(),'One fit; preserve outputs'
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
    diffs=[z[p[:,0]]-z[p[:,1]] for z,p in zip(x,pairs)];initial=np.zeros(14)
    a,g=objective(initial,diffs);b,h=base.objective(initial,diffs,.1)
    np.testing.assert_allclose(a,b,rtol=0,atol=2e-15);np.testing.assert_allclose(g,h,rtol=0,atol=2e-15)
    protocol={'created_utc':datetime.now(timezone.utc).isoformat(),'train_ids':helper.IDS,'cases':cases,
              'baseline':str(BASE.relative_to(ROOT)),'L2':.1,'sole_change':'Incident loss mean softplus(-margin) -> log(1+mean exp(-margin)).',
              'fixed':'All14 coefficients unrestricted; frozen14features,PCA,normalization,12candidates,PTS,weak reference intervals,53pairs,equal incident weight.',
              'formula':'Li=logsumexp([0,-D_i w-log n_i]); L=mean_i Li+.1||w||²/2',
              'gradient':'mean_i[-D_i.T exp(-D_i w-log n_i-Li)]+.1w',
              'initialization':'zeros(14), no warm start','optimizer':'L-BFGS-B','jac':True,'bounds':None,
              'options':helper.OPTIONS,'fits_authorized':1,
              'scope':'Train4 only; no CCD, encoder, submission changes or followup trial.',
              'evaluation':'Frames,possible-nearest membership,inside-outside gap,full-reference errors,±.3 and same-T paired MAE; loss alone not success.',
              'analysis':'Compare old and hard gradient coefficients at the same baseline weights, also at new weights; fixed-weight score decomposition.'}
    write('protocol.json',protocol)
    paths=[ROOT/p for p in inherited]+[BASE/'model.npz',BASE/'result.json',BASE/'freeze.json',BASE/'STATUS.json',Path(__file__),HERE/'protocol.json']
    frozen={str(p.relative_to(ROOT)):helper.sha(p) for p in paths}
    write('freeze.json',{'created_utc':datetime.now(timezone.utc).isoformat(),'sha256':frozen})
    opt=minimize(lambda w:objective(w,diffs),initial,jac=True,method='L-BFGS-B',options=helper.OPTIONS)
    fit={'success':bool(opt.success),'message':str(opt.message),'iterations':int(opt.nit),'function_evaluations':int(opt.nfev),
         'initial':loss_record(initial,diffs,'hard'),'final':loss_record(opt.x,diffs,'hard')}
    write('fit.json',fit)
    assert opt.success and np.isfinite(opt.fun) and opt.fun<objective(initial,diffs)[0]
    np.savez(HERE/'model.npz',**{**state,'weights':opt.x})
    rows=[];mean_delta=[F(0),F(0)]
    for i,(z,c,p) in enumerate(zip(x,cases,pairs)):
        sid=c['ID'];old=helper.evaluate(z,state['weights'],jobs[sid],refs[sid],c['possible_nearest_indices'],p)
        assert old==baseline['cases'][i]['new']
        new=helper.evaluate(z,opt.x,jobs[sid],refs[sid],c['possible_nearest_indices'],p)
        lo,hi=map(lambda v:F(str(v)),c['reference_seconds']);a,b=map(lambda v:F(str(v)),[old['selected_seconds'],new['selected_seconds']])
        points={lo,hi,*[t for t in [a,b] if lo<=t<=hi]};changes=[abs(b-t)-abs(a-t) for t in points]
        delta=[min(changes),max(changes)];mean_delta=[s+t/4 for s,t in zip(mean_delta,delta)]
        decomposed={}
        for label,w in [('old',state['weights']),('new',opt.x)]:
            s=z@w;inside=max(c['possible_nearest_indices'],key=lambda j:(s[j],-j))
            outside=max([j for j in range(12) if j not in c['possible_nearest_indices']],key=lambda j:(s[j],-j))
            v=(z[inside]-z[outside])*w
            decomposed[label]={'appearance_pca':float(v[:4].sum()),'previous_difference':float(v[4:8].sum()),
                               'following_difference':float(v[8:12].sum()),'relative_time':float(v[12]),'first_indicator':float(v[13]),'sum':float(v.sum())}
        rows.append({**c,'old':old,'new':new,'paired_error_change_range':[float(t) for t in delta],
                     'paired_error_change_exact':[str(t) for t in delta],'score_decomposition':decomposed})
    summary={}
    for version in ['old','new']:
        ev=[c[version] for c in rows]
        summary[version]={'possible_nearest_top1':sum(v['in_possible_nearest_set'] for v in ev),
                          'within_0_3_counts':{s:sum(v['within_0_3_status']==s for v in ev) for s in ['correct_all_reference_times','wrong_all_reference_times','indeterminate']},
                          'mean_error_lower':float(np.mean([v['absolute_error_seconds'][0] for v in ev])),
                          'mean_error_upper':float(np.mean([v['absolute_error_seconds'][1] for v in ev]))}
    assert all(helper.sha(ROOT/p)==h for p,h in frozen.items())
    result={'completed_utc':datetime.now(timezone.utc).isoformat(),'platform':platform.platform(),'python':sys.version,'numpy':np.__version__,'scipy':scipy.__version__,
            'cases':rows,'summary':summary,'fit':fit,'paired_mean_error_change_range':[float(t) for t in mean_delta],
            'paired_mean_error_change_exact':[str(t) for t in mean_delta],
            'objectives':{f'{name}_weights_{kind}_objective':loss_record(w,diffs,kind) for name,w in [('old',state['weights']),('new',opt.x)] for kind in ['old','hard']},
            'gradient_weight_analysis':{'at_old_weights':pair_weight_analysis(state['weights'],diffs,cases),'at_new_weights':pair_weight_analysis(opt.x,diffs,cases)},
            'training_runs':1,'ccd_evaluations':0,'encoder_forwards':0,'followup_runs':0,'submission_changed':False,'candidate_adopted':False,'frozen_files_preserved':len(frozen)}
    write('result.json',result)
    print(json.dumps({'summary':summary,'fit':fit,'paired_mean_error_change_range':result['paired_mean_error_change_range']},indent=2))
    for c in rows:print(c['ID'],{v:{k:c[v][k] for k in ['selected_frame','inside_minus_outside','absolute_error_seconds']} for v in ['old','new']})

if __name__=='__main__':main()
