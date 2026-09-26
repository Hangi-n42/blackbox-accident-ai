"""One train-only fit changing preference-pair membership, with frozen features."""
import hashlib
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

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OLD = ROOT / 'artifacts/stage2_entry_selector_20260920'
PRIOR = ROOT / 'artifacts/stage2_entry_feasibility_20260920'
IDS = ['00000', '00003', '00006', '00013']
OPTIONS = {'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8, 'maxls': 50}

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def write(name, obj):
    (HERE/name).write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False)+'\n')

def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def objective(w, diffs):
    # Same lambda, order of summation, incident means and gradient as selector.fit.
    loss = np.dot(w, w)/2
    grad = w.copy()
    for d in diffs:
        margin = d @ w
        loss += np.logaddexp(0, -margin).mean()/len(diffs)
        grad -= (d.T @ np.exp(-np.logaddexp(0, margin)))/len(d)/len(diffs)
    return float(loss), grad

def loss_record(w, diffs):
    loss, grad = objective(w, diffs)
    return {'total': loss, 'l2_penalty': float(w@w/2), 'data_loss': float(loss-w@w/2),
            'gradient_l2': float(np.linalg.norm(grad)), 'gradient_inf': float(np.abs(grad).max()),
            'numerical_strong_convexity_gap_bound': float(grad@grad/2),
            'pair_accuracy_per_incident': [float(np.mean(d@w>0)) for d in diffs]}

def evaluate(z, w, job, ref, allowed, pairs):
    scores = z@w
    k = int(np.argmax(scores))
    outside = [j for j in range(12) if j not in allowed]
    best_in = max(allowed, key=lambda j: (scores[j], -j))
    best_out = max(outside, key=lambda j: (scores[j], -j))
    t = F(str(job['times'][k])); lo, hi = map(lambda t: F(str(t)), ref['reference_seconds'])
    emin = max(lo-t, t-hi, F(0)); emax = max(abs(t-lo), abs(t-hi))
    oracle_worst = min(max(abs(F(str(a))-lo),abs(F(str(a))-hi)) for a in job['times'])
    pair_margins = (z[pairs[:,0]]-z[pairs[:,1]])@w
    return {'selected_index': k, 'selected_frame': job['frames'][k], 'selected_seconds': float(t),
            'in_possible_nearest_set': k in allowed, 'scores': scores.tolist(),
            'best_inside_frame': job['frames'][best_in], 'best_outside_frame': job['frames'][best_out],
            'max_inside_score': float(scores[best_in]), 'max_outside_score': float(scores[best_out]),
            'inside_minus_outside': float(scores[best_in]-scores[best_out]),
            'absolute_error_seconds': [float(emin), float(emax)],
            'within_0_3_status': 'correct_all_reference_times' if emax<=F(3,10) else 'wrong_all_reference_times' if emin>F(3,10) else 'indeterminate',
            'minimax_regret_seconds': float(emax-oracle_worst),
            'top_pairs_positive': int(np.sum(pair_margins>0)), 'top_pairs_nonpositive': int(np.sum(pair_margins<=0))}

def main():
    assert platform.system()=='Darwin' and platform.machine()=='arm64'
    assert not (HERE/'protocol.json').exists(), 'One registered fit only; do not overwrite'
    selector = load_module('selector_frozen', OLD/'selector.py')
    diagnostic = load_module('feasibility_helpers', PRIOR/'diagnose.py')
    jobs_by_id = {j['ID']:j for j in json.loads((OLD/'training_inputs.json').read_text())['jobs']}
    refs_by_id = {r['ID']:r for r in json.loads((OLD/'training_references.json').read_text())['cases'] if r['eligible']}
    assert sorted(refs_by_id)==IDS
    jobs, refs = [jobs_by_id[i] for i in IDS], [refs_by_id[i] for i in IDS]
    state = dict(np.load(OLD/'model.npz', allow_pickle=False))
    embeddings = np.stack([np.load(OLD/'features'/f'{i}.npy', allow_pickle=False) for i in IDS])
    x = selector.design(embeddings, np.array([j['times'] for j in jobs]), state)
    np.testing.assert_array_equal(x, np.load(PRIOR/'design.npy', allow_pickle=False))
    assert sha(OLD/'model.npz')=='13122d0b662d254d02fdcd6f7f9ad816e848d9155c3ea7dfb73c1e0504e6afde'
    # Check allowed original artifacts against the previous diagnostic's freeze.
    previous = json.loads((PRIOR/'freeze.json').read_text())['sha256']
    inherited = [OLD/'selector.py', OLD/'training_inputs.json', OLD/'training_references.json', OLD/'model.npz',
                 *[OLD/'features'/f'{i}.npy' for i in IDS], PRIOR/'diagnose.py', PRIOR/'design.npy']
    for path in inherited:
        assert sha(path)==previous[str(path.relative_to(ROOT))]
    all_pairs, top_pairs, records = [], [], []
    for job, ref in zip(jobs, refs):
        p = selector.certain_pairs(job['times'], *ref['reference_seconds'])
        assert p.tolist()==ref['preference_pairs']
        cells = diagnostic.possible_nearest(job['times'], ref['reference_seconds'])
        allowed = [c['index'] for c in cells]
        top = np.asarray([(a,b) for a,b in p if a in allowed and b not in allowed], dtype=int)
        all_pairs.append(p); top_pairs.append(top)
        records.append({'ID':job['ID'], 'reference_seconds':ref['reference_seconds'],
                        'reference_frames':ref['reference_frames'], 'frames':job['frames'], 'times':job['times'],
                        'possible_nearest_indices':allowed, 'possible_nearest_frames':[job['frames'][k] for k in allowed],
                        'old_pair_count':len(p), 'top_pairs':top.tolist(), 'top_pair_count':len(top)})
    assert [len(p) for p in all_pairs]==[61,66,66,65]
    assert [len(p) for p in top_pairs]==[11,11,11,20]
    assert not any(a in (10,11) and b in (10,11) for a,b in top_pairs[3])
    protocol = {'created_utc':datetime.now(timezone.utc).isoformat(), 'train_ids':IDS, 'cases':records,
                'sole_change':'Use P_top={(i,j) in frozen certain P: i in possible-nearest M, j not in M}; 258 to 53 pairs.',
                'fixed':'Original embeddings, PCA, normalization,14 features,12 candidates,PTS,weak reference intervals,linear model.',
                'reference_status':'Human drafts filtered/widened by AI reviewers; weak training references, not adjudicated human GT.',
                'loss':'Mean pair logistic loss per incident, then equal mean over4 incidents + 0.5||w||²',
                'L2':1.0, 'initialization':'zeros(14), not LP witness or old weights',
                'optimizer':'L-BFGS-B', 'jac':True, 'options':OPTIONS, 'fits_authorized':1,
                'evaluation':'Only train4; argmax first-index ties; selected frame, possible-nearest membership, max inside minus max outside, full reference error range and ±0.3 status.',
                'success_rule':'Report top1 membership and time-error changes per case. Loss decrease alone is not success; membership is not ±0.3 correctness.',
                'scope':'No CCD reads/evaluation, no visual forward, no submission changes, no follow-up trial.'}
    write('protocol.json', protocol)
    paths = inherited+[Path(__file__),HERE/'protocol.json',PRIOR/'freeze.json']
    frozen = {str(p.relative_to(ROOT)):sha(p) for p in paths}
    write('freeze.json', {'created_utc':datetime.now(timezone.utc).isoformat(),'sha256':frozen})
    diffs = [z[p[:,0]]-z[p[:,1]] for z,p in zip(x,top_pairs)]
    old_diffs = [z[p[:,0]]-z[p[:,1]] for z,p in zip(x,all_pairs)]
    initial = np.zeros(14)
    opt = minimize(lambda w:objective(w,diffs), initial, jac=True, method='L-BFGS-B', options=OPTIONS)
    fitlog = {'success':bool(opt.success), 'message':str(opt.message), 'iterations':int(opt.nit),
              'function_evaluations':int(opt.nfev), 'initial_new_objective':loss_record(initial,diffs),
              'final_new_objective':loss_record(opt.x,diffs)}
    write('fit.json',fitlog)
    assert opt.success and np.isfinite(opt.fun) and opt.fun<objective(initial,diffs)[0]
    candidate = {**state, 'weights':opt.x}
    np.savez(HERE/'model.npz', **candidate)
    result_cases = []
    for z,job,ref,rec,p in zip(x,jobs,refs,records,top_pairs):
        result_cases.append({**rec, 'old':evaluate(z,state['weights'],job,ref,rec['possible_nearest_indices'],p),
                             'new':evaluate(z,opt.x,job,ref,rec['possible_nearest_indices'],p)})
    summary = {}
    for version in ('old','new'):
        rows=[c[version] for c in result_cases]
        summary[version]={'possible_nearest_top1':sum(r['in_possible_nearest_set'] for r in rows),
                          'within_0_3_counts':{s:sum(r['within_0_3_status']==s for r in rows) for s in ['correct_all_reference_times','wrong_all_reference_times','indeterminate']},
                          'mean_error_lower':float(np.mean([r['absolute_error_seconds'][0] for r in rows])),
                          'mean_error_upper':float(np.mean([r['absolute_error_seconds'][1] for r in rows]))}
    assert all(sha(ROOT/p)==h for p,h in frozen.items())
    write('result.json', {'completed_utc':datetime.now(timezone.utc).isoformat(),'platform':platform.platform(),
                         'python':sys.version,'numpy':np.__version__,'scipy':scipy.__version__, 'cases':result_cases,
                         'summary':summary,'fit':fitlog,'old_weights_on_new_objective':loss_record(state['weights'],diffs),
                         'old_weights_on_old_objective':loss_record(state['weights'],old_diffs),
                         'new_weights_on_old_objective':loss_record(opt.x,old_diffs),
                         'training_runs':1,'ccd_evaluations':0,'encoder_forwards':0,'followup_runs':0,
                         'submission_changed':False,'candidate_adopted':False,'frozen_files_preserved':len(frozen)})
    print(json.dumps({'summary':summary,'fit':fitlog},indent=2))
    for c in result_cases:
        print(c['ID'], {v:{k:c[v][k] for k in ['selected_frame','inside_minus_outside','absolute_error_seconds','within_0_3_status']} for v in ['old','new']})

if __name__=='__main__':main()
