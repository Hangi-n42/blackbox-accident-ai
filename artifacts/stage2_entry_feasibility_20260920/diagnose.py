"""Train-only expressibility diagnostic. No encoder, refit, or evaluation split access."""
import hashlib
import importlib.util
import itertools
import json
import platform
import sys
from datetime import datetime, timezone
from fractions import Fraction as F
from pathlib import Path

import numpy as np
import scipy
from scipy.optimize import linprog

ROOT = Path(__file__).resolve().parents[2]
OLD = ROOT / 'artifacts/stage2_entry_selector_20260920'
OUT = Path(__file__).resolve().parent
IDS = ['00000', '00003', '00006', '00013']

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def write(name, value):
    (OUT / name).write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')

def possible_nearest(times, reference):
    ts = list(map(lambda t: F(str(t)), times))
    lo, hi = map(lambda t: F(str(t)), reference)
    assert all(a < b for a, b in zip(ts, ts[1:]))
    cells = []
    for k, t in enumerate(ts):
        left = max(lo, (ts[k-1]+t)/2) if k else lo
        right = min(hi, (t+ts[k+1])/2) if k+1 < len(ts) else hi
        if left <= right:
            cells.append({'index': k, 'true_time_support': [float(left), float(right)],
                          'exact_support': [str(left), str(right)]})
    return cells

def objective(x, w, refs):
    loss = 0.0
    grad = w.copy()  # Original lambda = 1, lambda * ||w||^2 / 2.
    pair_accuracy = []
    for z, ref in zip(x, refs):
        pairs = np.asarray(ref['preference_pairs'])
        d = z[pairs[:, 0]] - z[pairs[:, 1]]
        margin = d @ w
        loss += np.logaddexp(0, -margin).mean() / len(refs)
        grad -= (d.T @ np.exp(-np.logaddexp(0, margin))) / len(d) / len(refs)
        pair_accuracy.append(float((margin > 0).mean()))
    penalty = float(w @ w / 2)
    return {'pairwise_logistic_loss': float(loss), 'l2_penalty': penalty,
            'original_total_objective': float(loss + penalty),
            'gradient_l2': float(np.linalg.norm(grad)),
            'strong_convexity_suboptimality_upper_bound': float(grad @ grad / 2),
            'pair_accuracy_by_incident': pair_accuracy,
            'mean_incident_pair_accuracy': float(np.mean(pair_accuracy))}

def main():
    assert not (OUT / 'result.json').exists(), 'Do not overwrite a completed diagnostic'
    assert platform.system() == 'Darwin' and platform.machine() == 'arm64'
    jobs_by_id = {j['ID']: j for j in json.loads((OLD/'training_inputs.json').read_text())['jobs']}
    refs_by_id = {r['ID']: r for r in json.loads((OLD/'training_references.json').read_text())['cases'] if r['eligible']}
    assert sorted(refs_by_id) == IDS
    jobs, refs = [jobs_by_id[i] for i in IDS], [refs_by_id[i] for i in IDS]
    spec = importlib.util.spec_from_file_location('frozen_selector', OLD/'selector.py')
    selector = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(selector)
    with np.load(OLD/'model.npz') as archive:
        state = {k: archive[k] for k in archive.files}
    assert sha(OLD/'model.npz') == '13122d0b662d254d02fdcd6f7f9ad816e848d9155c3ea7dfb73c1e0504e6afde'
    embeddings = np.stack([np.load(OLD/'features'/f'{i}.npy') for i in IDS])
    times = np.asarray([j['times'] for j in jobs])
    x = selector.design(embeddings, times, state)
    assert x.shape == (4, 12, 14) and np.isfinite(x).all()
    np.save(OUT/'design.npy', x)
    cases = []
    old_scores = x @ state['weights']
    for j, r, scores in zip(jobs, refs, old_scores):
        cells = possible_nearest(j['times'], r['reference_seconds'])
        for cell in cells:
            k = cell['index']; t = F(str(j['times'][k])); lo, hi = map(lambda a: F(str(a)), r['reference_seconds'])
            cell.update(frame=j['frames'][k], time=float(t),
                        absolute_error_range=[float(max(lo-t, t-hi, F(0))), float(max(abs(t-lo), abs(t-hi)))],
                        within_0_3_for_some_time=max(lo-t, t-hi, F(0)) <= F(3,10),
                        within_0_3_for_all_times=max(abs(t-lo), abs(t-hi)) <= F(3,10))
        cases.append({'ID': j['ID'], 'reference_seconds': r['reference_seconds'], 'reference_frames': r['reference_frames'],
                      'frames': j['frames'], 'times': j['times'], 'possible_nearest': cells,
                      'old_index': int(np.argmax(scores)), 'old_frame': j['frames'][int(np.argmax(scores))]})
    write('protocol.json', {
        'created_utc': datetime.now(timezone.utc).isoformat(), 'train_ids': IDS,
        'question': 'Can one shared linear score strictly rank one possible-nearest candidate first in each fixed training case?',
        'reference_status': 'Existing human drafts filtered/widened by AI reviewers; weak training references, not adjudicated human truth.',
        'candidate_oracle': 'Closed Voronoi time cell intersects the closed reference interval, computed using exact rational decimal PTS. Not a 0.3-second accuracy target.',
        'frozen_transform': 'Existing selector.design, existing PCA4 and normalization; 14 dimensions; no refit.',
        'primary_lp': 'Enumerate all winner combinations; min L1(w), D w >= 1 for all 44 winner-versus-other comparisons; unrestricted w.',
        'fallback_only_if_no_strict_witness': 'Tie-policy feasibility: D w >= 1 against earlier indices, D w >= 0 against later indices, matching first-index argmax in exact arithmetic.',
        'solver': 'scipy linprog highs; primal and dual feasibility tolerance 1e-9; verify scores and margins separately.',
        'negative_result_policy': 'No feature-impossibility conclusion from solver failures or uncertified infeasibility. Return unresolved for independent audit.',
        'scope': 'No CCD reads, no visual model forwards, no production model changes; diagnostic witness is not an adopted candidate.',
        'secondary_analysis': 'Evaluate the unchanged original objective and gradient at old and diagnostic weights, without another training run.',
        'cases': cases})
    paths = [OLD/'selector.py', OLD/'training_inputs.json', OLD/'training_references.json', OLD/'model.npz',
             *[OLD/'features'/f'{i}.npy' for i in IDS], Path(__file__), OUT/'protocol.json', OUT/'design.npy']
    frozen = {str(p.relative_to(ROOT)): sha(p) for p in paths}
    write('freeze.json', {'created_utc': datetime.now(timezone.utc).isoformat(), 'sha256': frozen})
    combos = list(itertools.product(*[[c['index'] for c in row['possible_nearest']] for row in cases]))
    runs = []
    def solve(winners, strict):
        comparisons = [(i, k, j) for i, k in enumerate(winners) for j in range(12) if j != k]
        d = np.asarray([x[i,k]-x[i,j] for i,k,j in comparisons], dtype=np.float64)
        rhs = np.asarray([1 if strict or j < k else 0 for i,k,j in comparisons], dtype=np.float64)
        # w = positive - negative, so sum(positive + negative) minimizes L1.
        opt = linprog(np.ones(28), A_ub=np.column_stack([-d,d]), b_ub=-rhs,
                      bounds=(0,None), method='highs', options={'primal_feasibility_tolerance':1e-9, 'dual_feasibility_tolerance':1e-9})
        row = {'winners': list(winners), 'strict': strict, 'status': int(opt.status), 'message': opt.message, 'success': bool(opt.success)}
        if opt.success:
            w = opt.x[:14]-opt.x[14:]
            scores = x @ w
            actual = np.argmax(scores, axis=1)
            margins = d @ w
            scale = float(np.linalg.norm(w))
            # Independent extended-precision arithmetic over the saved float features.
            wide = np.sum(d.astype(np.longdouble)*w.astype(np.longdouble), axis=1)
            verified = np.array_equal(actual, winners) and float(np.min(margins-rhs)) >= -1e-7
            if strict: verified = verified and float(np.min(wide)) > 0
            row.update(weights=w.tolist(), scores=scores.tolist(), actual_indices=actual.tolist(),
                       actual_frames=[j['frames'][int(k)] for j,k in zip(jobs,actual)],
                       min_score_margin=float(margins.min()), min_extended_precision_margin=float(wide.min()),
                       min_constraint_slack=float((margins-rhs).min()), weight_l1=float(np.abs(w).sum()), weight_l2=scale,
                       min_unit_l2_weight_margin=float(margins.min()/scale), verified=bool(verified),
                       original_objective=objective(x,w,refs))
        runs.append(row)
    for combo in combos: solve(combo, True)
    if not any(r.get('verified') for r in runs):
        for combo in combos: solve(combo, False)
    unchanged = all(sha(ROOT/p) == h for p,h in frozen.items())
    assert unchanged
    result = {'completed_utc': datetime.now(timezone.utc).isoformat(),
              'platform': platform.platform(), 'machine': platform.machine(), 'python': sys.version,
              'numpy': np.__version__, 'scipy': scipy.__version__, 'feature_shape': list(x.shape), 'feature_dtype': str(x.dtype),
              'cases': cases, 'combination_count': len(combos), 'lp_solver_calls': len(runs),
              'strict_verified_combinations': sum(r['strict'] and r.get('verified',False) for r in runs),
              'conclusion': 'FEASIBLE_STRICT_TRAIN_TOP1' if any(r['strict'] and r.get('verified') for r in runs) else 'REQUIRES_INDEPENDENT_ADJUDICATION',
              'original_weight_l2': float(np.linalg.norm(state['weights'])), 'original_objective': objective(x,state['weights'],refs),
              'runs': runs, 'frozen_files_unchanged': unchanged, 'encoder_forwards': 0, 'ccd_evaluations': 0, 'training_runs': 0,
              'submission_changed': False, 'witness_adopted': False}
    write('result.json', result)
    print(json.dumps({k:result[k] for k in ['conclusion','combination_count','lp_solver_calls','strict_verified_combinations','original_objective']}, indent=2))
    for r in runs:
        print({k:v for k,v in r.items() if k not in ['weights','scores','message']})

if __name__ == '__main__': main()
