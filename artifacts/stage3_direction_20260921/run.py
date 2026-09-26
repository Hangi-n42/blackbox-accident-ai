"""One fixed acceleration-only pixel-reflection training control; no deployment."""
from pathlib import Path
import importlib.util, sys, time, copy, warnings
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import cv2, joblib
from sklearn.base import clone
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.exceptions import ConvergenceWarning
from threadpoolctl import threadpool_limits

O = Path(__file__).resolve().parent
R = O.parents[1]
def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m

roi = module('roi_prior', R/'artifacts/stage3_roi_augmentation_20260920/run.py')
ctrl, old = roi.ctrl, roi.old
read, write, sha = ctrl.read, ctrl.write, ctrl.sha
flow = module('flow_prior', R/'artifacts/stage3_factor_comparison_20260918/run.py')
roi.O = O
DEADLINE = datetime(2026, 9, 21, 6, 50, 34, tzinfo=timezone.utc).timestamp()

def check_time(stage_deadline=None):
    assert time.time() < DEADLINE, '45-minute overall budget exhausted; stop, no extensions'
    if stage_deadline is not None:
        assert time.monotonic() < stage_deadline, 'Fixed stage time budget exhausted'

def freeze():
    assert not (O/'freeze.json').exists()
    cs, z, contexts, _ = ctrl.load()
    files = dict(read(R/'artifacts/stage3_zod_controls_20260920/freeze.json')['inputs'])
    for p in [Path(__file__), R/'artifacts/stage3_roi_augmentation_20260920/run.py',
              R/'artifacts/stage3_factor_comparison_20260918/run.py',
              R/'model/stage3/motion_model.joblib', O/'PROTOCOL.md',
              O/'candidate_review.md', O/'adversarial_review.md',
              Path(ctrl.__file__), Path(old.__file__), Path(old.prior.__file__),
              flow.B/'cases.json']:
        files[str(p.relative_to(R))] = sha(p)
    write(O/'freeze.json', {
        'utc': datetime.now(timezone.utc).isoformat(), 'deadline_utc': '2026-09-21T06:50:34Z',
        'max_candidate_experiments': 1, 'max_feasibility_seconds': 300,
        'max_feature_extraction_seconds': 900, 'fits': 9,
        'candidate': 'Real grayscale width256 frame horizontal reflection before DIS; training original0.5 plus reflected0.5. Acceleration only. Original input at inference.',
        'fixed': 'Same2395 comma rows and public40 perLOVO; exact original per-fold scaler; same logistic settings, effective class and per-example loss mass, labels, split, steering. No ZOD training.',
        'gates': {'public_S3': 'strictly greater than same-condition research baseline',
                  'date_macro_F1': 'no decrease in each existing date fold',
                  'new_opposite': 'zero new A-to-D or D-to-A on every public/comma/ZOD diagnostic row; no tolerance change',
                  'source_opposite': 'nonincrease per source and direction, implied by zero new opposite',
                  'reserved': 'Only if all development gates pass; fixed18window centers, same external-only baseline/candidate plus production reference. No fit or repeat tuning. Each route fixed A/D/C three-class F1 nondecrease and zero new opposite versus both references. Also report fixed4class F1 with absent STOP; never S3.',
                  'adoption': 'No production overwrite; passing small diagnostic gates is insufficient to prove official/private improvement'},
        'contexts': [{'name': c['name'], 'selection': c['selection'], 'base': str(c['base'].relative_to(R))} for c in contexts],
        'inputs': files,
        'raw_video_stat': {c['raw_path']: {'bytes': (R/c['raw_path']).stat().st_size,
                                          'mtime_ns': (R/c['raw_path']).stat().st_mtime_ns} for c in flow.cases()}})
    print('Frozen one candidate, nine fixed training contexts; reserved unopened', flush=True)

def protected():
    f = read(O/'freeze.json')
    return all(sha(R/p) == h for p, h in f['inputs'].items()) and all(
        (R/p).stat().st_size == s['bytes'] and (R/p).stat().st_mtime_ns == s['mtime_ns']
        for p, s in f['raw_video_stat'].items())

def extract_case(c, original=False, stage_deadline=None):
    start = time.monotonic()
    images = []
    for im in flow.frames(c):
        check_time(stage_deadline)
        images.append(im)
    assert len(images) >= 2
    results = {}
    for mode in ['original', 'flip'] if original else ['flip']:
        dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_FAST)
        ff = flow.FrameFeatureComputer()
        frames = images if mode == 'original' else [cv2.flip(im, 1) for im in images]
        if mode == 'flip':
            assert all(np.array_equal(cv2.flip(im, 1), src) for im, src in zip(frames, images))
        raw = []
        for a, b in zip(frames, frames[1:]):
            check_time(stage_deadline)
            raw.append(ff(dis.calc(a, b, None)*10))
        results[mode] = flow.combine(raw, len(images))
    cached = np.load(old.prior.F/(c['id']+'.npz'))['base']
    assert results['flip'].shape == cached.shape and np.isfinite(results['flip']).all()
    if original:
        assert np.array_equal(results['original'], cached), (c['id'], 'original extractor/cache mismatch')
    target = O/'features'/(c['id']+'.npz')
    assert not target.exists()
    np.savez_compressed(target, base=results['flip'])
    log = {'id': c['id'], 'frames': len(images), 'seconds': time.monotonic()-start,
           'source': c['raw_path'], 'feature_sha256': sha(target),
           'original_cache_bit_exact': True if original else None,
           'reflection_twice_identity': True, 'feature_dim': 864}
    write(O/'features'/(c['id']+'_timing.json'), log)
    print(c['id'], 'reflected features', round(log['seconds'], 2), 'sec', flush=True)
    return log

def smoke():
    assert protected()
    (O/'features').mkdir()
    start = time.monotonic()
    cases = flow.cases()
    logs = [extract_case(next(c for c in cases if c['public']), True, start+300),
            extract_case(next(c for c in cases if not c['public']), True, start+300)]
    elapsed = time.monotonic()-start
    assert elapsed <= 300, 'Feasibility exceeded five minutes'
    estimate = max(r['seconds']/2 for r in logs)*len(cases)
    assert estimate <= 900, 'Feature work estimate exceeds fixed15min sub-budget'
    write(O/'feasibility.json', {'pass': True, 'seconds': elapsed, 'cases': logs,
         'conservative_feature_estimate_seconds': estimate, 'fit_count': 0,
         'predictions_inspected': False, 'original_parity': True, 'reserved_used': False})

def extract():
    assert read(O/'feasibility.json')['pass'] and protected()
    start = time.monotonic()
    for c in flow.cases():
        check_time()
        assert time.monotonic()-start <= 900, 'Fixed extraction sub-budget exceeded'
        if not (O/'features'/(c['id']+'.npz')).exists():
            extract_case(c, stage_deadline=start+900)
    write(O/'extraction.json', {'seconds': time.monotonic()-start,
          'clips': len(flow.cases()), 'feature_files': {p.name: sha(p) for p in (O/'features').glob('*.npz')},
          'protected_unchanged': protected(), 'reserved_used': False})

def train():
    assert protected() and read(O/'extraction.json')['clips'] == 28
    cs, z, contexts, _ = ctrl.load()
    df = pd.read_csv(ctrl.B/'comparison_predictions.csv', dtype={'id': str})
    rows, logs = [], []
    dest = O/'models'; dest.mkdir()
    flipped = {sid: np.load(O/'features'/(sid+'.npz'))['base'] for sid in cs}
    for c in contexts:
        check_time()
        name, sel = c['name'], c['selection']
        base = joblib.load(c['base'])
        x = np.stack([cs[s]['x'][i] for s, i in sel])
        xf = np.stack([flipped[s][i] for s, i in sel])
        y = np.array([cs[s]['y'][i] for s, i in sel]); n = len(y)
        xx = np.concatenate([base[0].transform(x), base[0].transform(xf)])
        yy = np.tile(y, 2); sw = np.full(2*n, .5)
        bw = compute_sample_weight('balanced', y)
        eff = sw*compute_sample_weight('balanced', yy)
        assert np.allclose(eff.reshape(2, n).sum(0), bw)
        masses = [float(eff[yy == k].sum()) for k in range(4)]
        assert np.allclose(masses, n/4) and sw.sum() == n
        m = copy.deepcopy(base); m.steps[-1] = (m.steps[-1][0], clone(base[-1]))
        start = time.monotonic()
        with warnings.catch_warnings(record=True) as ws:
            warnings.simplefilter('always')
            m[-1].fit(xx, yy, sample_weight=sw)
        assert not any(issubclass(w.category, ConvergenceWarning) for w in ws)
        assert m[-1].get_params() == base[-1].get_params()
        for attr in ['mean_', 'scale_', 'var_', 'n_samples_seen_']:
            assert np.array_equal(getattr(m[0], attr), getattr(base[0], attr))
        joblib.dump(m, dest/(name+'.joblib'))
        logs.append({'name': name, 'n_original': n, 'n_augmented': 2*n,
                     'sample_weight_sum': float(sw.sum()), 'class_effective_masses': masses,
                     'seconds': time.monotonic()-start, 'warnings': [str(w.message) for w in ws]})
        part = df[df.scope == 'zod_heldout'] if name == 'zod' else df[df.id == name] if name.startswith('OPEN') else df[df.scope == name]
        for sid, g in part.groupby('id', sort=False):
            data = z[sid]['x'] if name == 'zod' else cs[sid]['x']
            bp, cp = base.predict_proba(data), m.predict_proba(data)
            cache = np.load(ctrl.B/'models'/(sid+'_heldout.npz' if name == 'zod' else name+'_'+sid+'.npz'))
            assert np.array_equal(bp, cache['baseline']), (name, sid)
            ix = g.sample_index.to_numpy(int)
            assert np.array_equal(bp[ix].argmax(1), g.baseline)
            assert np.isfinite(cp).all() and np.allclose(cp.sum(1), 1)
            np.savez_compressed(dest/(name+'_'+sid+'.npz'), prob=cp)
            gg = g.drop(columns=['candidate']+[f'candidate_p{k}' for k in range(4)]+['new_opposite', 'fixed_opposite', 'corrected', 'new_wrong']).copy()
            gg['candidate'] = cp[ix].argmax(1)
            for k in range(4): gg[f'candidate_p{k}'] = cp[ix, k]
            rows.append(gg)
        write(O/'execution.json', logs)
        print(name, 'fit and predict done', round(logs[-1]['seconds'], 2), 'sec', flush=True)
    pd.concat(rows, ignore_index=True).to_csv(O/'predictions.csv', index=False)
    assert protected()
    write(O/'train_checks.json', {'fits': len(logs), 'baseline_probability_replay_bit_exact': True,
          'class_total_and_per_sample_mass_preserved': True, 'classifier_params_and_scaler_unchanged': True,
          'protected_inputs_unchanged': True, 'reserved_used': False})

def evaluate():
    roi.evaluate()
    df = pd.read_csv(O/'predictions.csv', dtype={'id': str})
    source_metrics, source_changes = [], []
    for (scope, sid), g in df.groupby(['scope', 'id'], sort=False):
        for variant in ['baseline', 'candidate']:
            source_metrics.append({'scope': scope, 'id': sid, 'group': g.group.iloc[0],
                                   'variant': variant, **old.metric(g.truth, g[variant])})
        b, a = old.opposite(g.truth, g.baseline), old.opposite(g.truth, g.candidate)
        source_changes.append({'scope': scope, 'id': sid, 'new_opposite': int((a & ~b).sum()),
            'resolved_to_correct': int((b & (g.candidate == g.truth)).sum()),
            'resolved_to_constant_wrong': int((b & (g.candidate == 2)).sum()),
            'corrected': int(((g.baseline != g.truth) & (g.candidate == g.truth)).sum()),
            'new_wrong': int(((g.baseline == g.truth) & (g.candidate != g.truth)).sum())})
    write(O/'source_metrics.json', source_metrics); write(O/'source_changes.json', source_changes)
    decision = read(O/'decision.json')
    write(O/'reserved_status.json', {'used': False,
          'next_action': 'eligible_for_one_fixed_reserved_comparison' if decision['all_pass'] else 'preserve_unopened_candidate_failed_development_gate',
          'rule_fixed_before_evaluation': True})
    assert protected()
    write(O/'final_checks.json', {'protected_inputs_unchanged': True, 'experiment_settings': 1,
          'candidate_fits': 9, 'evaluation_rows': len(df), 'production_replaced': False,
          'current_utc': datetime.now(timezone.utc).isoformat()})

if __name__ == '__main__':
    cv2.setNumThreads(2)
    with threadpool_limits(limits=2):
        check_time()
        {'freeze': freeze, 'smoke': smoke, 'extract': extract, 'train': train, 'evaluate': evaluate}[sys.argv[1]]()
