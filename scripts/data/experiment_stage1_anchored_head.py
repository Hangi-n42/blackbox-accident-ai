"""Frozen visual features, matched data, and a pre-registered anchored linear head."""
import os
os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '2')
import argparse
import collections
import hashlib
import importlib.util
import json
import socket
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw
from scipy.optimize import minimize, check_grad
from scipy.special import expit

from diagnose_uhdm_full_robustness import transform

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'artifacts/stage1_anchored_head_20260919'
OLD = ROOT / 'artifacts/stage1_head_pilot_20260919'
REL = ROOT / 'releases/v7/source'
MODES = ['full_frame', 'jpeg_75', 'blur_1']


def read(p):
    return json.loads(p.read_text())


def write(p, v):
    p.write_text(json.dumps(v, indent=2, ensure_ascii=False))


def sha(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda: f.read(1048576), b''):
            h.update(b)
    return h.hexdigest()


def load(name, p):
    spec = importlib.util.spec_from_file_location(name, p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def model():
    torch.set_num_threads(2)
    cv2.setNumThreads(2)
    m = load('anchor_tpo', REL / 'model/stage2/code/solution/stage1_tpo_merged.py')
    return m.TPODetector(REL / 'model/stage1/tpo', device='cpu')


def original_theta():
    a = torch.load(REL / 'model/stage1/tpo/merged_visual.pt', map_location='cpu', weights_only=True)
    assert a['class_order'] == ['RERECORDED', 'ORIGINAL']
    return np.r_[ (a['head']['weight'][0]-a['head']['weight'][1]).numpy(),
                  (a['head']['bias'][0]-a['head']['bias'][1]).item()].astype(np.float64)


def init():
    assert not (OUT / 'protocol.json').exists()
    sources = read(ROOT / 'research/stage1/comma_original_diagnostic/sources.json')['videos']
    dates = sorted({r['route'].split('|')[1][:10] for r in sources}, key=lambda s: hashlib.sha256(s.encode()).hexdigest())
    for r in sources:
        r['date_group'] = r['route'].split('|')[1][:10]
        r['split'] = 'train' if r['date_group'] in dates[:5] else 'development'
    write(OUT / 'comma_plan.json', sources)
    pro = {
        'registered_before_feature_extraction_and_training': True,
        'hypothesis': 'Preserving the V7 decision boundary while learning on multiple sources improves transfer versus a zero-centered replacement head.',
        'methods': {'v7': 'frozen reference', 'ordinary': {'lambda': .001, 'center': 'zero'},
                    'anchored': {'lambdas': [.0001, .001, .01], 'center': 'exact V7 binary logit weight and bias'}},
        'objective': 'weighted mean per-view binary cross entropy + lambda/2 * squared L2 distance of all 513 coefficients including bias from center',
        'optimizer': {'method': 'L-BFGS-B', 'maxiter': 2000, 'ftol': 1e-12, 'gtol': 1e-9},
        'train_mass': {'vd_original': .20, 'vd_recapture': .25, 'dlc_original': .20, 'dlc_recapture': .25, 'comma_original': .10},
        'weighting': 'Fixed source-class mass, equal content groups (comma date groups), then equal recordings, modes and views within group. Overall original/recapture mass .5/.5.',
        'vd_split': 'Existing Seoul train8; Turkey development8, preserving conservative regional separation. Old val iPhone/TCL are legacy guards, not independent holdout.',
        'comma_split': 'First5 SHA256-sorted dates training, remaining6 dates development; no same-date crossing. Same cars/commute geography remain.',
        'dlc_split': read(OUT / 'dlc_plan.json')['types'],
        'input': '12 full frames plus12 horizontal flips, normalized frozen512dim V7 features, arithmetic mean per-view probabilities; threshold .5; no forensic',
        'modes': MODES,
        'selection': 'Anchored candidates only: require no increase in original FP or recapture FN within either VD or DLC development; comma FPR must not increase. Max unweighted mean of VD/DLC development Macro-F1; tie stronger lambda. Strict mean improvement over V7 required. Ordinary is a diagnostic control, not selected.',
        'legacy_guard': 'After selecting one anchored candidate, require no increase in FP/FN separately for old val iPhone, old val TCL, public examples. Public synthetic replays are regression fixtures, not physical recapture confirmation. No retuning after guard failure.',
        'confirmation': 'Only if development and legacy gates pass: fetch and score six unseen DLC documents from three held-out document types with V7 and chosen candidate only. Require no FP/FN increase and strict Macro-F1 improvement; source/type cluster bootstrap reported. This does not establish independent road-recapture generalization.',
        'stop_rule': 'Failed gate -> retain V7, no threshold retuning, no LoRA, no submission. Data acquisition failure -> retain fixed selection; record incomplete scope.',
        'runtime': 'One merged binary linear head; no additional backbone forward. Actual entry point and off-line timing checked only for eligible candidate.',
        'references': ['https://dacon.io/competitions/official/236753/overview/rules', 'https://zenodo.org/records/7467028', 'https://zenodo.org/records/6466770'],
        'limitations': read(OUT / 'dlc_plan.json')['limitations'] + ['Public and prior validation scores have been observed. No leaderboard score optimization in this experiment.', 'No new independent dashcam source/replay pairs acquired.'],
    }
    paths = [Path(__file__), OUT / 'dlc_plan.json', OUT / 'comma_plan.json', ROOT / 'scripts/data/acquire_stage1_anchored_dlc.py']
    paths += [p for p in REL.rglob('*') if p.is_file() and '__pycache__' not in p.parts]
    paths += [p for p in OLD.glob('*_features.npz')] + [p for p in OLD.glob('*_records.json')]
    pro['frozen_sha256'] = {str(p.relative_to(ROOT)): sha(p) for p in paths}
    write(OUT / 'protocol.json', pro)
    np.save(OUT / 'v7_theta.npy', original_theta())
    print('PROTOCOL FROZEN; comma train/development', collections.Counter(r['split'] for r in sources), flush=True)


def features(m, images):
    aa = images + [np.ascontiguousarray(a[:, ::-1, :]) for a in images]
    ff, pp = [], []
    for i in range(0, len(aa), 8):
        batch = []
        for a in aa[i:i+8]:
            im = cv2.resize(a, (m.size, m.size), interpolation=cv2.INTER_LINEAR).astype(np.float32)/255
            batch.append(np.transpose((im-m.mean)/m.std, (2, 0, 1)))
        with torch.inference_mode():
            f = torch.nn.functional.normalize(m.visual(torch.from_numpy(np.stack(batch))).float(), dim=-1)
            pp.extend(m.head(f).softmax(1)[:, 0].tolist())
        ff.append(f.numpy())
    return np.concatenate(ff), float(np.mean(pp))


def extract(domain, split):
    pro = read(OUT / 'protocol.json')
    if split == 'confirmation':
        assert read(OUT / 'selection.json')['gate_pass']
    if domain == 'dlc':
        rows = read(OUT / f'dlc_{split}_acquired.json')
        groups = collections.defaultdict(list)
        for r in rows:
            groups[r['video_id']].append(r)
        jobs = sorted(groups.items())
    elif domain == 'comma':
        helper = load('anchor_comma_helper', ROOT / 'research/stage1/comma_original_diagnostic.py')
        jobs = [(r['segment'], r) for r in read(OUT / 'comma_plan.json') if r['split'] == split]
    else:
        import sys
        sys.path.insert(0, str(REL / 'model/stage2/code'))
        sampler = load('solution.anchor_sampler', REL / 'model/stage2/code/solution/stage1_v4.py')
        provenance = read(OLD / 'public_label_provenance.json')
        jobs = [(p.stem, p) for p in sorted((ROOT / 'artifacts/public_eval/stage1/videos').glob('*.mp4'))]
        assert len(jobs) == 10
    dest = OUT / 'features' / f'{domain}_{split}'
    dest.mkdir(parents=True, exist_ok=True)
    m = model()
    start = time.perf_counter()
    try:
        for ident, data in jobs:
            key = hashlib.sha256(ident.encode()).hexdigest()[:16]
            cache, meta = dest / f'{key}.npz', dest / f'{key}.json'
            if cache.exists() and meta.exists():
                continue
            if domain == 'dlc':
                rs = sorted(data, key=lambda r: int(Path(r['archive_path']).stem))
                assert len(rs) == 12
                images = []
                for r in rs:
                    assert sha(ROOT / r['path']) == r['sha256']
                    images.append(np.asarray(Image.open(ROOT / r['path']).convert('RGB')))
                r = rs[0]
                info = {'group': r['document_id'], 'cluster': r['document_type'], 'label': 'original' if r['label']=='or' else 'recapture', 'camera': r['camera'], 'display': r['condition_or_display']}
            elif domain == 'comma':
                p = ROOT / data['video_path'].replace('\\', '/')
                t = ROOT / data['local'].replace('\\', '/') / 'global_pose/frame_times'
                assert sha(p) == data['video_sha256'] and sha(t) == data['frame_times_sha256']
                images, sampling = helper.sequential_frames(p, len(np.load(t).ravel()))
                info = {'group': data['date_group'], 'cluster': data['date_group'], 'label': 'original', 'sampling': sampling}
            else:
                images = sampler.sample_video(data, 12)
                info = {'group': ident.replace('_O_', '_').replace('_R_', '_'), 'cluster': ident[-3:], 'label': 'original' if '_O_' in ident else 'recapture', 'file_sha256': sha(data)}
            assert len(images) == 12
            blocks, records = [], []
            for mode in (MODES if domain != 'public' else ['full_frame']):
                ff, base = features(m, [transform(a, mode) for a in images])
                records.append(info | {'domain': domain, 'split': split, 'video': ident, 'mode': mode, 'offset': len(blocks)*24, 'count': 24, 'baseline': base})
                blocks.append(ff)
            x = np.concatenate(blocks)
            assert np.isfinite(x).all()
            np.savez_compressed(cache, x=x)
            write(meta, records)
            if domain == 'dlc':
                thumb = Image.fromarray(images[6]); thumb.thumbnail((144, 220)); thumb.save(dest / f'{key}.jpg')
            print('EXTRACT', domain, split, ident, flush=True)
    finally:
        m.close()
    write(OUT / f'{domain}_{split}_extraction.json', {'clips': len(jobs), 'seconds_this_run': time.perf_counter()-start, 'feature_files': len(list(dest.glob('*.npz')))})


def dataset(domain, split):
    if domain == 'vd':
        z = np.load(OLD / f'{split}_features.npz')
        rs = read(OLD / f'{split}_records.json')
        records = [r | {'domain': domain, 'group': r['source_group'], 'cluster': r['source_group'], 'video': r['source_group']+'_'+r['label'], 'baseline': float(b)} for r, b in zip(rs, z['baseline'])]
        return z['x'].astype(np.float64), records
    arrays, records, offset = [], [], 0
    for p in sorted((OUT / 'features' / f'{domain}_{split}').glob('*.npz')):
        x = np.load(p)['x']
        records.extend(r | {'offset': r['offset']+offset} for r in read(p.with_suffix('.json')))
        arrays.append(x); offset += len(x)
    assert arrays
    return np.concatenate(arrays).astype(np.float64), records


def objective(theta, x, y, weights, center, lam):
    logits = x @ theta[:-1] + theta[-1]
    delta = theta-center
    loss = np.dot(weights, np.logaddexp(0, logits)-y*logits) + lam/2*np.dot(delta, delta)
    err = weights*(expit(logits)-y)
    grad = np.r_[x.T @ err, err.sum()] + lam*delta
    return float(loss), grad


def fit():
    pro = read(OUT / 'protocol.json')
    assert not (OUT / 'training.json').exists()
    arrays, ys, ws, counts = [], [], [], {}
    for domain in ['vd', 'dlc', 'comma']:
        x, rs = dataset(domain, 'train'); y = np.zeros(len(x)); w = np.zeros(len(x))
        for label in ['original', 'recapture']:
            subset = [r for r in rs if r['label'] == label]
            if not subset: continue
            groups = collections.Counter(r['group'] for r in subset)
            mass = pro['train_mass'][domain+'_'+label]
            for r in subset:
                sl = slice(r['offset'], r['offset']+r['count'])
                w[sl] = mass / len(groups) / groups[r['group']] / r['count']
                y[sl] = int(label=='recapture')
        assert (w > 0).all()
        arrays.append(x); ys.append(y); ws.append(w)
        counts[domain] = {'feature_rows': len(x), 'groups': len({r['group'] for r in rs}), 'recordings': len({r['video'] for r in rs})}
    x, y, w = np.concatenate(arrays), np.concatenate(ys), np.concatenate(ws)
    assert np.isclose(w.sum(), 1) and np.isclose(w[y==1].sum(), .5)
    theta0 = np.load(OUT / 'v7_theta.npy')
    rng = np.random.default_rng(20260919)
    xx, yy, ww = rng.normal(size=(9, 4)), rng.integers(0, 2, 9), np.full(9, 1/9)
    center, th = rng.normal(size=5), rng.normal(size=5)
    grad_error = check_grad(lambda a: objective(a,xx,yy,ww,center,.01)[0], lambda a: objective(a,xx,yy,ww,center,.01)[1], th)
    assert grad_error < 1e-6
    results = {}
    for name, lam, center in [('ordinary', .001, np.zeros_like(theta0))] + [(f'anchored_{v:g}', v, theta0) for v in pro['methods']['anchored']['lambdas']]:
        start = time.perf_counter()
        res = minimize(objective, center.copy(), args=(x,y,w,center,lam), jac=True, method='L-BFGS-B', options={k:v for k,v in pro['optimizer'].items() if k!='method'})
        assert res.success and np.isfinite(res.x).all(), res.message
        np.savez(OUT / f'{name}.npz', theta=res.x)
        results[name] = {'lambda': lam, 'iterations': res.nit, 'loss': res.fun, 'seconds': time.perf_counter()-start, 'gradient_max_abs': float(np.max(np.abs(res.jac))), 'distance_from_v7': float(np.linalg.norm(res.x-theta0)), 'sha256': sha(OUT / f'{name}.npz')}
        print('FIT', name, results[name], flush=True)
    write(OUT / 'training.json', {'counts': counts, 'total_weight': w.sum(), 'positive_weight': w[y==1].sum(), 'gradient_check_error': grad_error, 'parameters': 513, 'models': results})


def metric(rs, scores):
    y = np.asarray([r['label']=='recapture' for r in rs]); p = np.asarray(scores)>=.5
    tp, tn, fp, fn = (int(v.sum()) for v in (y&p, ~y&~p, ~y&p, y&~p))
    result = {'fp': fp, 'fn': fn, 'original_n': tn+fp, 'recapture_n': tp+fn, 'fpr': fp/(tn+fp) if tn+fp else None, 'fnr': fn/(tp+fn) if tp+fn else None}
    result['macro_f1'] = ((2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0)+(2*tn/(2*tn+fp+fn) if 2*tn+fp+fn else 0))/2 if y.any() and (~y).any() else None
    return result


def evaluate(domain, split, names):
    x, rs = dataset(domain, split)
    out, report = {}, {}
    for name in ['v7', *names]:
        theta = np.load(OUT / 'v7_theta.npy') if name=='v7' else np.load(OUT / f'{name}.npz')['theta']
        p = expit(x @ theta[:-1]+theta[-1])
        pp = [float(p[r['offset']:r['offset']+r['count']].mean()) for r in rs]
        if name=='v7': assert max(abs(a-r['baseline']) for a,r in zip(pp,rs)) < 1e-6
        out[name] = pp
        report[name] = {'overall': metric(rs, pp), 'modes': {mode: metric([r for r in rs if r['mode']==mode], [v for r,v in zip(rs,pp) if r['mode']==mode]) for mode in sorted({r['mode'] for r in rs})}}
    records = [r | {'scores': {name: ps[i] for name,ps in out.items()}} for i,r in enumerate(rs)]
    write(OUT / f'{domain}_{split}_predictions.json', records)
    write(OUT / f'{domain}_{split}_summary.json', report)
    return report


def select():
    assert not (OUT / 'selection.json').exists()
    train = read(OUT / 'training.json'); names = list(train['models'])
    reports = {d: evaluate(d, 'development', names) for d in ['vd', 'dlc', 'comma']}
    base = np.mean([reports[d]['v7']['overall']['macro_f1'] for d in ['vd', 'dlc']])
    choices = []
    for name in names:
        reasons = []
        for d in ['vd', 'dlc', 'comma']:
            b, c = reports[d]['v7']['overall'], reports[d][name]['overall']
            for k in (['fp'] if d=='comma' else ['fp','fn']):
                if c[k] > b[k]: reasons.append(f'{d}_{k}_increased_{b[k]}_to_{c[k]}')
        mean = float(np.mean([reports[d][name]['overall']['macro_f1'] for d in ['vd', 'dlc']]))
        choices.append({'name': name, 'mean_paired_domain_f1': mean, 'reasons': reasons, 'eligible': not reasons and mean>base and name.startswith('anchored'), 'lambda': train['models'][name]['lambda']})
    eligible = [r for r in choices if r['eligible']]
    chosen = max(eligible, key=lambda r:(r['mean_paired_domain_f1'],r['lambda'])) if eligible else None
    result = {'baseline_mean_f1': float(base), 'choices': choices, 'selected': chosen, 'development_gate_pass': bool(chosen), 'gate_pass': False}
    if chosen:
        name = chosen['name']; guards = {}
        for d,s in [('vd','new_val'),('vd','new_tcl'),('public','guard')]:
            rr = evaluate(d,s,[name]); b,c = rr['v7']['overall'],rr[name]['overall']
            guards[f'{d}_{s}'] = {'baseline':b,'candidate':c,'pass':c['fp']<=b['fp'] and c['fn']<=b['fn']}
        result['legacy_guards'] = guards
        result['gate_pass'] = all(r['pass'] for r in guards.values())
    write(OUT / 'selection.json', result)
    print(json.dumps(result,indent=2), flush=True)


if __name__ == '__main__':
    p=argparse.ArgumentParser(); p.add_argument('action', choices=['init','extract','fit','select','confirmation']); p.add_argument('--domain'); p.add_argument('--split'); a=p.parse_args()
    if a.action != 'init':
        for path,h in read(OUT/'protocol.json')['frozen_sha256'].items():
            assert sha(ROOT/path)==h, path
    if a.action=='init': init()
    elif a.action=='extract': extract(a.domain,a.split)
    elif a.action=='fit': fit()
    elif a.action=='select': select()
    else:
        sel=read(OUT/'selection.json'); assert sel['gate_pass']; evaluate('dlc','confirmation',[sel['selected']['name']])
