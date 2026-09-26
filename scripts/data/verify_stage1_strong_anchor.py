"""One frozen strong-anchor candidate; never fits, tunes, or overwrites prior runs."""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.special import expit
from sklearn.metrics import confusion_matrix, f1_score

import experiment_stage1_anchored_head as prior

ROOT = prior.ROOT
OLD = prior.OUT
OUT = ROOT / 'artifacts/stage1_strong_anchor_20260919'
NAME = 'anchored_0.01'
PAIRS = [('vd', 'new_val'), ('vd', 'new_tcl'), ('public', 'guard')]


def init():
    OUT.mkdir(exist_ok=False)
    candidate = OLD / f'{NAME}.npz'
    assert prior.sha(candidate) == prior.read(OLD / 'training.json')['models'][NAME]['sha256']
    paths = [Path(__file__), ROOT / 'scripts/data/experiment_stage1_anchored_head.py', candidate, OLD / 'v7_theta.npy']
    paths += [p for p in OLD.rglob('*') if p.is_file() and p.suffix in ('.json', '.npz', '.npy')]
    paths += [p for p in prior.REL.rglob('*') if p.is_file() and '__pycache__' not in p.parts]
    paths += [ROOT / 'artifacts/submissions/submit_v7.zip']
    for _, split in PAIRS[:2]:
        paths += [prior.OLD / f'{split}_features.npz', prior.OLD / f'{split}_records.json']
    pro = {
        'candidate': NAME, 'lambda': .01, 'threshold': .5,
        'hypothesis': 'Does the already trained, more strongly anchored head avoid the weak candidate original regression while retaining its observed development improvements?',
        'scope': 'New post-failure diagnostic, not a retrospective change to the rejected prior experiment. Repeated public/VD exposure is selection bias; these are known regression fixtures.',
        'input': 'V7 frozen normalized features of12 full frames and12 horizontal flips; mean per-view probabilities; no forensic',
        'candidate_fixed_before_new_guard_scores': True,
        'no_training_or_parameter_search': True,
        'guards': PAIRS,
        'gate': 'FP and FN must not increase separately within each guard dataset; additionally zero newly false-positive public originals that V7 classified correctly.',
        'confirmation': 'Only if all guards pass: frozen reserved DLC6 document IDs/3 types from prior dlc_plan; V7 and this candidate only. Require FP/FN nonincrease and strict Macro-F1 improvement. Report grouped uncertainty, not independent road generalization.',
        'runtime': 'Only after confirmation pass: real submission entry, offline loading, decision/output/missing-file agreement and rough time.',
        'failure': 'Stop without another lambda, threshold, blend, LoRA or confirmation scoring. Preserve V7. No submission.',
        'prior_decision': prior.read(OLD / 'candidate_decision.json'),
        'frozen_sha256': {str(p.relative_to(ROOT)): prior.sha(p) for p in sorted(set(paths))},
    }
    assert not (OLD / 'dlc_confirmation_acquired.json').exists()
    prior.write(OUT / 'protocol.json', pro)
    print('FROZEN', NAME, prior.sha(candidate), flush=True)


def check_hashes():
    pro = prior.read(OUT / 'protocol.json')
    for p, h in pro['frozen_sha256'].items():
        assert prior.sha(ROOT / p) == h, p


def guards():
    assert not (OUT / 'decision.json').exists()
    check_hashes()
    candidate = np.load(OLD / f'{NAME}.npz')['theta']
    reference = np.load(OLD / 'v7_theta.npy')
    reports, failures, maxdiff = {}, [], 0.
    for domain, split in PAIRS:
        x, rs = prior.dataset(domain, split)
        out = {}
        for name, theta in [('v7', reference), (NAME, candidate)]:
            frame_scores = expit(x @ theta[:-1] + theta[-1])
            out[name] = [float(frame_scores[r['offset']:r['offset']+r['count']].mean()) for r in rs]
        diff = max(abs(s-r['baseline']) for s, r in zip(out['v7'], rs))
        maxdiff = max(maxdiff, diff); assert diff < 1e-6
        records = [r | {'scores': {k:v[i] for k,v in out.items()}} for i,r in enumerate(rs)]
        report = {k:prior.metric(rs, values) for k,values in out.items()}
        y = np.array([r['label']=='recapture' for r in rs])
        for k, values in out.items():
            pred = np.array(values)>=.5
            tn, fp, fn, tp = confusion_matrix(y, pred, labels=[False,True]).ravel()
            assert (fp,fn)==(report[k]['fp'],report[k]['fn'])
            assert abs(f1_score(y,pred,average='macro')-report[k]['macro_f1'])<1e-12
        b, c = report['v7'], report[NAME]
        changes = [r for r in records if (r['scores']['v7']>=.5)!=(r['scores'][NAME]>=.5)]
        new_fp = [r['video'] for r in changes if r['label']=='original' and r['scores'][NAME]>=.5]
        reasons = [f'{domain}_{split}_{k}_increased_{b[k]}_to_{c[k]}' for k in ['fp','fn'] if c[k]>b[k]]
        if domain=='public' and new_fp:
            reasons.append('new_public_original_fp:'+','.join(new_fp))
        failures.extend(reasons)
        reports[f'{domain}_{split}'] = {'metrics': report, 'new_fp_videos': new_fp, 'pass': not reasons}
        prior.write(OUT / f'{domain}_{split}_predictions.json', records)
        prior.write(OUT / f'{domain}_{split}_changes.json', changes)
        print(domain,split,json.dumps(report),flush=True)
    check_hashes()
    decision = {'candidate':NAME,'gate_pass':not failures,'decision':'PROCEED_TO_RESERVED_DLC' if not failures else 'DO_NOT_ADOPT_GUARD_REGRESSION',
                'failures':failures,'guards':reports,'baseline_probability_max_difference':maxdiff,
                'independent_sklearn_metric_check':True,'prior_artifacts_and_v7_unchanged':True,
                'confirmation_scored':False,'new_submission':False}
    prior.write(OUT / 'decision.json',decision)
    print(json.dumps(decision,indent=2),flush=True)


if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['init','guards']);a=p.parse_args()
    (init if a.action=='init' else guards)()
