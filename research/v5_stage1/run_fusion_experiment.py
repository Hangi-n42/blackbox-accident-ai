"""Bounded CPU score-fusion diagnostic; never exports a deployment model."""
import os
for _name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_name] = '2'
import argparse
import csv
import hashlib
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'research/v5_stage1'
INPUTS = [
    'research/stage1/dlc_subset/frozen_predictions.json',
    'research/stage1/dlc_subset/acquisition_manifest.json',
    'research/stage1/dlc_subset/frozen_diagnostic_manifest.json',
    'research/stage1/comma_original_diagnostic/predictions.json',
    'research/stage1/comma_original_diagnostic/sources.json',
    'research/stage1/comma_original_diagnostic/frozen_manifest.json',
    'research/stage1/training_manifest.json',
    'research/stage1/tpo_judgments.json',
    'research/stage1/robustness_features.npz',
    'research/stage1/features.csv',
    'Baseline/data/stage1/labels.csv',
    'research/stage3_external_overlap.json',
    'model/stage1/config.json',
    'model/stage1/forensic_frequency_augmented.json',
    'evaluate_stage1.py',
    'research/v5_stage1/run_fusion_experiment.py',
]

def read(path):
    return json.loads((ROOT / path).read_text(encoding='utf-8'))

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def save(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')

def freeze():
    target = OUT / 'experiment_frozen_plan.json'
    if target.exists():
        raise RuntimeError('Plan already exists; do not overwrite a frozen experiment')
    d = read(INPUTS[0])
    docs = sorted({r['document_id'] for r in d})
    plan = {
        'frozen_at_utc': datetime.now(timezone.utc).isoformat(),
        'experiment': 'single two-logit L2 fusion; no candidate search',
        'features': ['logit(pT)', 'logit(pF)'], 'clip_epsilon': 1e-6,
        'standardization': 'fit on each calibration training fold only',
        'C': 0.1, 'penalty': 'l2', 'solver': 'lbfgs', 'fit_intercept': True,
        'seed': 20260913, 'max_iter': 1000, 'threshold': 0.5,
        'class_weight': None, 'sample_weight': None, 'cpu_thread_limit': 2,
        'documents': docs, 'cameras': ['android', 'iphone'],
        'main_split': '6 leave-document-out folds, all four rows of held document excluded',
        'stress_split': '12 held-document plus held-camera folds; train other camera and other documents',
        'comma': 'all 23 original segments excluded from fitting; report every main fold separately',
        'public': 'clean/jpeg75/half_resolution; final forensic fitted on these source labels: training-fit diagnostic only',
        'primary_go': 'main DLC OOF FN < 6, FP <= 0; all six comma guards FP == 0; all six folds and all public variants do not reduce same-row baseline macro-F1',
        'camera_gate': 'each camera stress pooled held-document predictions: FN <= matching baseline FN and FP <= matching baseline FP',
        'source_stability_gate': 'main per-document FN and FP may not exceed corresponding baseline counts',
        'decision': 'any failed gate STOP; all passed only GO to new real-road-recapture validation, never automatic deployment',
        'forbidden': ['threshold/C/feature search', 'best-fold selection', 'GPU', 'base model updates', 'deployment export', 'new data download'],
        'input_sha256': {p: sha(ROOT / p) for p in INPUTS},
    }
    save(target, plan)
    print(json.dumps({'plan_sha256': sha(target), 'frozen_at_utc': plan['frozen_at_utc']}))

def run(expected_hash):
    import numpy as np
    import sklearn
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import confusion_matrix, f1_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from threadpoolctl import threadpool_limits, threadpool_info
    plan_path = OUT / 'experiment_frozen_plan.json'
    assert sha(plan_path) == expected_hash
    plan = json.loads(plan_path.read_text(encoding='utf-8'))
    assert all(sha(ROOT / p) == h for p, h in plan['input_sha256'].items())
    d, c = read(INPUTS[0]), read(INPUTS[3])
    acquire, csources, public_sources = read(INPUTS[1]), read(INPUTS[4]), read(INPUTS[6])
    fm, cm = read(INPUTS[2]), read(INPUTS[5])
    assert fm['artifact_sha256'] == cm['artifact_sha256']
    artifact_hashes = {p: sha(ROOT / p) for p in fm['artifact_sha256']}
    assert artifact_hashes == fm['artifact_sha256']
    assert len(d) == 24 and len(c) == 23
    assert len({r['video_id'] for r in d}) == 24
    assert len({r['segment'] for r in c}) == 23
    excluded = {r['segment'] for r in read('research/stage3_external_overlap.json')['excluded']}
    assert not excluded.intersection(r['segment'] for r in c)
    assert set(r['segment'] for r in c) == set(r['segment'] for r in csources['videos'])
    content_hash_groups = defaultdict(set)
    for r in acquire:
        content_hash_groups[r['sha256']].add(r['document_id'])
    cross_document_duplicates = {h: sorted(v) for h, v in content_hash_groups.items() if len(v) > 1}
    assert not cross_document_duplicates
    hashes_public = {r['sha256'] for r in public_sources}
    hashes_comma = {r['video_sha256'] for r in csources['videos']}
    assert not hashes_public & hashes_comma
    assert not set(content_hash_groups) & (hashes_public | hashes_comma)
    origins = {
        'pT': 'frozen merged TPO; original/print/replay object data used in TPO training; no DLC or comma adaptation in this experiment',
        'pF': 'same frozen forensic_frequency_augmented fitted on five DACON source groups and their clean/jpeg75/jpeg95/half variants; no DLC or comma fitting',
        'calibration': 'DLC training rows only, explicit video/document/camera IDs recorded per fold',
        'public_status': 'forensic already fitted on public GT; all public results are training-fit/regression diagnostics, not holdout',
        'limits': 'CLIP pretraining source overlap unknown; prior development already inspected all three diagnostic datasets; metadata/hash nonoverlap does not prove visual independence',
    }
    def x(rows):
        a = np.asarray([[r['tpo'], r['forensic']] for r in rows], dtype=float)
        a = np.clip(a, plan['clip_epsilon'], 1 - plan['clip_epsilon'])
        return np.log(a / (1-a))
    def metrics(rows, probs):
        y = np.asarray([r['y_true'] for r in rows], dtype=int)
        pred = np.asarray(probs) >= plan['threshold']
        matrix = confusion_matrix(y, pred, labels=[0, 1]).tolist()
        return {'n': len(rows), 'macro_f1': float(f1_score(y, pred, labels=[0, 1], average='macro')),
                'confusion_matrix_true_rows_pred_columns_0original_1recapture': matrix,
                'false_positives': matrix[0][1], 'false_negatives': matrix[1][0]}
    def baseline(rows):
        values = [(r['tpo'] + r['forensic']) / 2 for r in rows]
        assert all(abs(v-r['blend']) < 1e-12 for v, r in zip(values, rows))
        return metrics(rows, values)
    def comparison(rows, probs):
        return {'baseline': baseline(rows), 'candidate': metrics(rows, probs)}
    with (ROOT / 'Baseline/data/stage1/labels.csv').open(encoding='utf-8-sig') as f:
        labels = list(csv.DictReader(f))
    with (ROOT / 'research/stage1/features.csv').open(encoding='utf-8-sig') as f:
        feature_rows = list(csv.DictReader(f))
    assert [r['ID'] for r in labels] == [r['ID'] for r in feature_rows]
    cache = np.load(ROOT / 'research/stage1/robustness_features.npz')
    assert np.allclose(cache['clean'], [[float(v) for k, v in r.items() if k != 'ID'] for r in feature_rows])
    a = read('model/stage1/forensic_frequency_augmented.json')
    tpo = {r['ID']: r for r in read('research/stage1/tpo_judgments.json')['samples']}
    publics = {}
    for variant in ('clean', 'jpeg75', 'half_resolution'):
        xx = (cache[variant][:, a['indices']] - a['mean']) / a['scale']
        pp = 1 / (1 + np.exp(-np.clip(xx @ np.asarray(a['coef']) + a['intercept'], -40, 40)))
        publics[variant] = [{'ID': r['ID'], 'y_true': int(r['label'] == 'RERECORDED'),
                             'tpo': tpo[r['ID']]['results'][variant]['probability'], 'forensic': float(pp[i]),
                             'blend': (tpo[r['ID']]['results'][variant]['probability'] + float(pp[i])) / 2}
                            for i, r in enumerate(labels)]
    def fit_fold(doc, camera=None):
        train = [r for r in d if r['document_id'] != doc and (camera is None or r['camera'] != camera)]
        valid = [r for r in d if r['document_id'] == doc and (camera is None or r['camera'] == camera)]
        assert not set(r['document_id'] for r in train) & set(r['document_id'] for r in valid)
        assert len(train) == (20 if camera is None else 10)
        assert len(valid) == (4 if camera is None else 2)
        if camera is not None:
            assert not set(r['camera'] for r in train) & set(r['camera'] for r in valid)
        model = make_pipeline(StandardScaler(), LogisticRegression(C=plan['C'], penalty=plan['penalty'],
                             solver=plan['solver'], max_iter=plan['max_iter'], random_state=plan['seed']))
        model.fit(x(train), [r['y_true'] for r in train])
        prob = model.predict_proba(x(valid))[:, 1]
        scale, clf = model.steps[0][1], model.steps[1][1]
        rec = {'held_document': doc, 'held_camera': camera, 'score_source_provenance': origins,
               'train_rows': [{k: r[k] for k in ('video_id', 'document_id', 'camera', 'y_true')} for r in train],
               'valid_rows': [{**r, 'candidate': float(p)} for r, p in zip(valid, prob)],
               'fit_parameters_for_audit_not_deployment': {'mean': scale.mean_.tolist(), 'scale': scale.scale_.tolist(),
                    'coef': clf.coef_[0].tolist(), 'intercept': float(clf.intercept_[0]), 'iterations': clf.n_iter_.tolist()},
               'comparison': comparison(valid, prob)}
        if camera is None:
            cp = model.predict_proba(x(c))[:, 1]
            rec['comma_original_guard'] = {'n': 23, 'baseline_false_positives': sum(r['blend'] >= .5 for r in c),
                 'candidate_false_positives': int(sum(cp >= .5)), 'candidate_false_positive_rate': float(np.mean(cp >= .5)),
                 'rows': [{k: r[k] for k in ('segment', 'vehicle', 'tpo', 'forensic', 'blend')} | {'candidate': float(p)} for r, p in zip(c, cp)]}
            rec['public_training_fit_only'] = {v: comparison(rows, model.predict_proba(x(rows))[:, 1]) for v, rows in publics.items()}
        return rec
    with threadpool_limits(limits=2):
        main = [fit_fold(doc) for doc in plan['documents']]
        stress = [fit_fold(doc, camera) for doc in plan['documents'] for camera in plan['cameras']]
        pools = threadpool_info()
    main_rows = [r for fold in main for r in fold['valid_rows']]
    stress_rows = [r for fold in stress for r in fold['valid_rows']]
    assert sorted(r['video_id'] for r in main_rows) == sorted(r['video_id'] for r in d)
    assert sorted(r['video_id'] for r in stress_rows) == sorted(r['video_id'] for r in d)
    overall = comparison(main_rows, [r['candidate'] for r in main_rows])
    device = {cam: comparison([r for r in stress_rows if r['camera'] == cam],
                             [r['candidate'] for r in stress_rows if r['camera'] == cam]) for cam in plan['cameras']}
    no_worse = lambda v: all(v['candidate'][k] <= v['baseline'][k] for k in ('false_positives', 'false_negatives'))
    gates = {'dlc_main_improves_misses_without_new_false_positives': overall['candidate']['false_negatives'] < 6 and overall['candidate']['false_positives'] == 0,
             'all_six_comma_guards_preserve_zero_fp': all(f['comma_original_guard']['candidate_false_positives'] == 0 for f in main),
             'public_training_fit_regression': all(v['candidate']['macro_f1'] >= v['baseline']['macro_f1'] for f in main for v in f['public_training_fit_only'].values()),
             'both_camera_stresses_no_worse': all(no_worse(v) for v in device.values()),
             'all_documents_no_worse': all(no_worse(f['comparison']) for f in main)}
    report = {'finished_at_utc': datetime.now(timezone.utc).isoformat(), 'plan_sha256': expected_hash,
        'plan': plan, 'scope': 'DLC cross-fitted calibration diagnostic; no actual road recapture positives; no generalization claim',
        'runtime': {'sklearn': sklearn.__version__, 'numpy': np.__version__, 'threadpools_during_fit': pools, 'GPU_used': False},
        'source_overlap_audit': {'dlc_unique_video_ids': 24, 'dlc_unique_documents': 6,
            'cross_document_exact_JPEG_duplicates': cross_document_duplicates,
            'comma_unique_segments': 23, 'comma_unique_routes': len({r['route'] for r in c}),
            'known_public_duplicate_excluded': sorted(excluded), 'recorded_cross_dataset_file_hash_overlap': [],
            'visual_or_CLIP_pretraining_overlap': 'not fully verifiable; hash comparison across encodings cannot exclude source overlap',
            'base_score_origins': origins, 'base_artifact_sha256': artifact_hashes},
        'main_oof': overall, 'camera_stress_oof': comparison(stress_rows, [r['candidate'] for r in stress_rows]),
        'camera_stress_by_camera': device, 'main_folds': main, 'camera_stress_folds': stress,
        'public_rows_training_fit_only': publics, 'gates': gates,
        'decision': 'GO_TO_NEW_TARGET_VALIDATION_ONLY' if all(gates.values()) else 'STOP',
        'deployment_model_exported': False, 'final_all_DLC_refit': False,
        'no_cross_domain_pooled_metric': True}
    assert all(sha(ROOT / p) == h for p, h in plan['input_sha256'].items())
    assert all(sha(ROOT / p) == h for p, h in artifact_hashes.items())
    report['input_and_base_artifact_hashes_unchanged_after'] = True
    save(OUT / 'experiment_report.json', report)
    print(json.dumps({'main_oof': overall, 'camera_stress_by_camera': device,
                      'comma_fp_by_main_fold': [f['comma_original_guard']['candidate_false_positives'] for f in main],
                      'gates': gates, 'decision': report['decision']}, ensure_ascii=True))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--freeze', action='store_true')
    parser.add_argument('--run-plan-sha')
    args = parser.parse_args()
    if args.freeze:
        freeze()
    else:
        run(args.run_plan_sha)
