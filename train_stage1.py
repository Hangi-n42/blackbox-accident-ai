"""Train small recapture model with source-held-out diagnostics on public examples."""
from pathlib import Path
import argparse
import hashlib
import json
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, confusion_matrix
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from solution.stage1 import extract_features, sample_video


def fit_model(x, y):
    model = make_pipeline(StandardScaler(), LogisticRegression(C=.1, max_iter=1000, random_state=20260911))
    model.fit(x, y)
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='Baseline/data/stage1')
    parser.add_argument('--output', default='model/stage1')
    parser.add_argument('--reports', default='research/stage1')
    args = parser.parse_args()
    data, output, reports = Path(args.data), Path(args.output), Path(args.reports)
    output.mkdir(parents=True, exist_ok=True)
    reports.mkdir(parents=True, exist_ok=True)
    labels = pd.read_csv(data / 'labels.csv')
    rows = []
    manifest = []
    for row in labels.itertuples():
        features, names = extract_features(sample_video(data / row.path))
        rows.append(features)
        manifest.append({'path': row.path, 'label': row.label, 'source_group': Path(row.path).stem,
                         'sha256': hashlib.sha256((data / row.path).read_bytes()).hexdigest(),
                         'origin': 'DACON competition-provided baseline sample',
                         'recapture_kind': 'simulated' if row.label == 'RERECORDED' else 'original'})
    x = np.asarray(rows)
    y = (labels.label == 'RERECORDED').to_numpy().astype(int)
    # Only development grouping uses source names; deployed prediction never uses names.
    groups = [Path(p).stem for p in labels.path]
    n = 19
    families = {'quality': list(range(8)) + list(range(n, n + 8)),
                'frequency': list(range(8, n)) + list(range(n + 8, 2 * n)),
                'border': list(range(2 * n, x.shape[1])), 'all': list(range(x.shape[1]))}
    report = {'scope': 'Synthetic public-example diagnostic only; no actual recapture validation.',
              'fixed_C': .1, 'threshold': .5, 'source_groups': len(set(groups)), 'models': {}}
    predictions = labels[['ID', 'path', 'label']].copy()
    for family, indices in families.items():
        probabilities = np.zeros(len(y))
        for train, valid in LeaveOneGroupOut().split(x, y, groups):
            fitted = fit_model(x[train][:, indices], y[train])
            probabilities[valid] = fitted.predict_proba(x[valid][:, indices])[:, 1]
        pred = (probabilities >= .5).astype(int)
        report['models'][family] = {'macro_f1': f1_score(y, pred, average='macro', labels=[0, 1]),
                                     'confusion_matrix': confusion_matrix(y, pred, labels=[0, 1]).tolist()}
        predictions[family + '_oof_probability'] = probabilities
        fitted_family = fit_model(x[:, indices], y)
        fs, fc = fitted_family.steps[0][1], fitted_family.steps[1][1]
        family_artifact = {'version': 1, 'sample_frames': 12, 'feature_names': names, 'indices': indices,
                           'mean': fs.mean_.tolist(), 'scale': fs.scale_.tolist(),
                           'coef': fc.coef_[0].tolist(), 'intercept': float(fc.intercept_[0]),
                           'threshold': .5, 'training_scope': report['scope'], 'family': family}
        (output / f'forensic_{family}.json').write_text(json.dumps(family_artifact, indent=2), encoding='utf-8')
    # Family fixed in advance; the small diagnostic set is not an independent test.
    indices = families['all']
    fitted = fit_model(x[:, indices], y)
    scaler, clf = fitted.steps[0][1], fitted.steps[1][1]
    artifact = {'version': 1, 'sample_frames': 12, 'feature_names': names, 'indices': indices,
                'mean': scaler.mean_.tolist(), 'scale': scaler.scale_.tolist(),
                'coef': clf.coef_[0].tolist(), 'intercept': float(clf.intercept_[0]),
                'threshold': .5, 'training_scope': report['scope'],
                'model_selection': 'all feature family and C=0.1 fixed; threshold 0.5 fixed'}
    (output / 'forensic.json').write_text(json.dumps(artifact, indent=2), encoding='utf-8')
    (reports / 'group_oof_metrics.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    predictions.to_csv(reports / 'group_oof_predictions.csv', index=False)
    pd.DataFrame(x, columns=names).assign(ID=labels.ID).to_csv(reports / 'features.csv', index=False)
    (reports / 'training_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
