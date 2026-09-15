"""Source-held-out robustness diagnostics; class-preserving re-encoding variants."""
from pathlib import Path
import json
import cv2
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, confusion_matrix
from sklearn.model_selection import LeaveOneGroupOut
from solution.stage1 import extract_features, sample_video
from train_stage1 import fit_model


def jpeg(frames, quality):
    result = []
    for f in frames:
        ok, encoded = cv2.imencode('.jpg', cv2.cvtColor(f, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, quality])
        if not ok:
            raise RuntimeError('JPEG encoding failed')
        result.append(cv2.cvtColor(cv2.imdecode(encoded, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB))
    return result


def resize(frames):
    return [cv2.resize(f, (max(32, f.shape[1] // 2), max(32, f.shape[0] // 2)), interpolation=cv2.INTER_AREA) for f in frames]


def main():
    data = Path('Baseline/data/stage1')
    reports = Path('research/stage1')
    labels = pd.read_csv(data / 'labels.csv')
    y = (labels.label == 'RERECORDED').to_numpy().astype(int)
    groups = np.asarray([Path(p).stem for p in labels.path])
    variants = {'clean': [], 'jpeg75': [], 'jpeg95': [], 'half_resolution': []}
    for row in labels.itertuples():
        frames = sample_video(data / row.path)
        for name, altered in [('clean', frames), ('jpeg75', jpeg(frames, 75)),
                              ('jpeg95', jpeg(frames, 95)), ('half_resolution', resize(frames))]:
            variants[name].append(extract_features(altered)[0])
    variants = {name: np.asarray(values) for name, values in variants.items()}
    families = {'frequency': list(range(8, 19)) + list(range(27, 38)), 'all': list(range(50))}
    report = {'scope': 'Synthetic example source-held-out robustness diagnostic. Not actual recapture performance.', 'models': {}}
    probability_rows = []
    for family, indices in families.items():
        for train_augmentation in [False, True]:
            probs = {name: np.zeros(len(y)) for name in variants}
            for train, valid in LeaveOneGroupOut().split(variants['clean'], y, groups):
                training = list(variants.values()) if train_augmentation else [variants['clean']]
                x_train = np.concatenate([v[train][:, indices] for v in training])
                y_train = np.tile(y[train], len(training))
                model = fit_model(x_train, y_train)
                for name, features in variants.items():
                    probs[name][valid] = model.predict_proba(features[valid][:, indices])[:, 1]
            tag = family + ('_augmented' if train_augmentation else '_clean')
            report['models'][tag] = {}
            for name, probabilities in probs.items():
                pred = (probabilities >= .5).astype(int)
                report['models'][tag][name] = {'macro_f1': f1_score(y, pred, average='macro', labels=[0, 1]),
                                               'confusion_matrix': confusion_matrix(y, pred, labels=[0, 1]).tolist()}
                probability_rows += [{'model': tag, 'variant': name, 'ID': labels.iloc[i].ID,
                                      'label': int(y[i]), 'probability': float(p)} for i, p in enumerate(probabilities)]
            if train_augmentation:
                model = fit_model(np.concatenate([v[:, indices] for v in variants.values()]), np.tile(y, len(variants)))
                scaler, clf = model.steps[0][1], model.steps[1][1]
                artifact = {'version': 1, 'sample_frames': 12, 'indices': indices,
                            'mean': scaler.mean_.tolist(), 'scale': scaler.scale_.tolist(),
                            'coef': clf.coef_[0].tolist(), 'intercept': float(clf.intercept_[0]),
                            'threshold': .5, 'training_scope': report['scope'], 'family': tag,
                            'augmentation': list(variants), 'note': 'Both classes receive identical class-preserving variants.'}
                Path(f'model/stage1/forensic_{tag}.json').write_text(json.dumps(artifact, indent=2), encoding='utf-8')
    (reports / 'robustness_metrics.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    pd.DataFrame(probability_rows).to_csv(reports / 'robustness_predictions.csv', index=False)
    np.savez_compressed(reports / 'robustness_features.npz', **variants)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
