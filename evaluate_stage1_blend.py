"""Reproduce candidate blend diagnostics from held-out feature and frozen TPO scores."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score


def main():
    root = Path('research/stage1')
    features = pd.read_csv(root / 'robustness_predictions.csv')
    tpo = json.loads((root / 'tpo_judgments.json').read_text(encoding='utf-8'))
    lookup = {(s['ID'], variant): values['probability']
              for s in tpo['samples'] for variant, values in s['results'].items()}
    results = {}
    for model_name, rows in features.groupby('model'):
        results[model_name] = {}
        for variant, group in rows.groupby('variant'):
            if variant not in ('clean', 'jpeg75', 'half_resolution'):
                continue
            pretrained = np.asarray([lookup[(identity, variant)] for identity in group.ID])
            results[model_name][variant] = {
                str(weight): float(f1_score(group.label,
                                           (weight * group.probability + (1 - weight) * pretrained) >= .5,
                                           labels=[0, 1], average='macro'))
                for weight in [0, .25, .5, .75, 1]
            }
    report = {'scope': 'Candidate comparison on reused public diagnostic groups, not independent test performance.',
              'weight_definition': 'forensic weight; TPO weight = 1 - forensic weight',
              'threshold': .5, 'models': results}
    (root / 'fixed_blend_diagnostic.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
