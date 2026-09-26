"""Frozen V6 branches on the 50 pre-reviewed UHDM pairs; no training."""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
RELEASE = ROOT / 'artifacts/submissions/verify_v6'
sys.path.insert(0, str(RELEASE / 'model/stage2/code'))
from solution.stage1 import extract_features, feature_probability
from solution.stage1_tpo_merged import TPODetector


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')


def stats(values):
    a = np.asarray(values, dtype=float)
    return {'mean': float(a.mean()), 'median': float(np.median(a)),
            'q25': float(np.quantile(a, .25)), 'q75': float(np.quantile(a, .75)),
            'positive': int((a > 0).sum()), 'negative': int((a < 0).sum())}


def main(out):
    out.mkdir(parents=True, exist_ok=False)
    qa_path = ROOT / 'artifacts/data_curation_20260917/uhdm/patch_qa_review.json'
    manifest_path = ROOT / 'artifacts/data_curation_20260917/uhdm/manifest.jsonl'
    qa = json.loads(qa_path.read_text())
    selected = [r for r in qa['records'] if r['status'] == 'pass_auxiliary_pair_content_QA']
    manifest = {r['id']: r for r in map(json.loads, manifest_path.read_text().splitlines())}
    assert len(selected) == len({r['id'] for r in selected}) == 50
    model = RELEASE / 'model/stage1'
    config = json.loads((model / 'config.json').read_text())
    artifact_path = model / config['forensic_artifact']
    artifact = json.loads(artifact_path.read_text())
    modes = ['full_frame', 'native384', 'native384_gray', 'native384_blur1']
    branches = ['forensic', 'tpo', 'ensemble']
    files = [Path(__file__), qa_path, manifest_path, model / 'config.json', artifact_path,
             model / 'tpo/merged_visual.pt', model / 'tpo/clip_source/clip/model.py']
    files += [RELEASE / 'model/stage2/code/solution' / n
              for n in ['stage1.py', 'stage1_v4.py', 'stage1_tpo_merged.py']]
    frozen = {str(p.relative_to(ROOT)): sha(p) for p in files}
    old = json.loads((ROOT / 'artifacts/stage1_sensitivity_20260917/provenance.json').read_text())['files']
    for path, digest in frozen.items():
        if path in old:
            assert digest == old[path]['current_sha256'], path
    protocol = {
        'scope': 'Exploratory paired image diagnostic; not video evaluation or contest ORIGINAL GT',
        'selection': 'All 50 previously passed native384 pairs; no result-dependent exclusions',
        'pair_ids': [r['id'] for r in selected], 'modes': modes,
        'transforms': {'full_frame': 'Original RGB JPEG decode',
                       'native384': 'Previously reviewed identical-coordinate lossless RGB crop',
                       'native384_gray': 'cv2 RGB2GRAY then GRAY2RGB, same size',
                       'native384_blur1': 'GaussianBlur sigma=1.0, kernel=(0,0), BORDER_REFLECT_101'},
        'controls': 'Both labels receive identical transforms. V6 CPU eval, weights/config fixed. No fitting.',
        'threshold': config['threshold'], 'forensic_weight': config['forensic_weight'],
        'limitations': ['Provider clean-reference is not verified camera ORIGINAL',
                       'Full-to-crop changes field of view, scale and forensic patch locations',
                       'Gray and blur are nonspecific sensitivity interventions, not causal isolation of moire',
                       'One image per score; no 12-frame video aggregation or temporal evidence',
                       '50 selected photo-like pairs are development data, source independence unresolved'],
        'analysis': 'Per-branch paired recapture-reference score gap; threshold rates; exact forensic logit decomposition; paired intervention effects',
        'files_sha256': frozen,
        'versions': {'torch': torch.__version__, 'numpy': np.__version__, 'opencv': cv2.__version__},
    }
    write(out / 'protocol.json', protocol)
    torch.set_num_threads(2)
    cv2.setNumThreads(2)
    idx = np.asarray(artifact['indices'])
    coef = np.asarray(artifact['coef'])
    mean, scale = np.asarray(artifact['mean']), np.asarray(artifact['scale'])
    detector = TPODetector(model / 'tpo', device='cpu')
    rows, inputs = [], []
    started = time.monotonic()
    max_crop_diff = 0
    max_logit_error = 0.
    try:
        for number, q in enumerate(selected, 1):
            record = manifest[q['id']]
            assert not record['exclude']
            arrays, metadata = [], []
            for label in ['gt', 'moire']:
                source = ROOT / record[label]['path']
                patch = Path(q[label + '_png'])
                assert sha(source) == record[label]['sha256'], source
                assert sha(patch) == q[label + '_png_sha256'], patch
                with Image.open(source) as im:
                    full = np.array(im.convert('RGB'))
                with Image.open(patch) as im:
                    native = np.array(im.convert('RGB'))
                x0, y0, x1, y1 = q['rect_xyxy']
                diff = int(np.max(np.abs(full[y0:y1, x0:x1].astype(int) - native.astype(int))))
                max_crop_diff = max(max_crop_diff, diff)
                assert native.shape == (384, 384, 3) and diff == 0
                gray = cv2.cvtColor(cv2.cvtColor(native, cv2.COLOR_RGB2GRAY), cv2.COLOR_GRAY2RGB)
                blur = cv2.GaussianBlur(native, (0, 0), 1.0, borderType=cv2.BORDER_REFLECT_101)
                inputs.append({'id': q['id'], 'label': label, 'source_group': record['source_group_id'],
                               'path': str(source.relative_to(ROOT)), 'source_sha256': record[label]['sha256'],
                               'patch_path': str(patch.relative_to(ROOT)), 'patch_sha256': q[label + '_png_sha256'],
                               'shape': list(full.shape), 'rect_xyxy': q['rect_xyxy']})
                for mode, a in zip(modes, [full, native, gray, blur]):
                    features, names = extract_features([a])
                    terms = ((features[idx] - mean) / scale) * coef
                    logit = float(terms.sum() + artifact['intercept'])
                    f = feature_probability(features, artifact)
                    error = abs(f - 1 / (1 + np.exp(-np.clip(logit, -40, 40))))
                    max_logit_error = max(max_logit_error, error)
                    metadata.append({'id': q['id'], 'source_group': record['source_group_id'],
                                     'label': 'screen_recapture' if label == 'moire' else 'provided_clean_reference',
                                     'mode': mode, 'forensic': f, 'forensic_logit': logit,
                                     'forensic_terms': terms.tolist(), 'features': features[idx].tolist()})
                    arrays.append(a)
            tpo = detector.score(arrays)
            for row, score in zip(metadata, tpo):
                row['tpo'] = float(score)
                row['ensemble'] = config['forensic_weight'] * row['forensic'] + (1-config['forensic_weight']) * row['tpo']
                assert all(np.isfinite(row[b]) and 0 <= row[b] <= 1 for b in branches)
                rows.append(row)
            if number % 10 == 0:
                print(f'Completed {number}/50 pairs ({len(rows)} scores)', flush=True)
    finally:
        detector.close()
    assert len(rows) == 400 and len(inputs) == 100
    assert max_logit_error < 1e-12
    lookup = {(r['id'], r['label'], r['mode']): r for r in rows}
    summary = {'pairs': 50, 'unique_source_groups': len({r['source_group'] for r in rows}),
               'input_images': 100, 'scoring_rows': len(rows), 'seconds': time.monotonic()-started,
               'max_patch_pixel_difference': max_crop_diff, 'max_logit_reconstruction_error': max_logit_error,
               'by_mode': {}, 'interventions_vs_native384': {}}
    contributions = {}
    for mode in modes:
        pairs = [(lookup[q['id'], 'provided_clean_reference', mode], lookup[q['id'], 'screen_recapture', mode]) for q in selected]
        result = {}
        for b in branches:
            result[b] = {'reference_mean': float(np.mean([g[b] for g, m in pairs])),
                         'recapture_mean': float(np.mean([m[b] for g, m in pairs])),
                         'reference_above_threshold': sum(g[b] >= config['threshold'] for g, m in pairs),
                         'recapture_above_threshold': sum(m[b] >= config['threshold'] for g, m in pairs),
                         'recapture_minus_reference': stats([m[b]-g[b] for g, m in pairs])}
        result['recapture_tpo_detected_ensemble_missed'] = sum(m['tpo'] >= .5 and m['ensemble'] < .5 for g, m in pairs)
        result['reference_tpo_above_ensemble_below'] = sum(g['tpo'] >= .5 and g['ensemble'] < .5 for g, m in pairs)
        summary['by_mode'][mode] = result
        deltas = np.array([np.array(m['forensic_terms']) - np.array(g['forensic_terms']) for g, m in pairs])
        contributions[mode] = [{'feature': names[i], 'feature_index': int(i),
                                'mean_paired_logit_contribution': float(deltas[:, j].mean()),
                                'median_paired_logit_contribution': float(np.median(deltas[:, j]))}
                               for j, i in enumerate(idx)]
    for mode in modes[2:]:
        summary['interventions_vs_native384'][mode] = {}
        for label in ['provided_clean_reference', 'screen_recapture']:
            summary['interventions_vs_native384'][mode][label] = {
                b: stats([lookup[q['id'], label, mode][b] - lookup[q['id'], label, 'native384'][b] for q in selected])
                for b in branches}
    unchanged = all(sha(ROOT / p) == digest for p, digest in frozen.items())
    assert unchanged
    summary['frozen_files_unchanged_after_run'] = unchanged
    write(out / 'inputs.json', inputs)
    write(out / 'predictions.json', rows)
    write(out / 'forensic_contributions.json', contributions)
    write(out / 'summary.json', summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    main(parser.parse_args().out)
