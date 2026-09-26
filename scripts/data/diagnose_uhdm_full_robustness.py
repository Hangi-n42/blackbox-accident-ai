"""Frozen full-image UHDM sensitivity: independent resize, JPEG, and blur."""
import argparse
import csv
import json
import time
from pathlib import Path

from diagnose_uhdm_branches import (
    ROOT, RELEASE, Image, TPODetector, cv2, np, torch,
    extract_features, feature_probability, sha, stats, write,
)

MODES = ['full_frame', 'resize_050', 'resize_025', 'jpeg_75', 'jpeg_50', 'blur_1', 'blur_2']
BRANCHES = ['forensic', 'tpo', 'ensemble']


def transform(rgb, mode):
    if mode == 'full_frame':
        return rgb
    if mode.startswith('resize_'):
        factor = .5 if mode == 'resize_050' else .25
        h, w = rgb.shape[:2]
        return cv2.resize(rgb, (round(w*factor), round(h*factor)), interpolation=cv2.INTER_AREA)
    if mode.startswith('jpeg_'):
        quality = int(mode.split('_')[1])
        ok, encoded = cv2.imencode('.jpg', cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
                                   [cv2.IMWRITE_JPEG_QUALITY, quality])
        assert ok
        decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        assert decoded is not None and decoded.shape == rgb.shape
        return cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB)
    return cv2.GaussianBlur(rgb, (0, 0), float(mode.split('_')[1]), borderType=cv2.BORDER_REFLECT_101)


def main(out):
    out.mkdir(parents=True, exist_ok=False)
    prior = ROOT / 'artifacts/uhdm_branch_diagnostic_20260918'
    inputs = json.loads((prior / 'inputs.json').read_text())
    baseline = {(r['id'], r['label']): r for r in json.loads((prior / 'predictions.json').read_text()) if r['mode'] == 'full_frame'}
    old_protocol = json.loads((prior / 'protocol.json').read_text())
    assert len(inputs) == len(baseline) == 100
    model = RELEASE / 'model/stage1'
    config = json.loads((model / 'config.json').read_text())
    artifact = json.loads((model / config['forensic_artifact']).read_text())
    frozen = dict(old_protocol['files_sha256'])
    frozen[str(Path(__file__).relative_to(ROOT))] = sha(Path(__file__))
    for path, digest in frozen.items():
        assert sha(ROOT/path) == digest, path
    protocol = {
        'scope': 'Same 50 selected UHDM development pairs; full-image single-frame diagnostic, no fitting',
        'pair_ids': old_protocol['pair_ids'], 'modes': MODES,
        'transforms': {'resize': 'INTER_AREA; width and height each 0.5 or 0.25; no upsampling back',
                       'jpeg': 'OpenCV RGB-BGR JPEG encode/decode, quality75 or50, all other codec defaults',
                       'blur': 'Full-resolution RGB GaussianBlur sigma1 or2 pixels, kernel0, BORDER_REFLECT_101'},
        'independence': 'Each condition starts from unchanged full RGB decode; same transform for both labels; no combinations',
        'severity': 'Diagnostic levels fixed before inference, not claimed to represent hidden evaluation distribution',
        'config': config, 'device': 'cpu', 'threads': 2,
        'versions': {'torch': torch.__version__, 'opencv': cv2.__version__, 'numpy': np.__version__},
        'files_sha256': frozen,
        'limitations': old_protocol['limitations'][:1] + [
            'Reference threshold crossings are not verified camera ORIGINAL false positives',
            'Image-only single-frame scores, not video aggregation or official F1',
            'Resize changes forensic patch footprint and frequencies; blur and JPEG change multiple cues',
            'Selected source groups are provisional; all data already development exposed'],
        'analysis': 'Failure counts, per-image new/recovered decisions versus original, branch disagreement, paired gaps',
    }
    write(out/'protocol.json', protocol)
    torch.set_num_threads(2)
    cv2.setNumThreads(2)
    idx = np.asarray(artifact['indices'])
    coef, mean, scale = map(np.asarray, [artifact['coef'], artifact['mean'], artifact['scale']])
    detector = TPODetector(model/'tpo', device='cpu')
    rows = []
    start = time.monotonic()
    try:
        for number, item in enumerate(inputs, 1):
            source = ROOT/item['path']
            assert sha(source) == item['source_sha256']
            with Image.open(source) as im:
                rgb = np.array(im.convert('RGB'))
            before = hashlib_array(rgb)
            arrays = [transform(rgb, mode) for mode in MODES]
            tpo = detector.score(arrays)
            for mode, a, score in zip(MODES, arrays, tpo):
                feat, names = extract_features([a])
                f = feature_probability(feat, artifact)
                terms = ((feat[idx]-mean)/scale)*coef
                logit = float(terms.sum()+artifact['intercept'])
                assert abs(f-1/(1+np.exp(-np.clip(logit,-40,40)))) < 1e-12
                row = {'id': item['id'], 'source_group': item['source_group'],
                       'label': 'screen_recapture' if item['label']=='moire' else 'provided_clean_reference',
                       'mode': mode, 'shape': list(a.shape), 'forensic': f, 'tpo': float(score),
                       'ensemble': config['forensic_weight']*f+(1-config['forensic_weight'])*float(score),
                       'forensic_logit': logit, 'forensic_terms': terms.tolist()}
                assert all(np.isfinite(row[b]) and 0<=row[b]<=1 for b in BRANCHES)
                rows.append(row)
            assert hashlib_array(rgb) == before
            if number % 10 == 0:
                print(f'Completed {number}/100 images; {len(rows)} scores', flush=True)
    finally:
        detector.close()
    assert len(rows) == 700
    write(out/'predictions.json', rows)
    with (out/'predictions.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['id','source_group','label','mode',*BRANCHES,'forensic_logit'])
        writer.writeheader()
        writer.writerows({k:r[k] for k in writer.fieldnames} for r in rows)
    lookup = {(r['id'],r['label'],r['mode']):r for r in rows}
    assert len(lookup)==700
    threshold = config['threshold']
    summary = {'pairs':50, 'scoring_rows':700, 'seconds_excluding_model_load':time.monotonic()-start,
               'baseline_max_abs_delta':{b:max(abs(r[b]-baseline[r['id'],r['label']][b]) for r in rows if r['mode']=='full_frame') for b in BRANCHES},
               'baseline_decision_changes':{b:sum((r[b]>=threshold)!=(baseline[r['id'],r['label']][b]>=threshold) for r in rows if r['mode']=='full_frame') for b in BRANCHES},
               'by_mode':{}}
    transitions = []
    contributions = {}
    for mode in MODES:
        result = {}
        for branch in BRANCHES:
            info = {}
            for label, prefix in [('provided_clean_reference','reference'),('screen_recapture','recapture')]:
                part = [r for r in rows if r['mode']==mode and r['label']==label]
                assert len(part)==50
                info[prefix+'_above_threshold'] = sum(r[branch]>=threshold for r in part)
                info[prefix+'_mean'] = float(np.mean([r[branch] for r in part]))
                info[prefix+'_delta_vs_original'] = stats([r[branch]-lookup[r['id'],label,'full_frame'][branch] for r in part])
                flips = {'below_to_above':[], 'above_to_below':[]}
                for r in part:
                    prev = lookup[r['id'],label,'full_frame'][branch]
                    if (prev>=threshold)!=(r[branch]>=threshold):
                        direction = 'above_to_below' if prev>=threshold else 'below_to_above'
                        flips[direction].append(r['id'])
                        transitions.append({'id':r['id'],'label':label,'mode':mode,'branch':branch,'direction':direction,'before':prev,'after':r[branch]})
                info[prefix+'_flips'] = flips
            gaps = [lookup[i,'screen_recapture',mode][branch]-lookup[i,'provided_clean_reference',mode][branch] for i in protocol['pair_ids']]
            info['paired_gap'] = stats(gaps)
            info['recapture_misses'] = 50-info['recapture_above_threshold']
            result[branch] = info
        for label,prefix in [('screen_recapture','recapture'),('provided_clean_reference','reference')]:
            part = [r for r in rows if r['mode']==mode and r['label']==label]
            result[prefix+'_tpo_above_ensemble_below']=[r['id'] for r in part if r['tpo']>=threshold and r['ensemble']<threshold]
            result[prefix+'_tpo_below_ensemble_above']=[r['id'] for r in part if r['tpo']<threshold and r['ensemble']>=threshold]
        summary['by_mode'][mode] = result
        contributions[mode] = {}
        for label in ['provided_clean_reference','screen_recapture']:
            ds=np.array([np.array(lookup[i,label,mode]['forensic_terms'])-np.array(lookup[i,label,'full_frame']['forensic_terms']) for i in protocol['pair_ids']])
            contributions[mode][label]=[{'feature':names[k],'mean_logit_delta':float(ds[:,j].mean())} for j,k in enumerate(idx)]
    assert all(v<1e-5 for v in summary['baseline_max_abs_delta'].values())
    assert all(v==0 for v in summary['baseline_decision_changes'].values())
    assert all(sha(ROOT/p)==digest for p,digest in frozen.items())
    assert all(sha(ROOT/r['path'])==r['source_sha256'] for r in inputs)
    summary['frozen_files_and_100_source_hashes_unchanged'] = True
    write(out/'summary.json', summary)
    write(out/'decision_transitions.json', transitions)
    write(out/'forensic_transform_contributions.json', contributions)
    print(json.dumps({'baseline_max_abs_delta':summary['baseline_max_abs_delta'],
                      'seconds':summary['seconds_excluding_model_load'],
                      'conditions':{m:{b:[x[b]['reference_above_threshold'],x[b]['recapture_misses']] for b in BRANCHES} for m,x in summary['by_mode'].items()}},indent=2))


def hashlib_array(array):
    import hashlib
    return hashlib.sha256(array.tobytes()).hexdigest()


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--out',type=Path,required=True)
    main(parser.parse_args().out)
