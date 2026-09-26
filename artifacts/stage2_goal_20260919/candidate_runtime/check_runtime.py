"""Replay all frozen calls or freshly infer one declared case with the same candidate."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import socket
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
for key, value in dict(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', HF_HUB_DISABLE_IMPLICIT_TOKEN='1',
                       TOKENIZERS_PARALLELISM='false', OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2',
                       MKL_NUM_THREADS='2', BLACKBOX_DEEPSTACK_FIX='1', BLACKBOX_DECODE_MODE='sync',
                       BLACKBOX_COMPUTE_DTYPE='native').items():
    os.environ[key] = value
sys.path.insert(0, str(ROOT / 'artifacts/submissions/verify_v6/model/stage2/code'))
sys.path.insert(0, str(ROOT / 'scripts/mac'))
import numpy as np
from PIL import Image
import candidate


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def save(path, data):
    with path.open('x') as stream:
        json.dump(data, stream, ensure_ascii=False, indent=2, allow_nan=False)


class Replay:
    def __init__(self, calls):
        self.calls, self.index = calls, 0

    def ask(self, images, prompt, max_new_tokens=128):
        record = self.calls[self.index]
        assert record['prompt'] == prompt and record['max_new_tokens'] == max_new_tokens
        hashes, sizes = [], []
        for image in images:
            image = image.convert('RGB')
            budget = max(1024, 1_200_000 // len(images))
            scale = min(1.0, math.sqrt(budget / (image.width * image.height)))
            size = (max(32, int(image.width * scale) // 32 * 32), max(32, int(image.height * scale) // 32 * 32))
            bounded = image.resize(size, Image.Resampling.BICUBIC)
            hashes.append(hashlib.sha256(bounded.tobytes()).hexdigest())
            sizes.append(list(size))
        assert hashes == record['image_sha256'] and sizes == record['image_sizes']
        self.index += 1
        return record['text']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['replay', 'live'])
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    report = dict(status='RUNNING', mode=args.mode, new_model_calls=0, network_attempts=0,
                  official_S2=None, production_changed=False, independent_validation=False, cases=[])
    originals = socket.socket.connect, socket.socket.connect_ex, socket.create_connection
    def deny(*args, **kwargs):
        report['network_attempts'] += 1
        raise RuntimeError('Unexpected Python network connection')
    socket.socket.connect = socket.socket.connect_ex = socket.create_connection = deny
    try:
        frozen = read(HERE / 'freeze.json')
        for path, digest in frozen['files'].items():
            assert sha(ROOT / path) == digest, path
        core = ROOT / frozen['core_dir']
        manifest = read(core / 'evaluation_manifest.json')
        expected = {c['ID']: c for c in read(ROOT / frozen['ablation_result'])['cases']}
        if args.mode == 'replay':
            assert candidate.scores_from_features(np.zeros((1, 3), np.float32)).tolist() == [0.]
            short = np.array([[0., 0., 0.], [1., 500., 500.]], np.float32)
            assert int(np.argmax(candidate.scores_from_features(short))) == 1
            for bad in [np.empty((0, 3)), np.zeros((2, 2)), np.array([[0., np.nan, 0.]])]:
                try:
                    candidate.scores_from_features(bad)
                except ValueError:
                    continue
                raise AssertionError('Invalid feature array accepted')
        selected = [c for c in manifest['cases'] if args.mode == 'replay' or c['ID'] == frozen['live_case']]
        for case in selected:
            paths = [ROOT / image['path'] for image in case['images']]
            for path, image in zip(paths, case['images']):
                assert sha(path) == image['sha256'], path
            result_path = ROOT / case['prediction_source']
            old = read(result_path)
            motion = np.load(result_path.parent / 'motion.npz')
            if args.mode == 'live':
                import torch
                from mlx_stage2 import MLXVLM
                torch.set_num_threads(2)
                candidate.reference.primitives.cv2.setNumThreads(2)
                np.random.seed(0)
                torch.manual_seed(0)
                vlm = MLXVLM(ROOT / 'artifacts/mac_experiments/stage2_mlx/model', out)
                valid, base, new, features = candidate.reference._dual_motion_scan(paths)
                assert valid == paths
                for name, value in [('features', features), ('base_scores', base), ('new_scores', new)]:
                    assert value.dtype == motion[name].dtype and value.tobytes() == motion[name].tobytes(), name
            else:
                vlm = Replay(old['calls'])
                base, new, features = motion['base_scores'], motion['new_scores'], motion['features']
            before = [v.tobytes() for v in (base, new, features)]
            prediction, diagnostics = candidate.predict_file(paths, base, new, features, vlm)
            assert before == [v.tobytes() for v in (base, new, features)]
            assert prediction == expected[case['ID']]['candidate']
            assert diagnostics['baseline_prediction'] == old['baseline_prediction']
            assert diagnostics['contact_ablation']['new_score_sha256'] == expected[case['ID']]['candidate_score_sha256']
            if args.mode == 'live':
                report['new_model_calls'] += len(vlm.calls)
                assert len(vlm.calls) == len(old['calls']) == 4
                for new_call, old_call in zip(vlm.calls, old['calls']):
                    for field in ['text', 'prompt_sha256', 'image_sha256', 'processor_input_sha256', 'token_trace']:
                        assert new_call[field] == old_call[field], field
                save(out / 'result.json', dict(ID=case['ID'], prediction=prediction, diagnostics=diagnostics,
                                               calls=vlm.calls, model_load_seconds=vlm.load_seconds))
            else:
                assert vlm.index == 4
            report['cases'].append(dict(ID=case['ID'], prediction=prediction, baseline=diagnostics['baseline_prediction'],
                                        four_call_input_equivalence=True, other_three_fields_equal=True,
                                        candidate_score_hash_matches_ablation=True))
        assert report['network_attempts'] == 0
        report['status'] = 'PASS'
    except Exception as error:
        report.update(status='FAIL', error_type=type(error).__name__, error=str(error))
        raise
    finally:
        socket.socket.connect, socket.socket.connect_ex, socket.create_connection = originals
        report['wall_seconds'] = time.perf_counter() - started
        save(out / 'report.json', report)
        print(json.dumps({k: v for k, v in report.items() if k != 'cases'}, ensure_ascii=False))


if __name__ == '__main__':
    main()
