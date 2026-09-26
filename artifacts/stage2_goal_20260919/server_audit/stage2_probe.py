"""Offline Stage2 probe for a frozen manifest and preserved V7 package; no downloads."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import socket
import sys
import tempfile
import time


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def write(path, value):
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)


def validate_output(result, frame_sets):
    import pandas as pd
    # Same Stage2 contract as scripts/verify_submission.py:49-57; also reject float/bool frame columns.
    columns = ['ID', 'collision_frame', 'entry_frame', 'evasion_space', 'entry_side']
    require(isinstance(result, pd.DataFrame) and list(result.columns) == columns, 'Wrong Stage2 DataFrame columns')
    require(not result.isna().any().any() and result.ID.is_unique, 'Nulls or duplicate IDs')
    require(set(result.ID) == set(frame_sets), 'Missing or additional IDs')
    require(set(result.entry_side) <= {'LEFT', 'RIGHT'}, 'Invalid entry_side')
    require(set(result.evasion_space) <= {0, 1}, 'Invalid evasion_space')
    for column in ('collision_frame', 'entry_frame', 'evasion_space'):
        require(pd.api.types.is_integer_dtype(result[column]), 'Expected integer column: ' + column)
    for row in result.itertuples():
        require(row.collision_frame in frame_sets[row.ID] and row.entry_frame in frame_sets[row.ID], 'Output references absent original frame')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('repo-root', 'package', 'freeze-dir', 'readiness', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--candidate', type=Path, help='Optional frozen research candidate module')
    parser.add_argument('--candidate-freeze', type=Path, help='Required with --candidate')
    args = parser.parse_args()
    if bool(args.candidate) != bool(args.candidate_freeze):
        parser.error('--candidate and --candidate-freeze must be supplied together')
    root, package, frozen_dir, readiness_path, out = [getattr(args, name).resolve()
        for name in ('repo_root', 'package', 'freeze_dir', 'readiness', 'output')]
    out.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    report = dict(status='NOT_STARTED', platform=platform.platform(), python=sys.version,
                  package=str(package), freeze_dir=str(frozen_dir), readiness=str(readiness_path),
                  model_inference_started=False, network_attempts=0, official_S2=None,
                  policy='research_candidate' if args.candidate else 'preserved_v7',
                  stage2_probe_sha256=sha(Path(__file__)),
                  scope='Offline packaged execution and output contract only; not accuracy or full-server time proof')
    exit_code = 1
    originals = (socket.socket.connect, socket.socket.connect_ex, socket.create_connection)
    def deny(*args, **kwargs):
        report['network_attempts'] += 1
        raise RuntimeError('Unexpected Python socket connection during offline probe')
    socket.socket.connect = socket.socket.connect_ex = socket.create_connection = deny
    for key in ('HF_HUB_OFFLINE', 'TRANSFORMERS_OFFLINE', 'HF_HUB_DISABLE_IMPLICIT_TOKEN', 'PYTHONDONTWRITEBYTECODE'):
        os.environ[key] = '1'
    sys.dont_write_bytecode = True
    try:
        import torch
        report.update(torch_version=torch.__version__, cuda_build=torch.version.cuda,
                      cuda_available=torch.cuda.is_available(), cuda_device_count=torch.cuda.device_count())
        if not torch.cuda.is_available():
            report.update(status='NOT_EXECUTED_CUDA_UNAVAILABLE', frozen_hashes_verified=False,
                          package_assets_verified=False, message='CUDA is required by preserved NF4 loader; no model loaded.')
            exit_code = 2
            return exit_code
        require(platform.system() == 'Linux', 'This probe requires the target Linux/CUDA runtime')
        import cv2
        import importlib.metadata
        report['versions'] = {name: importlib.metadata.version(name)
                              for name in ('torch', 'transformers', 'accelerate', 'bitsandbytes', 'pandas', 'Pillow')}
        report['gpu_name'] = torch.cuda.get_device_name(0)
        torch.set_num_threads(2)
        cv2.setNumThreads(2)
        frozen = read(frozen_dir / 'freeze.json')
        manifest_path = frozen_dir / 'evaluation_manifest.json'
        def bound(path):
            key = str(path.relative_to(root))
            require(key in frozen['files'], 'File absent from core freeze: ' + key)
            require(sha(path) == frozen['files'][key], 'Frozen file changed: ' + key)
        bound(manifest_path)
        bound(readiness_path)
        for name, expected in frozen['files'].items():
            path = (root / name).resolve()
            require(path.is_relative_to(root), 'Frozen path escapes repo root')
            require(sha(path) == expected, 'Frozen file changed: ' + name)
        report.update(frozen_hashes_verified=True, frozen_file_count=len(frozen['files']),
                      freeze_sha256=sha(frozen_dir / 'freeze.json'), readiness_sha256=sha(readiness_path))
        audit = read(readiness_path)
        require(audit['readiness']['artifact_integrity_verified'], 'Readiness did not verify package integrity')
        expected_package = audit['packages']['v7']
        shard_sizes = expected_package['stage2_checkpoint']['shards']
        asset_records, before = [], {}
        package_hash_started = time.perf_counter()
        for item in expected_package['records']:
            name = item['path']
            if not (name.startswith('model/stage2/') or name in ('inference.py', 'requirements.txt')):
                continue
            path = package / name
            require(path.is_file(), 'Missing package asset: ' + name)
            stat = path.stat()
            before[name] = [stat.st_size, stat.st_mtime_ns]
            large_weight = name.endswith('.safetensors')
            if large_weight:
                require(stat.st_size == shard_sizes[path.name], 'Weight shard size differs: ' + name)
            require(sha(path) == item['zip_sha256'], 'Package asset differs from audited V7: ' + name)
            asset_records.append(dict(path=name, expected_sha256=item['zip_sha256'], bytes=stat.st_size,
                                      mtime_ns=stat.st_mtime_ns, sha_rechecked_this_run=True))
        report.update(package_assets_verified=True, expected_zip_sha256=expected_package['sha256'],
                      package_asset_hash_seconds=time.perf_counter() - package_hash_started,
                      weight_hash_recheck='All required V7 Stage2 assets, including both NF4 shards, rehashed against bound readiness after CUDA guard.',
                      package_assets=asset_records)
        cases = read(manifest_path)['cases']
        ids = [case['ID'] for case in cases]
        require(cases and len(ids) == len(set(ids)), 'Manifest case IDs must be nonempty and unique')
        frame_sets, mapping = {}, []
        with tempfile.TemporaryDirectory(prefix='stage2_input_', dir=out) as temporary:
            data = Path(temporary)
            for case in cases:
                case_id = case['ID']
                require(Path(case_id).name == case_id and case_id not in ('.', '..'), 'Unsafe case ID')
                folder = data / 'images' / case_id
                folder.mkdir(parents=True)
                numbers = []
                for image in case['images']:
                    source = (root / image['path']).resolve()
                    require(source.is_relative_to(root), 'Image escapes repo root')
                    require(sha(source) == image['sha256'], 'Image hash differs: ' + str(source))
                    number = int(source.stem.rsplit('_', 1)[1])
                    require(number == image['frame'], 'Filename number differs from manifest')
                    require(source.suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}, 'Unsupported image extension')
                    (folder / source.name).symlink_to(source)
                    numbers.append(number)
                    mapping.append(dict(ID=case_id, frame=number, source=str(source), sha256=image['sha256']))
                require(numbers and numbers == sorted(set(numbers)), 'Image numbers must be sorted and unique')
                frame_sets[case_id] = set(numbers)
            write(out / 'input_mapping.json', mapping)
            spec = importlib.util.spec_from_file_location('preserved_stage2_inference', package / 'inference.py')
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            predictor = module.predict_stage2
            predictor_options = {}
            if args.candidate:
                candidate_path = args.candidate.resolve()
                candidate_freeze = read(args.candidate_freeze.resolve())
                candidate_key = str(candidate_path.relative_to(root))
                require(candidate_key in candidate_freeze['files'], 'Candidate module absent from runtime freeze')
                for name, expected in candidate_freeze['files'].items():
                    path = (root / name).resolve()
                    require(path.is_relative_to(root) and sha(path) == expected, 'Candidate evidence changed: ' + name)
                module._runtime(package / 'model/stage2')
                candidate_spec = importlib.util.spec_from_file_location('frozen_stage2_candidate', candidate_path)
                candidate_module = importlib.util.module_from_spec(candidate_spec)
                candidate_spec.loader.exec_module(candidate_module)
                predictor = candidate_module.predict_stage2
                predictor_options['trace_dir'] = out / 'candidate_traces'
                report.update(candidate_sha256=sha(candidate_path),
                              candidate_freeze_sha256=sha(args.candidate_freeze.resolve()))
            report['preflight_seconds'] = time.perf_counter() - started
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            inference_started = time.perf_counter()
            report['model_inference_started'] = True
            try:
                result = predictor(data, package / 'model/stage2', **predictor_options)
                torch.cuda.synchronize()
            finally:
                report.update(inference_seconds=time.perf_counter() - inference_started,
                              cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                              cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved())
            validate_output(result, frame_sets)
            for name, loaded in list(sys.modules.items()):
                if (name == 'solution' or name.startswith('solution.')) and getattr(loaded, '__file__', None):
                    require(Path(loaded.__file__).resolve().is_relative_to(package), 'Runtime imported outside preserved package')
            result.to_csv(out / 'stage2.csv', index=False, encoding='utf-8')
            report.update(status='PASS_OUTPUT_CONTRACT_ONLY', rows=len(result), output_sha256=sha(out / 'stage2.csv'))
        require(all([ (package / name).stat().st_size, (package / name).stat().st_mtime_ns] == state
                    for name, state in before.items()), 'Package file metadata changed during inference')
        require(report['network_attempts'] == 0, 'Offline guard observed network attempts')
        report['package_metadata_unchanged'] = True
        exit_code = 0
    except Exception as error:
        report.update(status='FAIL', error_type=type(error).__name__, error=str(error))
    finally:
        socket.socket.connect, socket.socket.connect_ex, socket.create_connection = originals
        report.update(exit_code=exit_code, total_probe_seconds=time.perf_counter() - started)
        write(out / 'report.json', report)
        print(json.dumps({k: report[k] for k in ('status', 'exit_code', 'model_inference_started', 'network_attempts')}, ensure_ascii=False))
    return exit_code


if __name__ == '__main__':
    raise SystemExit(main())
