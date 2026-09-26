"""Read-only package/runtime audit. Never imports a model or contacts a server."""
from pathlib import Path
import ast
import datetime
import glob
import hashlib
import json
import shlex
import struct
import subprocess
import zipfile

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent


def digest(stream):
    h = hashlib.sha256()
    for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
        h.update(block)
    return h.hexdigest()


def filehash(path):
    before = path.stat()
    with path.open('rb') as stream:
        sha = digest(stream)
    after = path.stat()
    assert (before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns), path
    return sha


def entry_ast(source):
    tree = ast.parse(source)
    return {n.name: ast.dump(n, include_attributes=False) for n in tree.body
            if isinstance(n, ast.FunctionDef) and n.name in ('_runtime', 'predict_stage2')}


def checkpoint_structure(folder, archive=None):
    def open_file(name):
        return archive.open(str(folder / name)) if archive else (folder / name).open('rb')
    def file_size(name):
        return archive.getinfo(str(folder / name)).file_size if archive else (folder / name).stat().st_size
    with open_file('model.safetensors.index.json') as stream:
        index = json.load(stream)
    shards = sorted(set(index['weight_map'].values()))
    errors = []
    tensors = {}
    sizes = {}
    for name in shards:
        path = folder / name
        if not archive and not path.is_file():
            errors.append('missing shard: ' + name)
            continue
        sizes[name] = file_size(name)
        with open_file(name) as stream:
            header_size = struct.unpack('<Q', stream.read(8))[0]
            assert header_size < sizes[name], name
            header = json.loads(stream.read(header_size))
        tensors[name] = set(header) - {'__metadata__'}
        for key, value in header.items():
            if key == '__metadata__':
                continue
            start, stop = value['data_offsets']
            if not 0 <= start <= stop <= sizes[name] - 8 - header_size:
                errors.append('invalid tensor offset: ' + key)
    for key, name in index['weight_map'].items():
        if key not in tensors.get(name, set()):
            errors.append('index tensor missing: ' + key)
    return {'index_path': str(folder / 'model.safetensors.index.json'), 'inside_zip': bool(archive),
            'mapped_tensors': len(index['weight_map']), 'shards': sizes,
            'index_header_offsets_valid': not errors, 'errors': errors,
            'model_loaded': False}


def mlx_runtime_structure(folder):
    # Installed mlx-vlm 0.3.4 loads the actual *.safetensors files, not the HF index.
    files, errors = [], []
    for path in sorted(folder.glob('*.safetensors')):
        size = path.stat().st_size
        with path.open('rb') as stream:
            header_size = struct.unpack('<Q', stream.read(8))[0]
            assert header_size < size, path
            header = json.loads(stream.read(header_size))
        names = set(header) - {'__metadata__'}
        for name in names:
            start, stop = header[name]['data_offsets']
            if not 0 <= start <= stop <= size - 8 - header_size:
                errors.append('invalid tensor offset: ' + name)
        files.append({'path': str(path), 'bytes': size, 'header_tensor_count': len(names)})
    loader = ROOT / 'artifacts/mac_experiments/stage2_mlx/.venv/lib/python3.12/site-packages/mlx_vlm/utils.py'
    return {'loader_path': str(loader), 'loader_sha256': filehash(loader),
            'loader_method': 'load_model uses glob *.safetensors at utils.py:142 and mx.load at :167; HF index is not consulted',
            'files': files, 'actual_files_header_offsets_valid': bool(files) and not errors,
            'errors': errors, 'model_loaded': False,
            'note': 'The original HF shard index refers to absent shard files. Its inconsistency is retained; it does not establish failure of the installed MLX glob loader.'}


def package(version):
    manifest_path = ROOT / f'releases/{version}/submission.manifest.json'
    manifest = json.loads(manifest_path.read_text())
    archive_path = ROOT / f'artifacts/submissions/submit_{version}.zip'
    archive_sha = filehash(archive_path)
    records = []
    contents = {}
    with zipfile.ZipFile(archive_path) as archive:
        names = [x.filename for x in archive.infolist() if not x.is_dir()]
        expected_names = [x['path'] for x in manifest['files']]
        for record in manifest['files']:
            name = record['path']
            with archive.open(name) as stream:
                sha = digest(stream)  # Also verifies the stored member CRC on EOF.
            item = {'path': name, 'zip_sha256': sha,
                    'manifest_match': sha == record['sha256'],
                    'zip_bytes_match': archive.getinfo(name).file_size == record['bytes']}
            if name.startswith('model/stage2/') or name in ('inference.py', 'requirements.txt'):
                item['copies'] = {}
                for label, folder in [
                    ('extracted', ROOT / f'artifacts/submissions/verify_{version}'),
                    ('release_source', ROOT / f'releases/{version}/source')]:
                    path = folder / name
                    actual = filehash(path) if path.is_file() else None
                    item['copies'][label] = {'path': str(path), 'exists': path.is_file(), 'sha256': actual,
                                            'matches_zip': actual == sha}
            records.append(item)
        contents['entry_ast'] = entry_ast(archive.read('inference.py').decode())
        contents['config'] = json.loads(archive.read('model/stage2/vlm/config.json'))
        contents['checkpoint'] = checkpoint_structure(Path('model/stage2/vlm'), archive)
    return {'version': version, 'zip_path': str(archive_path),
            'zip_bytes': archive_path.stat().st_size, 'sha256': archive_sha,
            'manifest_sha256': filehash(manifest_path), 'manifest_path': str(manifest_path),
            'zip_sha_matches_manifest': archive_sha == manifest['sha256'],
            'file_count': len(names), 'member_names_match_manifest': sorted(names) == sorted(expected_names),
            'duplicate_member_names': len(names) != len(set(names)),
            'all_member_sha_and_size_match': all(x['manifest_match'] and x['zip_bytes_match'] for x in records),
            'records': records,
            'stage2_checkpoint': contents['checkpoint'],
            'stage2_config': {k: contents['config'].get(k) for k in ('model_type', 'architectures', 'quantization_config')},
            'stage2_entry_ast': contents['entry_ast']}


def environment(python):
    probe = '''import sys,platform,importlib.metadata,json
r={'python':sys.version,'executable':sys.executable,'platform':platform.platform(),'machine':platform.machine(),'dependencies':{}}
for name in ['torch','transformers','bitsandbytes','accelerate','safetensors','numpy','opencv-python','opencv-python-headless','Pillow','pandas','mlx','mlx-vlm','scipy']:
 try:r['dependencies'][name]=importlib.metadata.version(name)
 except importlib.metadata.PackageNotFoundError:r['dependencies'][name]=None
import torch
r.update(cuda_available=torch.cuda.is_available(),cuda_build=torch.version.cuda,cuda_device_count=torch.cuda.device_count(),mps_available=torch.backends.mps.is_available())
print(json.dumps(r))'''
    completed = subprocess.run([str(python), '-B', '-c', probe], capture_output=True, text=True, check=True)
    return json.loads(completed.stdout)


def ssh_aliases():
    seen, configs = set(), []
    def scan(path):
        path = path.resolve()
        if path in seen or not path.is_file():
            return
        seen.add(path)
        aliases = []
        for line in path.read_text(errors='replace').splitlines():
            try:
                words = shlex.split(line, comments=True)
            except ValueError:
                continue
            if not words:
                continue
            if words[0].lower() == 'host':
                aliases.extend(x for x in words[1:] if not any(c in x for c in '*?!'))
            if words[0].lower() == 'include':
                for pattern in words[1:]:
                    location = Path(pattern).expanduser()
                    if not location.is_absolute():
                        location = (Path.home() / '.ssh' if path.is_relative_to(Path.home()) else Path('/etc/ssh')) / location
                    for included in glob.glob(str(location)):
                        scan(Path(included))
        configs.append({'path': str(path), 'literal_host_aliases': aliases})
    scan(Path.home() / '.ssh/config')
    scan(Path('/etc/ssh/ssh_config'))
    return {'user_config_exists': (Path.home() / '.ssh/config').is_file(),
            'configs': configs, 'remote_connections_attempted': 0, 'credentials_read': False,
            'configured_literal_alias_count': sum(len(x['literal_host_aliases']) for x in configs)}


def main():
    target = OUT / 'readiness.json'
    if target.exists():
        raise FileExistsError(target)
    packages = {version: package(version) for version in ('v6', 'v7')}
    maps = {version: {x['path']: x['zip_sha256'] for x in data['records']}
            for version, data in packages.items()}
    common = sorted(set(maps['v6']) & set(maps['v7']))
    changed = [p for p in common if maps['v6'][p] != maps['v7'][p]]
    added = sorted(set(maps['v7']) - set(maps['v6']))
    stage2_common = [p for p in common if p.startswith('model/stage2/')]
    mlx_dir = ROOT / 'artifacts/mac_experiments/stage2_mlx'
    mlx_records = []
    for item in json.loads((mlx_dir / 'model_integrity.json').read_text()):
        path = mlx_dir / 'model' / item['file']
        actual = filehash(path) if path.is_file() else None
        mlx_records.append({'file': item['file'], 'sha256': actual,
                            'recorded_sha256': item['sha256'], 'matches': actual == item['sha256']})
    source_paths = [ROOT / 'scripts/mac' / name for name in
                    ('run_stage2.py', 'mlx_stage2.py', 'deepstack_fix.py', 'sync_decode.py')]
    source_paths += [mlx_dir / 'run.sh', mlx_dir / 'run_stage2.py', mlx_dir / 'model_integrity.json', mlx_dir / 'model_metadata.json']
    result = {
        'created_at': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'scope': 'Read-only package bytes, config, checkpoint header/index and runtime inventory. No model inference or remote connection.',
        'audit_script_sha256': filehash(Path(__file__)),
        'packages': packages,
        'v6_v7_comparison': {
            'all_common_changed_paths': changed, 'v7_added_paths': added,
            'stage2_common_member_count': len(stage2_common),
            'stage2_common_members_byte_identical': all(maps['v6'][p] == maps['v7'][p] for p in stage2_common),
            'stage2_entry_function_and_runtime_ast_identical': packages['v6']['stage2_entry_ast'] == packages['v7']['stage2_entry_ast'],
            'note': 'model/stage2/code also stores Stage1/3 code; the sole newly added V7 module serves Stage1.'},
        'mac_research': {
            'model_path': str(mlx_dir / 'model'),
            'recorded_file_count': len(mlx_records), 'all_recorded_hashes_match': all(x['matches'] for x in mlx_records),
            'files': mlx_records, 'checkpoint_structure': checkpoint_structure(mlx_dir / 'model'),
            'runtime_checkpoint_structure': mlx_runtime_structure(mlx_dir / 'model'),
            'source_hashes': {str(p.relative_to(ROOT)): filehash(p) for p in source_paths},
            'same_weights_or_quantization_as_cuda_nf4': False,
            'fresh_model_loading_or_inference_performed': False},
        'environments': {
            name: environment(ROOT / f'artifacts/mac_experiments/{name}/.venv/bin/python')
            for name in ('scipy_compat', 'stage2_mlx')},
        'remote_path_inventory': ssh_aliases(),
        'readiness': {
            'artifact_integrity_verified': all(x['zip_sha_matches_manifest'] and x['all_member_sha_and_size_match'] and x['member_names_match_manifest'] and not x['duplicate_member_names'] and x['stage2_checkpoint']['index_header_offsets_valid'] for x in packages.values()),
            'all_existing_source_and_extracted_copies_match_zip': all(c['matches_zip'] for x in packages.values() for r in x['records'] for c in r.get('copies', {}).values() if c['exists']),
            'unmaterialized_copy_files': [{'version': version, 'path': r['path'], 'copy': name} for version,x in packages.items() for r in x['records'] for name,c in r.get('copies', {}).items() if not c['exists']],
            'offline_checkpoint_file_completeness_verified': all(x['stage2_checkpoint']['index_header_offsets_valid'] for x in packages.values()),
            'offline_cuda_model_load_verified_this_run': False,
            'linux_cuda_inference_verified_this_run': False,
            'l40s_runtime_or_60minute_limit_verified': False,
            'server_execution_ready': False,
            'blocking_evidence': ['Local host is macOS arm64, torch CUDA unavailable.', 'bitsandbytes is absent in inspected environments.', 'No literal SSH aliases configured in user/system config inventory; no remote host was verified.'],
            'required_next_checks': ['Provide an existing authorized Linux CUDA host/runtime.', 'Run preserved package offline on a frozen full-frame image manifest.', 'Check all four Stage2 outputs, input/output hashes, time and CUDA memory on that host.', 'Compare CPU motion arrays on identical image bytes before attributing differences to VLM quantization.'],
            'unperformed': ['model load', 'model inference', 'CPU optical-flow recomputation', 'remote login', 'download', 'ZIP extraction', 'source or submission edits']}}
    target.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    assert result['readiness']['artifact_integrity_verified'], 'package integrity mismatch; see readiness.json'
    assert result['mac_research']['all_recorded_hashes_match'], 'MLX asset mismatch; see readiness.json'
    print(json.dumps({'result': str(target), 'package_integrity': result['readiness']['artifact_integrity_verified'],
                      'comparison': result['v6_v7_comparison'], 'mlx_hashes_match': result['mac_research']['all_recorded_hashes_match'],
                      'server_execution_ready': False}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
