"""Single frozen nine-call native-video contact diagnostic, never automatic adoption."""
import argparse
import hashlib
import importlib.util
import json
import os
import socket
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.dont_write_bytecode = True
for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[name] = '2'
os.environ['HF_HUB_OFFLINE'] = os.environ['TRANSFORMERS_OFFLINE'] = '1'
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PACKAGE = ROOT / 'artifacts/submissions/verify_v6'
IDS = ['00000', '00003', '00004', '00005', '00006', '00007', '00008', '00010', '00013']
OLD_PREFIX = 'These dashcam frames are ordered left to right, top to bottom. '

def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8-sig'))

def sha(p):
    with Path(p).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()

def put(p, obj):
    with Path(p).open('x', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, allow_nan=False)

def require(ok, message):
    if not ok:
        raise ValueError(message)

def load(name, p):
    spec = importlib.util.spec_from_file_location(name, p)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    require(sys.flags.isolated and output.is_relative_to(HERE) and not output.exists(), 'Isolated Python and new research output required')
    protocol_path = HERE / 'native_video_protocol.json'
    helper_path = HERE / 'solution/native_video_v7.py'
    cpu_path = HERE / 'native_video_cpu_probe/report.json'
    cpu = read(cpu_path)
    require(cpu['status'] == 'passed' and cpu['helper_sha256'] == sha(helper_path), 'Matching CPU processor contract PASS required')
    corpus_path = HERE / 'run_capacity.py'
    protocol = read(protocol_path)
    require(protocol['ids'] == IDS, 'Fixed nine IDs required')
    corpus = load('native_corpus', corpus_path)
    rows, bindings = corpus.corpus()
    by_id = {r['ID']: r for r in rows}
    full_path = ROOT / 'research/v6_stage2/v5_fullframe_diagnostic/report.json'
    replay_path = ROOT / 'research/v6_stage2/uncapped_jerk_fullframe_replay/freeze.json'
    require(sha(full_path) == read(replay_path)['files'][str(full_path)], 'Original calls changed')
    full = {r['ID']: r for r in read(full_path)['videos']}
    protected = [Path(__file__), protocol_path, helper_path, cpu_path, corpus_path, full_path, replay_path,
                 ROOT / 'artifacts/submissions/submit_v6.manifest.json']
    for item in read(protected[-1])['files']:
        path = PACKAGE / item['path']
        require(sha(path) == item['sha256'], 'Actual V6 package changed')
        bindings[str(path)] = item['sha256']
    bindings.update({str(p): sha(p) for p in protected})
    sys.path.insert(0, str(PACKAGE / 'model/stage2/code'))
    from solution import stage2_v2 as base
    helper = load('native_video_helper', helper_path)
    prepared = []
    for ID in IDS:
        row = by_id[ID]
        record, trace = row['record'], row['trace']
        original = trace if trace.get('calls') else full[ID]
        require(len(original['calls']) == 4, 'Need original four-call record')
        call = original['calls'][1]
        require(call['max_new_tokens'] == 48 and call['status'] == 'complete' and call['prompt'].startswith(OLD_PREFIX), 'Original call2 contract differs')
        # Nested diagnostics differ only in old/full trace serialization.
        diagnostics = original['diagnostics']
        offered = diagnostics['collision_candidates']
        require(offered == sorted(set(offered)) and offered, 'Candidate IDs invalid')
        require(f'Allowed frames: {offered}. ' in call['prompt'], 'Question and actual offered candidates differ')
        images = {r['frame']: r for r in record['input']['input_images']}
        all_paths = [Path(record['input_root']) / r['path'] for r in record['input']['input_images']]
        numbers = [base._frame_number(p) for p in all_paths]
        require(numbers == trace['frame_numbers'], 'Original frame mapping differs')
        indices = [numbers.index(n) for n in offered]
        selected = []
        for number, index in zip(offered, indices):
            path = all_paths[index]
            require(sha(path) == images[number]['file_sha256'], 'Candidate source pixels changed')
            bindings[str(path)] = images[number]['file_sha256']
            selected.append({'frame': number, 'path': str(path), 'sha256': images[number]['file_sha256']})
        source = Path(record['input']['source_path'])
        require(sha(source) == record['input']['source_sha256'], 'Source video changed')
        bindings[str(source)] = record['input']['source_sha256']
        prepared.append({'ID': ID, 'row': row, 'paths': all_paths, 'indices': indices, 'offered': offered,
                         'selected': selected, 'original_call': call, 'prompt': call['prompt'][len(OLD_PREFIX):],
                         'original_internal_frame': diagnostics['collision']['collision_frame']})
    output.mkdir()
    put(output / 'freeze.json', {'created_utc': datetime.now(timezone.utc).isoformat(), 'files': bindings,
        'ids': IDS, 'source_role': 'exposed_development', 'human_labels_read': False,
        'selected': {r['ID']: r['selected'] for r in prepared},
        'protocol_sha256': sha(protocol_path), 'no_automatic_adoption': True})
    report = {'status': 'running', 'videos': [], 'network_attempts': 0, 'call_count': 0,
              'model_loads': 0, 'source_role': 'exposed_development', 'cached_other_fields': True,
              'end_to_end_replay_performed': False, 'adopted': False, 'submission_approved': False,
              'freeze_sha256': sha(output / 'freeze.json')}
    def deny(*args, **kwargs):
        report['network_attempts'] += 1
        raise RuntimeError('Offline experiment attempted network')
    socket.socket.connect = socket.socket.connect_ex = socket.create_connection = deny
    start = time.perf_counter()
    try:
        from solution.vlm_candidate import CandidateVLM
        with CandidateVLM(PACKAGE / 'model/stage2/vlm', precision='nf4') as model:
            report['model_loads'] = 1
            report['runtime'] = model.candidate_metadata
            model.torch.set_num_threads(2)
            require(model.candidate_metadata['prequantized_checkpoint'], 'Saved NF4 required')
            for item in prepared:
                require(time.perf_counter() - start < 600, 'Frozen wall budget exceeded')
                folder = output / item['ID']
                folder.mkdir()
                tiles = [base._sheet(item['paths'], [i], columns=1) for i in item['indices']]
                for index, tile in enumerate(tiles):
                    tile.save(folder / f'input_{index:02d}.png')
                report['call_count'] += 1
                raw, video = helper.ask_native_video(model, tiles, item['prompt'], max_new_tokens=48)
                parsed = base._json_object(raw)
                selected = base._integer(parsed.get('collision_frame'))
                valid = selected in item['offered']
                baseline = dict(item['row']['trace']['candidate'])
                predicted = selected if valid else baseline['collision_frame']
                result = {'ID': item['ID'], 'baseline': baseline,
                    'candidate': {**baseline, 'collision_frame': predicted},
                    'frame_pts': item['row']['record']['input']['source_frame_pts'],
                    'source_sha256': item['row']['record']['input']['source_sha256'],
                    'offered_frames': item['offered'], 'selected_inputs': item['selected'],
                    'raw': raw, 'parsed': parsed, 'accepted_offered_frame': valid,
                    'fallback': not valid, 'fallback_reason': None if valid else ('not_parseable' if selected is None else 'out_of_offered'),
                    'original_internal_frame': item['original_internal_frame'],
                    'original_call2': item['original_call'], 'video': video,
                    'other_three_fields_copied': True, 'end_to_end_replay_performed': False}
                put(folder / 'result.json', result)
                report['videos'].append(result)
                print(json.dumps({'ID': item['ID'], 'status': 'complete', 'fallback': not valid}), flush=True)
        require(report['call_count'] == 9 and report['network_attempts'] == 0, 'Execution contract failed')
        require(time.perf_counter() - start <= 600, 'Frozen total wall budget exceeded')
        require(all(sha(path) == digest for path, digest in bindings.items()), 'Bindings changed')
        report.update(status='complete', bindings_unchanged=True)
    except BaseException as error:
        report.update(status='failed', error=repr(error))
        raise
    finally:
        report['seconds'] = time.perf_counter() - start
        put(output / 'report.json', report)

if __name__ == '__main__':
    main()
