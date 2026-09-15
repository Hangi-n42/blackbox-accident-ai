"""One actual space ask/video; all other outputs explicitly copied from frozen V6."""
import argparse, hashlib, importlib.util, json, os, socket, sys, time
from datetime import datetime, timezone
from pathlib import Path
sys.dont_write_bytecode = True
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '2'
os.environ['HF_HUB_OFFLINE'] = os.environ['TRANSFORMERS_OFFLINE'] = '1'
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PACKAGE = ROOT / 'artifacts/submissions/verify_v6'
IDS = ['00000', '00003', '00004', '00005', '00006', '00007', '00008', '00010', '00013']

def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))

def sha(path):
    with Path(path).open('rb') as file:
        return hashlib.file_digest(file, 'sha256').hexdigest()

def put(path, value):
    with Path(path).open('x', encoding='utf-8') as file:
        json.dump(value, file, ensure_ascii=False, indent=2, allow_nan=False)

def require(condition, message):
    if not condition:
        raise ValueError(message)

def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

class Recorder:
    def __init__(self, model, folder, prompt):
        self.model, self.folder, self.prompt, self.calls = model, folder, prompt, []

    def ask(self, images, prompt, max_new_tokens=128):
        require(not self.calls and prompt == self.prompt and max_new_tokens == 40,
                'Exactly one unchanged space prompt with40tokens required')
        require(len(images) == 1, 'Expected one original three-panel sheet')
        folder = self.folder / 'call_4'
        folder.mkdir()
        image_records = []
        for index, image in enumerate(images):
            path = folder / f'input_{index}.png'
            image.save(path)
            image_records.append(dict(path=str(path), sha256=sha(path), size=list(image.size)))
        record = dict(prompt=prompt, max_new_tokens=40, input_sizes=[list(x.size) for x in images],
                      images=image_records, status='running', original_call_number=4)
        self.calls.append(record)
        start = time.perf_counter()
        try:
            raw = self.model.ask(images, prompt, max_new_tokens=40)
            record.update(status='complete', raw=raw)
            return raw
        except BaseException as error:
            record.update(status='failed', error=type(error).__name__)
            raise
        finally:
            record['seconds'] = time.perf_counter() - start
            put(folder / 'record.json', record)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    require(sys.flags.isolated and output.is_relative_to(HERE) and not output.exists(),
            'Use isolated Python and a new research/v7 output directory')
    protocol_path = HERE / 'space_anchor_protocol.json'
    protocol = read(protocol_path)
    require(protocol['ids'] == IDS, 'Unexpected cohort')
    helper_path = HERE / 'solution/stage2_space_anchor_v7.py'
    corpus_path = HERE / 'run_capacity.py'
    corpus = load('space_anchor_corpus', corpus_path)
    rows, bindings = corpus.corpus()
    by_id = {row['ID']: row for row in rows}
    require(set(by_id) == set(IDS), 'Expected fixed nine sources')
    rows = [by_id[ID] for ID in IDS]
    inputs = [Path(__file__), helper_path, corpus_path, protocol_path,
              ROOT / 'artifacts/submissions/submit_v6.manifest.json']
    bindings.update({str(p): sha(p) for p in inputs})
    manifest = read(inputs[-1])
    require(len(manifest['files']) == 47, 'Expected frozen47-file V6 package')
    for item in manifest['files']:
        path = PACKAGE / item['path']
        require(sha(path) == item['sha256'], f'Changed V6 package file: {path}')
        bindings[str(path)] = item['sha256']
    # First six C traces contain no asks; use their SHA-bound original full V5/V6-policy calls.
    full_path = ROOT / 'research/v6_stage2/v5_fullframe_diagnostic/report.json'
    replay_freeze_path = ROOT / 'research/v6_stage2/uncapped_jerk_fullframe_replay/freeze.json'
    replay_freeze = read(replay_freeze_path)
    require(sha(full_path) == replay_freeze['files'][str(full_path)], 'Original calls changed')
    bindings.update({str(p): sha(p) for p in (full_path, replay_freeze_path)})
    full = {r['ID']: r for r in read(full_path)['videos']}
    sys.path.insert(0, str(PACKAGE / 'model/stage2/code'))
    import numpy as np
    from solution import stage2_uncapped_jerk_v6c as v6
    v6.primitives.cv2.setNumThreads(2)
    helper = load('solution.stage2_space_anchor_v7', helper_path)
    prepared = []
    for row in rows:
        ID, record, trace = row['ID'], row['record'], row['trace']
        paths = [Path(record['input_root']) / x['path'] for x in record['input']['input_images']]
        numbers = [v6.primitives._frame_number(path) for path in paths]
        require(numbers == trace['frame_numbers'] and len(numbers) == len(set(numbers)), 'Frame order mismatch')
        source = Path(record['input']['source_path'])
        require(sha(source) == record['input']['source_sha256'], 'Source changed')
        bindings[str(source)] = record['input']['source_sha256']
        features = np.asarray(trace['features'], dtype=np.float32)
        base, new = v6._scores_from_features(features)
        require(base.tobytes() == np.asarray(trace['base_scores'], dtype=np.float32).tobytes()
                and new.tobytes() == np.asarray(trace['new_scores'], dtype=np.float32).tobytes(), 'Cached scores differ')
        motion = int(np.argmax(new))
        baseline = trace['candidate']
        require(numbers[motion] == baseline['collision_frame'], 'Cached final V6 collision differs')
        original = trace if trace.get('calls') else full[ID]
        require(len(original['calls']) == 4, 'Missing original four-call trace')
        original_prediction = original.get('candidate', original.get('prediction'))
        require(all(original_prediction[k] == baseline[k] for k in ('entry_frame', 'entry_side', 'evasion_space')),
                'Original call source differs from V6 unchanged fields')
        call = original['calls'][3]
        require(call['max_new_tokens'] == 40 and call['status'] == 'complete', 'Invalid original space call')
        context = sorted({max(0, motion-2), motion, min(len(paths)-1, motion+2)})
        context_inputs = []
        for index in context:
            item = record['input']['input_images'][index]
            path = paths[index]
            require(sha(path) == item['file_sha256'], 'Selected context PNG changed')
            bindings[str(path)] = item['file_sha256']
            context_inputs.append(dict(index=index, frame=numbers[index], path=str(path), sha256=item['file_sha256']))
        prepared.append(dict(row=row, paths=paths, scores=new, prompt=call['prompt'], context=context_inputs))
    require(all(sha(path) == digest for path, digest in bindings.items()), 'Pre-freeze binding changed')
    output.mkdir()
    put(output / 'freeze.json', dict(created_utc=datetime.now(timezone.utc).isoformat(), files=bindings,
        protocol_sha256=sha(protocol_path), ids=IDS, source_role='exposed_development', model='frozen_V6_prequantized_4B',
        selected_contexts={p['row']['ID']:p['context'] for p in prepared}, actual_calls=9,
        cached_other_fields=True, end_to_end_equivalence_verified=False, human_answers_read=False))
    report = dict(status='running', videos=[], network_attempts=0, model_loads=0, call_count=0,
        max_calls=9, source_role='exposed_development', freeze_sha256=sha(output/'freeze.json'),
        protocol_sha256=sha(protocol_path), cached_other_fields=True, end_to_end_equivalence_verified=False,
        human_answers_read_by_inference=False, adopted=False, submission_approved=False)
    def deny(*args, **kwargs):
        report['network_attempts'] += 1
        raise RuntimeError('Offline diagnostic attempted network')
    socket.socket.connect = socket.socket.connect_ex = socket.create_connection = deny
    try:
        from solution.vlm_candidate import CandidateVLM
        with CandidateVLM(PACKAGE / 'model/stage2/vlm', precision='nf4') as model:
            report['model_loads'] = 1
            model.torch.set_num_threads(2)
            require(model.candidate_metadata['prequantized_checkpoint'], 'Expected saved NF4 weights')
            report['runtime'] = model.candidate_metadata
            for item in prepared:
                row = item['row']; record = row['record']; ID = row['ID']
                folder = output / ID; folder.mkdir()
                recorder = Recorder(model, folder, item['prompt'])
                start = time.perf_counter()
                try:
                    value, diagnostics = helper.predict_space_only(item['paths'], item['scores'], recorder, item['prompt'], max_new_tokens=40)
                finally:
                    report['call_count'] += len(recorder.calls)
                require(len(recorder.calls) == 1 and type(value) is int and value in (0,1), 'Space output contract')
                require(diagnostics['context_indices'] == [r['index'] for r in item['context']]
                        and diagnostics['context_frames'] == [r['frame'] for r in item['context']],
                        'Actual helper context differs from pre-frozen selected PNGs')
                baseline = dict(row['trace']['candidate'])
                prediction = {**baseline, 'evasion_space': value}
                require(all(prediction[k] == baseline[k] for k in ('collision_frame','entry_frame','entry_side')), 'Copied fields changed')
                result = dict(ID=ID, baseline=baseline, candidate=prediction, diagnostics=diagnostics,
                    selected_context=item['context'], calls=recorder.calls, seconds=time.perf_counter()-start,
                    source_sha256=record['input']['source_sha256'], frame_pts=record['input']['source_frame_pts'],
                    other_three_fields_copied=True, end_to_end_equivalence_verified=False,
                    peak_allocated_bytes=model.torch.cuda.max_memory_allocated(), peak_reserved_bytes=model.torch.cuda.max_memory_reserved())
                put(folder/'result.json', result); report['videos'].append(result)
                print(json.dumps(dict(ID=ID,status='complete')), flush=True)
        require(report['call_count'] == 9 and report['network_attempts'] == 0, 'Call/network contract')
        require(all(sha(path) == digest for path, digest in bindings.items()), 'Bindings changed during execution')
        report.update(status='complete', bindings_unchanged=True)
    except BaseException as error:
        report.update(status='failed', error=type(error).__name__, error_message=str(error))
        raise
    finally:
        put(output/'report.json', report)

if __name__ == '__main__':
    main()
