"""Exercise the DataFrame entry point with frozen features and replayed VLM answers."""
from contextlib import nullcontext
import importlib.util
import tempfile
from unittest.mock import patch

from check_runtime import ROOT, HERE, Replay, candidate, np, read, save, sha


def main():
    out = HERE / 'entrypoint_replay'
    out.mkdir(exist_ok=False)
    frozen = read(HERE / 'freeze.json')
    for path, digest in frozen['files'].items():
        assert sha(ROOT / path) == digest
    manifest = read(ROOT / frozen['core_dir'] / 'evaluation_manifest.json')['cases']
    cases = {case['ID']: case for case in manifest}
    expected = {case['ID']: case for case in read(ROOT / frozen['ablation_result'])['cases']}
    replay, seen = Replay([]), []

    def cached_scan(paths):
        if seen:
            assert replay.index == 4
        case_id = paths[0].parent.name
        case = cases[case_id]
        assert [path.resolve() for path in paths] == [(ROOT / image['path']).resolve() for image in case['images']]
        source = ROOT / case['prediction_source']
        record = read(source)
        replay.calls, replay.index = record['calls'], 0
        motion = np.load(source.parent / 'motion.npz')
        seen.append(case_id)
        return paths, motion['base_scores'], motion['new_scores'], motion['features']

    with tempfile.TemporaryDirectory(dir=out) as temporary:
        from pathlib import Path
        data = Path(temporary)
        for case in manifest:
            folder = data / 'images' / case['ID']
            folder.mkdir(parents=True)
            for image in case['images']:
                source = ROOT / image['path']
                (folder / source.name).symlink_to(source)
        with patch.object(candidate.reference, '_dual_motion_scan', cached_scan), \
             patch.object(candidate.reference.baseline, 'CandidateVLM', return_value=nullcontext(replay)):
            result = candidate.predict_stage2(data, ROOT / 'artifacts/submissions/verify_v7/model/stage2', trace_dir=out / 'traces')
    assert replay.index == 4 and seen == sorted(cases)
    assert list(result.columns) == candidate.COLUMNS and len(result) == len(cases) and result.ID.is_unique
    for row in result.to_dict('records'):
        case_id = row.pop('ID')
        assert row == expected[case_id]['candidate']
        trace = read(out / 'traces' / (case_id + '.json'))
        assert trace['diagnostics']['baseline_prediction'] == expected[case_id]['baseline']
    spec = importlib.util.spec_from_file_location('probe', HERE.parent / 'server_audit/stage2_probe.py')
    probe = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(probe)
    probe.validate_output(result, {c['ID']: {image['frame'] for image in c['images']} for c in manifest})
    result.to_csv(out / 'stage2.csv', index=False)
    report = dict(status='PASS_INTERFACE_REPLAY_ONLY', cases=len(cases), replayed_calls=4 * len(cases),
                  actual_model_calls=0, optical_flow_recomputed=False, cuda_execution=False,
                  same_candidate_values=True, output_contract_passed=True, trace_baseline_values_equal=True,
                  test_script_sha256=sha(HERE / 'check_entrypoint.py'))
    save(out / 'report.json', report)
    print(report)


if __name__ == '__main__':
    main()
