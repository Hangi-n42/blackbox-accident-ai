"""CPU-only official public contact corroboration; reused sources, never a holdout."""
import csv, hashlib, importlib.util, json, os, sys
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '2'
sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
PACKAGE = ROOT / 'artifacts/submissions/verify_v6'
sys.path.insert(0, str(PACKAGE / 'model/stage2/code'))
import av
import cv2
import numpy as np
from PIL import Image
from solution import stage2_uncapped_jerk_v6c as v6
cv2.setNumThreads(2)
spec = importlib.util.spec_from_file_location('solution.stage2_motion_init_v7', HERE / 'solution/stage2_motion_init_v7.py')
candidate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(candidate)

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))

def independent_scores(features):
    scaled = []
    for column in features.T:
        median = np.median(column)
        denominator = max(float(np.median(np.abs(column - median)) * 1.4826), .001)
        scaled.append(np.maximum((column - median) / denominator, 0))
    base = np.minimum(scaled[0], 10) + .6 * np.minimum(scaled[1], 10) + .25 * np.minimum(scaled[2], 10)
    new = scaled[0] + .6 * np.minimum(scaled[1], 10) + .25 * np.minimum(scaled[2], 10)
    base[0] = new[0] = 0
    fixed = new.copy()
    if len(fixed) > 1:
        fixed[1] = (.6 * np.minimum(scaled[1], 10) + .25 * np.minimum(scaled[2], 10))[1]
    return base, new, fixed

def main():
    assert sys.flags.isolated, 'Run with python -I'
    output = HERE / 'motion_public_audit.json'
    assert not output.exists(), 'Never overwrite an audit'
    labels_path = ROOT / 'Baseline/data/stage2/labels.csv'
    manifest_path = ROOT / 'research/v5_inputs/manifest.json'
    old_path = ROOT / 'research/v5_stage2/paired_layout_run/report.json'
    package_manifest_path = ROOT / 'artifacts/submissions/submit_v6.manifest.json'
    protocol_path = HERE / 'motion_initialization_protocol.json'
    dev_freeze_path = HERE / 'motion_init_dev/freeze.json'
    inputs = [Path(__file__), labels_path, manifest_path, old_path, package_manifest_path,
              protocol_path, dev_freeze_path, HERE / 'solution/stage2_motion_init_v7.py',
              ROOT / 'research/v6_stage2/public_official_partial_gt.json']
    inputs += list((ROOT / 'Baseline').glob('*.ipynb'))
    bindings = {str(p): sha(p) for p in inputs}
    package_manifest = read(package_manifest_path)
    for item in package_manifest['files']:
        if item['path'].startswith('model/stage2/code/'):
            p = PACKAGE / item['path']
            assert sha(p) == item['sha256'], str(p)
            bindings[str(p)] = item['sha256']
    dev = read(dev_freeze_path)
    for p in [protocol_path, HERE / 'solution/stage2_motion_init_v7.py']:
        assert sha(p) == dev['files'][str(p)]
    assert Path(v6.__file__).resolve().is_relative_to(PACKAGE)
    manifest = read(manifest_path)
    assert sha(labels_path) == manifest['source_labels_sha256']
    labels = {r['ID']: r for r in csv.DictReader(labels_path.open(encoding='utf-8-sig'))}
    archived = read(old_path)
    archived_rows = {r['ID']: r for r in archived['sets']['canonical']['V3']}
    results = []
    for item in manifest['videos']:
        ID = item['ID']
        source = ROOT / item['source_path']
        assert source.resolve() == (labels_path.parent / labels[ID]['path']).resolve()
        assert sha(source) == item['source_sha256']
        bindings[str(source)] = item['source_sha256']
        pts_path = ROOT / f'research/v6_stage2/public_pts/{ID}.json'
        pts_old = read(pts_path)
        bindings[str(pts_path)] = sha(pts_path)
        assert pts_old['source_video_sha256'] == sha(source)
        paths, times, native = [], [], []
        with av.open(str(source)) as container:
            stream = container.streams.video[0]
            stream.codec_context.thread_count = 2
            metadata = dict(average_rate=str(stream.average_rate), time_base=str(stream.time_base),
                            declared_frames=stream.frames, stream_duration=stream.duration)
            for index, frame in enumerate(container.decode(stream)):
                rec = item['frames'][index]
                assert rec['number'] == index
                image = rec['versions']['canonical']
                path = ROOT / image['path']
                assert sha(path) == image['file_sha256']
                bindings[str(path)] = image['file_sha256']
                rgb = frame.to_ndarray(format='rgb24')
                with Image.open(path) as png:
                    assert np.array_equal(rgb, np.asarray(png.convert('RGB'))), f'{ID}:{index}:source/PNG pixels'
                assert frame.pts is not None
                timestamp = Fraction(frame.pts) * frame.time_base
                old = pts_old['native_pts'][index]
                assert frame.pts == old['native_pts']
                assert timestamp == Fraction(old['native_pts'] * old['time_base_numerator'], old['time_base_denominator'])
                paths.append(path); times.append(timestamp)
                native.append(dict(frame=index, pts=frame.pts, time_base=str(frame.time_base), seconds=float(timestamp)))
        assert len(paths) == len(item['frames']) == len(pts_old['native_pts']) == item['frame_count']
        assert all(b > a for a, b in zip(times, times[1:]))
        target = int(labels[ID]['t_collision'])
        assert 0 <= target < len(paths)
        valid, base, new, features = v6._dual_motion_scan(paths)
        check_paths, check_base, _ = v6.primitives._motion_scan(paths)
        assert valid == paths == check_paths
        assert base.tobytes() == check_base.tobytes()
        assert base.tobytes() == np.asarray(archived_rows[ID]['motion_scores'], dtype=np.float32).tobytes()
        assert archived_rows[ID]['frame_numbers'] == list(range(len(paths)))
        fixed = candidate.corrected_scores(features, new)
        expected = independent_scores(features)
        assert all(a.tobytes() == b.tobytes() for a, b in zip((base, new, fixed), expected))
        assert np.array_equal(new[2:], fixed[2:]) and new[0] == fixed[0] == 0
        numbers = [v6.primitives._frame_number(p) for p in paths]
        before, after = [numbers[int(np.argmax(s))] for s in (new, fixed)]
        errors = [abs(times[n] - times[target]) for n in (before, after)]
        results.append(dict(ID=ID, source_sha256=item['source_sha256'], metadata=metadata,
            frame_count=len(paths), native_frame_pts=native, label_t_collision=target,
            label_unit='zero_based_provided_video_decoded_frame_index', label_seconds=float(times[target]),
            V6_collision_frame=before, corrected_collision_frame=after,
            V6_error_seconds=float(errors[0]), corrected_error_seconds=float(errors[1]),
            V6_within_0_3_seconds=errors[0] <= Fraction(3, 10), corrected_within_0_3_seconds=errors[1] <= Fraction(3, 10),
            first_pair_score_before=float(new[1]), first_pair_score_after=float(fixed[1]),
            base_scores=base.tolist(), V6_scores=new.tolist(), corrected_scores=fixed.tolist(), features=features.tolist(),
            verification=dict(original_PNG_pixels_exact=True, native_PTS_redecoded_exact=True,
                base_primitive_exact=True, archived_V3_canonical_base_scores_exact=True,
                independent_V6_and_corrected_scores_exact=True)))
        print(ID, target, before, after, flush=True)
    assert all(sha(p) == h for p, h in bindings.items())
    report = dict(created_utc=datetime.now(timezone.utc).isoformat(), status='complete', model_calls=0,
        GPU_used=False, source_role='previously_exposed_public5_development_not_independent_holdout',
        scope='Provided official contact label corroboration only; regenerated frozen V6 motion features, not archived V6 model predictions',
        failed_human9_advancement_gate_unchanged=True, adoption_allowed=False, official_S2=None,
        complete_four_field_GT=False, bindings_sha256=bindings, bindings_unchanged=True, videos=results,
        summary=dict(n=len(results), V6_correct=sum(r['V6_within_0_3_seconds'] for r in results),
            corrected_correct=sum(r['corrected_within_0_3_seconds'] for r in results),
            V6_MAE_seconds=sum(r['V6_error_seconds'] for r in results)/len(results),
            corrected_MAE_seconds=sum(r['corrected_error_seconds'] for r in results)/len(results)))
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(report['summary']))

if __name__ == '__main__':
    main()
