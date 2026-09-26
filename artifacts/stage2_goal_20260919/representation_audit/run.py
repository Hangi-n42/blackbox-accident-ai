"""Paired CPU-only representation audit; never overwrite evidence or run a VLM."""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '2'
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
import argparse
import hashlib
import importlib.metadata
import json
import platform
import sys
import time
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
CODE = ROOT / 'releases/v7/source/model/stage2/code'
sys.path.insert(0, str(CODE))
import numpy as np
from PIL import Image
from solution import stage2_uncapped_jerk_v6c as policy
policy.primitives.cv2.setNumThreads(2)


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def save(path, value):
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)


def relative(path):
    return str(path.resolve().relative_to(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=HERE)
    out = parser.parse_args().output_dir.resolve()
    if not out.is_relative_to(HERE):
        raise ValueError('Output must remain inside the owned representation_audit folder')
    if any((out / name).exists() for name in ('manifest.json', 'results.json', 'REPORT.md')):
        raise FileExistsError('Preserve existing evidence; use a new subdirectory for reproduction')
    out.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    manifest_path = ROOT / 'research/v5_inputs/manifest.json'
    public_path = ROOT / 'artifacts/pipeline_diagnosis_20260917/targeted_data/public_contact_manifest.json'
    historical_path = ROOT / 'research/v7/motion_public_audit.json'
    old = read(manifest_path)
    public = {v['id']: v for v in read(public_path)}
    historical = {v['ID']: v for v in read(historical_path)['videos']}
    bindings = {}

    def bind(path):
        digest = sha(path)
        bindings[relative(path)] = digest
        return digest

    for path in (Path(__file__), manifest_path, public_path, historical_path,
                 ROOT / 'Baseline/data/stage2/labels.csv',
                 ROOT / 'scripts/prepare_public_eval.py',
                 ROOT / 'scripts/prepare_v5_diagnostics.py'):
        bind(path)
    # Include all local policy imports, even though no VLM class is instantiated.
    for name in ('stage2_uncapped_jerk_v6c.py', 'stage2_motion_collision.py',
                 'stage2_v2.py', 'stage2.py', 'vlm_candidate.py', 'vlm.py'):
        bind(CODE / 'solution' / name)
    cases = []
    for video in old['videos']:
        identity = video['ID']
        source = ROOT / video['source_path'].replace('\\', '/')
        assert bind(source) == video['source_sha256']
        pts_path = ROOT / f'research/v6_stage2/public_pts/{identity}.json'
        pts = read(pts_path)
        bind(pts_path)
        assert pts['source_video_sha256'] == video['source_sha256']
        times = {r['frame']: Fraction(r['native_pts'] * r['time_base_numerator'],
                                    r['time_base_denominator']) for r in pts['native_pts']}
        current = public[identity]
        pngs, jpegs, records = [], [], []
        for frame, current_image in zip(video['frames'], current['input_images'], strict=True):
            number = frame['number']
            assert number == current_image['frame']
            png_record = frame['versions']['canonical']
            png = ROOT / png_record['path'].replace('\\', '/')
            jpeg = ROOT / current_image['path']
            png_hash, jpeg_hash = bind(png), bind(jpeg)
            assert png_hash == png_record['file_sha256']
            assert jpeg_hash == current_image['sha256']
            assert policy.primitives._frame_number(png) == policy.primitives._frame_number(jpeg) == number
            pngs.append(png); jpegs.append(jpeg)
            records.append(dict(frame=number, pts_seconds=float(times[number]),
                canonical_path=relative(png), canonical_sha256=png_hash,
                current_jpeg_path=relative(jpeg), current_jpeg_sha256=jpeg_hash,
                historical_jpeg_sha256=frame['versions']['existing_jpeg']['file_sha256'],
                current_matches_historical_jpeg=jpeg_hash == frame['versions']['existing_jpeg']['file_sha256']))
        cache_path = ROOT / f'artifacts/pipeline_diagnosis_20260917/stage2_public_metal/stage2/public/{identity}/motion.npz'
        result_path = cache_path.with_name('result.json')
        bind(cache_path); bind(result_path)
        cached_result = read(result_path)
        with np.load(cache_path, allow_pickle=False) as cache:
            for kind, key in [('base_scores', 'base_score_sha256'), ('new_scores', 'new_score_sha256')]:
                assert policy._score_hash(cache[kind]) == cached_result['diagnostics']['uncapped_jerk'][key]
        target = current['truth_frame']
        assert float(times[target]) == current['truth_seconds']
        cases.append(dict(ID=identity, source=relative(source), source_sha256=video['source_sha256'],
            truth_frame=target, truth_seconds=float(times[target]), pts_path=relative(pts_path),
            cache_path=relative(cache_path), frames=records))
    versions = {name: importlib.metadata.version(name) for name in ('numpy', 'pillow', 'pandas', 'opencv-python-headless')}
    manifest = dict(created_utc=datetime.now(timezone.utc).isoformat(),
        purpose='Frozen V6C CPU motion only: canonical PNG versus current public JPEG; exposed public5 development',
        protocol=dict(order='S2_001 to S2_005; canonical then current JPEG per video',
            model_calls=0, GPU_used=False, source_video_decode=False, original_frame_numbers=True,
            cpus_threads=2, function='solution.stage2_uncapped_jerk_v6c._dual_motion_scan',
            common_source=relative(Path(policy.__file__)), tolerance_seconds='3/10',
            frozen_before_motion_execution=True, expected_outputs='input differences and selection reproducibility; no candidate selection or tuning'),
        environment=dict(python=sys.version, executable=sys.executable, platform=platform.platform(),
            library_versions=versions, opencv_threads=policy.primitives.cv2.getNumThreads()),
        caveats=['Prior exploratory read-only S2_002 comparison was already observed before this audit.',
            'Current JPEG creation command and codec/library provenance are not fully established.',
            'Different encoded inputs do not isolate JPEG compression from source decoding differences.',
            'No private-evaluation inference, VLM, MLX, training, downloads, or production edits.'],
        bindings_sha256=bindings, cases=cases)
    save(out / 'manifest.json', manifest)
    rows = []
    for case in cases:
        identity = case['ID']
        times = {r['frame']: Fraction(r['native_pts'] * r['time_base_numerator'], r['time_base_denominator'])
                 for r in read(ROOT / case['pts_path'])['native_pts']}
        comparisons = []
        total_absolute, total_values, maximum = 0, 0, 0
        for frame in case['frames']:
            with Image.open(ROOT / frame['canonical_path']) as a, Image.open(ROOT / frame['current_jpeg_path']) as b:
                png, jpeg = np.asarray(a.convert('RGB')), np.asarray(b.convert('RGB'))
                assert png.shape == jpeg.shape
                difference = np.abs(png.astype(np.int16) - jpeg.astype(np.int16))
                summed = int(difference.sum(dtype=np.int64))
                peak = int(difference.max())
                total_absolute += summed; total_values += difference.size; maximum = max(maximum, peak)
                comparisons.append(dict(frame=frame['frame'], shape=list(png.shape),
                    rgb_mean_absolute_difference=summed / difference.size, rgb_max_absolute_difference=peak,
                    identical_pixels=bool(np.array_equal(png, jpeg))))
        row = dict(ID=identity, truth_frame=case['truth_frame'], truth_seconds=case['truth_seconds'],
            pixels=dict(mean_absolute_difference=total_absolute / total_values, max_absolute_difference=maximum,
                        identical_frames=sum(r['identical_pixels'] for r in comparisons), frames=comparisons), runs={})
        with np.load(ROOT / case['cache_path'], allow_pickle=False) as cache:
            for version, key in [('canonical', 'canonical_path'), ('current_jpeg', 'current_jpeg_path')]:
                paths = [ROOT / f[key] for f in case['frames']]
                tick = time.monotonic()
                valid, base, new, features = policy._dual_motion_scan(paths)
                elapsed = time.monotonic() - tick
                assert valid == paths and len(paths) == 50
                numbers = [policy.primitives._frame_number(path) for path in paths]
                winner = numbers[int(np.argmax(new))]
                error = abs(times[winner] - times[case['truth_frame']])
                reference = (np.asarray(historical[identity]['V6_scores'], dtype=np.float32)
                             if version == 'canonical' else cache['new_scores'])
                reference_features = (np.asarray(historical[identity]['features'], dtype=np.float32)
                                      if version == 'canonical' else cache['features'])
                row['runs'][version] = dict(winner_frame=winner, winner_seconds=float(times[winner]),
                    absolute_error_seconds=float(error), correct_at_0_3_seconds=error <= Fraction(3, 10),
                    elapsed_seconds=elapsed, original_frame_numbers=numbers,
                    base_scores=base.tolist(), new_scores=new.tolist(), features=features.tolist(),
                    base_score_sha256=policy._score_hash(base), new_score_sha256=policy._score_hash(new),
                    cached_reference=('research/v7/motion_public_audit.json' if version == 'canonical' else case['cache_path']),
                    cached_winner_frame=numbers[int(np.argmax(reference))],
                    score_bytes_equal=bool(new.tobytes() == reference.tobytes()),
                    score_max_absolute_difference=float(np.max(np.abs(new - reference))),
                    features_bytes_equal=bool(features.tobytes() == reference_features.tobytes()),
                    features_max_absolute_difference=float(np.max(np.abs(features - reference_features))))
        pair = row['runs']
        row['winner_changed'] = pair['canonical']['winner_frame'] != pair['current_jpeg']['winner_frame']
        compare_frames = sorted({pair[v]['winner_frame'] for v in pair})
        row['scores_at_both_winners'] = {v: {str(n): pair[v]['new_scores'][pair[v]['original_frame_numbers'].index(n)]
                                            for n in compare_frames} for v in pair}
        rows.append(row)
    assert all(sha(ROOT / path) == digest for path, digest in bindings.items()), 'Input/code changed during audit'
    summaries = {v: dict(n=len(rows), correct=sum(r['runs'][v]['correct_at_0_3_seconds'] for r in rows),
        mae_seconds=sum(r['runs'][v]['absolute_error_seconds'] for r in rows) / len(rows),
        predictions=[r['runs'][v]['winner_frame'] for r in rows],
        all_archived_winners_reproduced=all(r['runs'][v]['winner_frame'] == r['runs'][v]['cached_winner_frame'] for r in rows))
        for v in ('canonical', 'current_jpeg')}
    result = dict(status='complete', manifest_sha256=sha(out / 'manifest.json'), bindings_unchanged=True,
        model_calls=0, GPU_used=False, elapsed_seconds=time.monotonic()-started, videos=rows, summary=summaries,
        current_jpeg_matches_historical_count=sum(f['current_matches_historical_jpeg'] for c in cases for f in c['frames']),
        input_frames_per_representation=sum(len(c['frames']) for c in cases),
        changed_video_ids=[r['ID'] for r in rows if r['winner_changed']], official_S2=None)
    save(out / 'results.json', result)
    lines = ['# Stage2 공개 PNG/JPEG 움직임 선택 대조', '',
        '동일한 V7 배포 소스의 V6C CPU 함수에 canonical PNG와 현재 JPEG를 각각 입력했다. VLM/GPU/MLX 호출·학습·다운로드·운영 파일 수정은 없다.', '',
        '| 입력 | 최종 프레임 | 접촉 Acc@0.3초 | MAE |', '|---|---|---:|---:|']
    for version, s in summaries.items():
        lines.append(f"| {version} | {s['predictions']} | {s['correct']}/{s['n']} | {s['mae_seconds']:.6f}초 |")
    lines += ['', f"선택이 달라진 영상: {result['changed_video_ids']}. 기존 캐시 승자 재현 여부는 results.json의 항목별 기록에 보존했다. 점수 byte 동일성과 승자 동일성을 구분한다.", '',
        '## 해석과 한계', '',
        '- 입력 표현만 달리한 동일 CPU 실행에서 선택 차이가 재현되는지 확인하는 감사다. JPEG 압축만의 효과와 디코더·생성 경로 차이는 분리하지 않았다.',
        '- canonical PNG 250장은 과거 동결 해시, 현재 JPEG 250장은 9/17 입력 manifest 해시에 모두 일치했다. 현재 JPEG와 과거 v5 manifest의 existing_jpeg는 같은 경로여도 별도 입력이다.',
        f"- 현재 JPEG가 과거 existing_jpeg 해시와 같은 수: {result['current_jpeg_matches_historical_count']}/{result['input_frames_per_representation']}.",
        '- prepare_public_eval.py는 OpenCV 순차 디코딩과 기본 JPEG 인코딩 경로를 제공한다. 현재 JPEG를 생성한 실제 명령·시각·코덱 버전은 이번 감사로 확정하지 않았다.',
        '- PNG의 원본 RGB/native PTS 대응은 기존 motion_public_audit 기록을 근거로 한다. 이번에는 원본 영상 SHA를 확인했으며 원본 영상을 다시 디코딩하지 않았다.',
        '- 공개5개는 이미 노출된 개발자료이고 제공 접촉 부분 라벨만 평가했다. 다른 세 필드·공식 S2·비공개 일반화 개선을 주장하지 않는다.',
        '- 과거 캐시와 현재 연산 사이의 작은 수치 차이는 기록했다. 연산 라이브러리/플랫폼 원인을 추가로 분리하지 않았다.', '',
        '## 재현', '',
        '프로젝트 루트에서 다음 명령을 사용한다. 기존 결과는 덮어쓰지 않으며 새 하위 폴더를 지정한다.', '',
        '```sh', 'PYTHONDONTWRITEBYTECODE=1 artifacts/mac_experiments/scipy_compat/.venv/bin/python artifacts/stage2_goal_20260919/representation_audit/run.py --output-dir artifacts/stage2_goal_20260919/representation_audit/reproduction_01', '```', '',
        'manifest.json: 실행 전 입력·코드·환경 해시. results.json: 모든 점수·특징·픽셀 차이·캐시 대조. manifest 바인딩은 계산 완료 후 재확인했다.', '']
    with (out / 'REPORT.md').open('x', encoding='utf-8') as stream:
        stream.write('\n'.join(lines))
    print(json.dumps(dict(status='complete', output=str(out), summary=summaries,
                         changed_video_ids=result['changed_video_ids'], bindings_unchanged=True), ensure_ascii=False))


if __name__ == '__main__':
    main()
