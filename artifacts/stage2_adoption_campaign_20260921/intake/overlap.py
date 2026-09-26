"""CPU source/incident-overlap screening, reusing existing image signatures; no predictions or GT."""
import ast
import hashlib
import json
from pathlib import Path
import zlib
import cv2
import numpy as np
from PIL import Image, ImageDraw

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
OLD = ROOT / 'artifacts/stage2_goal_20260920'
EVIDENCE = HERE / 'overlap_evidence'
EVIDENCE.mkdir(exist_ok=False)
prior_script = OLD / 'ccd_public_overlap.py'
nodes = [n for n in ast.parse(prior_script.read_text()).body
         if isinstance(n, ast.FunctionDef) and n.name in {'rel', 'sha', 'signature', 'distance', 'descriptor'}]
exec(compile(ast.Module(body=nodes, type_ignores=[]), str(prior_script), 'exec'))
read = lambda p: json.loads(p.read_text())
plan = read(HERE / 'selection.json')
current = read(HERE / 'inputs.json')
past = read(OLD / 'ccd_intake/inputs.json') + read(ROOT / 'artifacts/stage2_first_frame_policy_20260920/intake/inputs.json')
assert len(past) == 24
# Independently reconstruct the deterministic metadata-only cohort.
prior_groups = {c['source_group'] for c in past}
excluded = prior_groups | {'CCD_YT_0000', 'CCD_YT_0010'}
assert set(plan['excluded_source_groups']) == excluded and len(excluded) == 26
meta = []
for line in (HERE / 'annotation.txt').read_text().splitlines():
    vid, rest = line.split(',', 1)
    labels, start, group, day, weather, ego = rest.rsplit(',', 5)
    if ego == 'Yes' and 'CCD_YT_' + group not in excluded:
        meta.append(('CCD_' + vid, 'CCD_YT_' + group))
meta.sort(key=lambda r: (sha(r[0].encode('utf-8')), r[0]))
selected, groups = [], set()
for sid, group in meta:
    if group not in groups:
        selected.append((sid, group)); groups.add(group)
    if len(selected) == 12:
        break
assert selected == [(c['ID'], c['source_group']) for c in current] == [(c['ID'], c['source_group']) for c in plan['selected']]
assert len({c['source_sha256'] for c in current}) == 12
for c in current:
    assert sha((ROOT / c['source_video']).read_bytes()) == c['source_sha256']
    assert sha((ROOT / c['pts_source']).read_bytes()) == c['pts_sha256']
    assert len(c['images']) == 50
    assert [f['frame'] for f in c['images']] == list(range(50))
    assert all(abs(f['pts_seconds'] - i / 10) < 1e-12 for i, f in enumerate(c['images']))
    for f in c['images']:
        assert sha((ROOT / f['path']).read_bytes()) == f['sha256']
        im = Image.open(ROOT / f['path']).convert('RGB')
        assert im.size == (f['width'], f['height']) and sha(im.tobytes()) == f['rgb_sha256']
public_audit = read(OLD / 'ccd_public_overlap.json')
members = read(HERE / 'zip_members.json')
public_paths = sorted((ROOT / 'Baseline/data/stage2/videos').glob('*.mp4'))
nexar_paths = [ROOT / p for p in public_audit['nexar_reference_paths']]
assert len(public_paths) == 5 and len(nexar_paths) == len(set(nexar_paths)) == 64
prior_videos = public_paths + nexar_paths + [ROOT / x['source_video'] for x in past]
video_hashes = {}
for path in prior_videos:
    video_hashes.setdefault(sha(path.read_bytes()), []).append(rel(path))
duplicates = [dict(case=c['ID'], paths=video_hashes[c['source_sha256']]) for c in current if c['source_sha256'] in video_hashes]
public_identity = []
for path in public_paths:
    data = path.read_bytes()
    matches = [m['name'] for m in members if m['size'] == len(data) and m['crc'] == zlib.crc32(data)]
    assert len(matches) == 1
    public_identity.append(dict(path=rel(path), member=matches[0], sha256=sha(data)))
assert not {c['source_group'] for c in current} & set(plan['excluded_source_groups'])

new, reference = [], []
for destination, cases, origin in ((new, current, 'new_ccd'), (reference, past, 'prior_ccd')):
    for case in cases:
        for f in case['images']:
            image = np.asarray(Image.open(ROOT / f['path']).convert('RGB'))
            destination.append(signature(image, case['ID'], origin, f['frame'], f['path']))
for path in public_paths + nexar_paths:
    cap = cv2.VideoCapture(str(path))
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    public = path in public_paths
    positions = list(range(count)) if public else sorted(set(np.linspace(0, count - 1, 12).round().astype(int).tolist()))
    for frame in positions:
        if not public:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
        ok, bgr = cap.read()
        assert ok, (path, frame)
        reference.append(signature(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), path.stem,
                                   'public' if public else 'nexar', frame, rel(path)))
    cap.release()
image_folders = sorted((ROOT / 'research/v5_external/dada/inputs/images').iterdir())
image_folders.append(ROOT / 'research/v6_stage2/mmau_probe/cap_8_009509/images')
image_sources = []
for folder in image_folders:
    files = sorted(p for p in folder.iterdir() if p.suffix.lower() in ('.jpg', '.jpeg', '.png'))
    positions = sorted(set(np.linspace(0, len(files) - 1, 12).round().astype(int).tolist()))
    origin = 'dada' if folder.name.startswith('DADA_') else 'mmau'
    sid = folder.name if origin == 'dada' else 'MMAU_CAP_8_009509'
    sampled = []
    for frame in positions:
        path = files[frame]
        reference.append(signature(np.asarray(Image.open(path).convert('RGB')), sid, origin, frame, rel(path)))
        sampled.append(dict(path=rel(path), sorted_position=frame, sha256=sha(path.read_bytes())))
    image_sources.append(dict(case=sid, origin=origin, available_images=len(files), sampled_images=sampled))
print(json.dumps(dict(new_frames=len(new), reference_frames=len(reference), reference_by_origin={o: sum(r['origin'] == o for r in reference) for o in ('public', 'nexar', 'prior_ccd', 'dada', 'mmau')})), flush=True)
exact = []
hash_index = {}
for r in reference:
    hash_index.setdefault((r['width'], r['height'], r['rgb_sha256']), []).append(r)
for r in new:
    for other in hash_index.get((r['width'], r['height'], r['rgb_sha256']), []):
        exact.append(dict(left=descriptor(r), right=descriptor(other)))


def render(left, right, name, d, variant):
    path = EVIDENCE / (name + '.jpg')
    canvas = Image.new('RGB', (768, 252), 'white')
    canvas.paste(Image.fromarray(left['thumb']), (0, 36))
    canvas.paste(Image.fromarray(right['thumb']), (384, 36))
    draw = ImageDraw.Draw(canvas)
    draw.text((4, 4), f"{left['case']} f{left['frame']} | {right['origin']} {right['case']} f{right['frame']}", fill='black')
    draw.text((4, 19), f'd={d}/63 {variant}; zero-based frames', fill='black')
    canvas.save(path, quality=92)
    return rel(path)


nearest = []
for case in current:
    frames = [r for r in new if r['case'] == case['ID']]
    for origin in ('public', 'nexar', 'prior_ccd', 'dada', 'mmau'):
        refs = [r for r in reference if r['origin'] == origin]
        best = None
        for left in frames:
            for right in refs:
                d, variant = distance(left, right)
                if best is None or d < best[0]:
                    best = (d, variant, left, right)
        d, variant, left, right = best
        nearest.append(dict(case=case['ID'], origin=origin, distance=d, variant=variant,
                            left=descriptor(left), right=descriptor(right), image=render(left, right, case['ID'] + '_' + origin, d, variant)))
within = []
for i, case in enumerate(current):
    lefts = [r for r in new if r['case'] == case['ID']]
    for other in current[i + 1:]:
        rights = [r for r in new if r['case'] == other['ID']]
        best = min((distance(a, b)[0], a['frame'], b['frame']) for a in lefts for b in rights)
        within.append(dict(left=case['ID'], right=other['ID'], distance=best[0], left_frame=best[1], right_frame=best[2]))
within = sorted(within, key=lambda r: (r['distance'], r['left'], r['right']))
within_review = []
for i, pair in enumerate(within[:6]):
    left = next(r for r in new if r['case'] == pair['left'] and r['frame'] == pair['left_frame'])
    right = next(r for r in new if r['case'] == pair['right'] and r['frame'] == pair['right_frame'])
    d, variant = distance(left, right)
    within_review.append(dict(**pair, image=render(left, right, f'within_{i:02d}', d, variant)))
for i in range(0, len(nearest), 6):
    canvas = Image.new('RGB', (768, 1512), 'white')
    for slot, pair in enumerate(nearest[i:i + 6]):
        canvas.paste(Image.open(ROOT / pair['image']), (0, slot * 252))
    canvas.save(EVIDENCE / f'nearest_{i // 6:02d}.jpg', quality=92)
canvas = Image.new('RGB', (768, 1512), 'white')
for slot, pair in enumerate(within_review):
    canvas.paste(Image.open(ROOT / pair['image']), (0, slot * 252))
canvas.save(EVIDENCE / 'within_top6.jpg', quality=92)
result = dict(status='computed_visual_review_pending', model_calls=0, model_predictions_read=False,
              scope=dict(new_ccd_cases=12, new_ccd_frames=len(new), reference_frames=len(reference), prior_videos=len(prior_videos),
                         reference_by_origin={o: sum(r['origin'] == o for r in reference) for o in ('public', 'nexar', 'prior_ccd', 'dada', 'mmau')}),
              public_identity=public_identity, excluded_groups=plan['excluded_source_groups'], source_group_overlap=[],
              whole_video_sha256_duplicates=duplicates, sampled_exact_rgb_matches=exact,
              nearest_per_case_and_origin=nearest, within_new_cases=within, within_top6=within_review,
              reference_mp4_paths=[rel(p) for p in prior_videos], prior_image_sources=image_sources,
              method='Reused63bit DCT median pHash: full, horizontal flip, central80percent crop. All600new/all1200priorCCD/all250public frames; uniform12 per64Nexar and16DADA+1MMAU.',
              selection_reconstruction=dict(passed=True, current_ids=[c['ID'] for c in current], distinct_new_groups=12, excluded_public_and_prior_groups=26, prior_ccd_cases=24, source_png_rgb_pts_manifest_hashes_checked=True),
              binding=dict(selection_sha256=sha((HERE / 'selection.json').read_bytes()), prior_signature_script_sha256=sha(prior_script.read_bytes()),
                           script_sha256=sha(Path(__file__).read_bytes())),
              limitation='A sampled fixed-transform nearest-frame screen cannot certify complete source/incident independence or rule out edited cross-compilation copies.')
(HERE / 'overlap.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
print(json.dumps(dict(status=result['status'], exact_rgb=len(exact), whole_video_duplicates=len(duplicates), nearest_pairs=len(nearest), min_distance=min(x['distance'] for x in nearest), within_min=within[0]['distance'])), flush=True)
