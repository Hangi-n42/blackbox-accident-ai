"""CPU-only source identity audit; never reads model outputs or changes labels."""
from pathlib import Path
import hashlib
import json
import zlib
import cv2
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
INTAKE = OUT / 'ccd_intake'
EVIDENCE = OUT / 'ccd_overlap_evidence'
EVIDENCE.mkdir(exist_ok=True)

def rel(p):
    return str(p.relative_to(ROOT))

def sha(b):
    return hashlib.sha256(b).hexdigest()

def signature(rgb, case, origin, frame, path):
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    def phash(a):
        d = cv2.dct(cv2.resize(a, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32))[:8, :8].flatten()[1:]
        return sum(int(v > np.median(d)) << i for i, v in enumerate(d))
    h, w = gray.shape
    variants = [phash(gray), phash(gray[:, ::-1]), phash(gray[h//10:h-h//10, w//10:w-w//10])]
    thumb = cv2.resize(rgb, (384, 216), interpolation=cv2.INTER_AREA)
    return dict(case=case, origin=origin, frame=frame, path=path, width=w, height=h, rgb_sha256=sha(rgb.tobytes()), hashes=variants, thumb=thumb)

selection = json.loads((INTAKE / 'selection.json').read_text())['selected']
members = json.loads((INTAKE / 'zip_members.json').read_text())
annotations = {}
for line in (INTAKE / 'annotation.txt').read_text().splitlines():
    vid, rest = line.split(',', 1)
    _, start, group, day, weather, ego = rest.rsplit(',', 5)
    annotations[vid] = dict(source_group='CCD_YT_' + group, original_startframe=start, ego=ego)

public_identity = []
public_paths = sorted((ROOT / 'Baseline/data/stage2/videos').glob('*.mp4'))
for p in public_paths:
    data = p.read_bytes()
    matched = [m for m in members if m['size'] == len(data) and m['crc'] == zlib.crc32(data)]
    public_identity.append(dict(path=rel(p), bytes=len(data), crc32=zlib.crc32(data), sha256=sha(data),
        matches=[dict(zip_member=m['name'], **annotations[Path(m['name']).stem]) for m in matched]))
excluded_groups = sorted({m['source_group'] for p in public_identity for m in p['matches']})
print('public_source_groups', excluded_groups, flush=True)

old = json.loads((ROOT / 'artifacts/stage2_goal_20260919/acquisition/prior_byte_inventory.json').read_text())
nexar_paths = [ROOT / x['path'] for x in old if any(s in x['path'] for s in ['/nexar/', '/nexar_validation/', '/nexar_review_candidates/', '/nexar_review_round2/', '/fresh_sources_retry1/', '/new_validation_sources/'])]
nexar_paths += sorted((ROOT / 'artifacts/stage2_goal_20260919/acquisition').glob('*.mp4'))
nexar_paths = sorted(set(nexar_paths))
whole_video_duplicates = []
prior_hashes = {}
for p in public_paths + nexar_paths:
    prior_hashes.setdefault(sha(p.read_bytes()), []).append(rel(p))
ccd = []
for c in selection:
    manifest = json.loads((INTAKE / c['ID'] / 'input_manifest.json').read_text())
    if manifest['source_sha256'] in prior_hashes:
        whole_video_duplicates.append(dict(case=c['ID'], matches=prior_hashes[manifest['source_sha256']]))
    for f in manifest['images']:
        rgb = np.array(Image.open(ROOT / f['path']).convert('RGB'))
        ccd.append(signature(rgb, c['ID'], 'ccd', f['frame'], f['path']))
reference = []
for p in public_paths + nexar_paths:
    cap = cv2.VideoCapture(str(p))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    is_public = p in public_paths
    indexes = list(range(n)) if is_public else sorted(set(np.linspace(0, n-1, 12).round().astype(int).tolist()))
    for i in indexes:
        if not is_public:
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ok, bgr = cap.read()
        if not ok:
            raise RuntimeError(f'Cannot decode {p} at {i}')
        reference.append(signature(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), ('public_' if is_public else 'nexar_') + p.stem,
                                   'public' if is_public else 'nexar', i, rel(p)))
    cap.release()
print('frames', len(ccd), len(reference), 'nexar_videos', len(nexar_paths), flush=True)

def distance(a, b):
    values = [(a['hashes'][i] ^ b['hashes'][j]).bit_count() for i, j in [(0, 0), (1, 0), (2, 0), (0, 2), (2, 2)]]
    k = min(range(len(values)), key=values.__getitem__)
    return values[k], ['full/full', 'mirror/full', 'crop80/full', 'full/crop80', 'crop80/crop80'][k]

def descriptor(r):
    return {k:v for k,v in r.items() if k not in ('thumb', 'hashes')}

exact_rgb = []
nearest = []
for c in selection:
    a = [x for x in ccd if x['case'] == c['ID']]
    for origin in ['public', 'nexar']:
        best = None
        for x in a:
            for y in reference:
                if y['origin'] != origin:
                    continue
                if x['width'] == y['width'] and x['height'] == y['height'] and x['rgb_sha256'] == y['rgb_sha256']:
                    exact_rgb.append(dict(left=descriptor(x), right=descriptor(y)))
                d, variant = distance(x, y)
                if best is None or d < best[0]:
                    best = (d, variant, x, y)
        d, variant, x, y = best
        pair_path = EVIDENCE / f'{c["ID"]}_{origin}.jpg'
        canvas = Image.new('RGB', (768, 252), 'white')
        canvas.paste(Image.fromarray(x['thumb']), (0, 36)); canvas.paste(Image.fromarray(y['thumb']), (384, 36))
        draw = ImageDraw.Draw(canvas)
        draw.text((4, 4), f'{x["case"]} frame {x["frame"]} | {y["case"]} frame {y["frame"]}', fill='black')
        draw.text((4, 19), f'd={d}/63 {variant}; all frame indexes zero based', fill='black')
        canvas.save(pair_path, quality=90)
        nearest.append(dict(case=c['ID'], origin=origin, distance=d, variant=variant,
                            left=descriptor(x), right=descriptor(y), image=rel(pair_path)))
for i in range(0, len(nearest), 6):
    sheet = Image.new('RGB', (768, 252 * 6), 'white')
    for j, p in enumerate(nearest[i:i+6]):
        sheet.paste(Image.open(ROOT / p['image']), (0, j*252))
    sheet.save(EVIDENCE / f'nearest_{i//6:02d}.jpg', quality=90)

within_ccd = []
for i, c in enumerate(selection):
    a = [x for x in ccd if x['case'] == c['ID']]
    for other in selection[i+1:]:
        b = [x for x in ccd if x['case'] == other['ID']]
        best = min((distance(x, y)[0], x['frame'], y['frame']) for x in a for y in b)
        within_ccd.append(dict(left=c['ID'], right=other['ID'], distance=best[0], left_frame=best[1], right_frame=best[2]))

result = dict(status='computed_visual_review_pending', scope=dict(ccd_videos=len(selection), ccd_frames=len(ccd), public_videos=len(public_paths), public_frames=sum(x['origin']=='public' for x in reference), nexar_videos=len(nexar_paths), nexar_frames=sum(x['origin']=='nexar' for x in reference)),
    public_zip_identity=public_identity, public_source_groups=excluded_groups,
    selected_source_group_collisions=[c['ID'] for c in selection if c['source_group'] in excluded_groups],
    selected_pairwise_source_group_unique=len({c['source_group'] for c in selection}) == len(selection),
    whole_video_sha256_duplicates=whole_video_duplicates, sampled_exact_rgb_matches=exact_rgb,
    nearest_per_case_and_reference_origin=nearest, within_ccd_nearest=within_ccd,
    nexar_reference_paths=[rel(p) for p in nexar_paths],
    method='63 AC DCT coefficients; median threshold; full, horizontal mirror, central80pct crop comparisons. 12 uniform frames per Nexar; all public/CCD frames. All-zero-based indexes.',
    limitation='Size+CRC32+unique member supports public source mapping but is not cryptographic official-video comparison. pHash is a screening feature, not duplicate proof. No temporal alignment or full Nexar coverage; cross-compilation source identity not certified. No predictions read or model execution.')
(OUT / 'ccd_public_overlap.json').write_text(json.dumps(result, ensure_ascii=False, indent=2))
print(json.dumps(dict(scope=result['scope'], whole_video_duplicates=len(whole_video_duplicates), exact_rgb=len(exact_rgb), nearest_min=min(x['distance'] for x in nearest), within_ccd_min=min(x['distance'] for x in within_ccd)), ensure_ascii=False), flush=True)
