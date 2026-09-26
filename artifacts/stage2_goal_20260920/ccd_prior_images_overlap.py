"""Extend frozen overlap screening to existing DADA/MM-AU images only."""
import ast
import hashlib
import json
from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
INTAKE = OUT / 'ccd_intake'
EVIDENCE = OUT / 'ccd_prior_overlap_evidence'
EVIDENCE.mkdir(exist_ok=True)

# Reuse the reviewed feature functions without executing the prior audit again.
prior_script = OUT / 'ccd_public_overlap.py'
tree = ast.parse(prior_script.read_text())
functions = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in {'rel', 'sha', 'signature', 'distance', 'descriptor'}]
exec(compile(ast.Module(body=functions, type_ignores=[]), str(prior_script), 'exec'))

selection = json.loads((INTAKE / 'selection.json').read_text())['selected']
ccd = []
for c in selection:
    m = json.loads((INTAKE / c['ID'] / 'input_manifest.json').read_text())
    for f in m['images']:
        rgb = np.array(Image.open(ROOT / f['path']).convert('RGB'))
        ccd.append(signature(rgb, c['ID'], 'ccd', f['frame'], f['path']))

reference = []
sources = []
dirs = sorted((ROOT / 'research/v5_external/dada/inputs/images').iterdir())
dirs.append(ROOT / 'research/v6_stage2/mmau_probe/cap_8_009509/images')
for folder in dirs:
    files = sorted([p for p in folder.iterdir() if p.suffix.lower() in {'.jpg', '.jpeg', '.png'}])
    indexes = sorted(set(np.linspace(0, len(files)-1, 12).round().astype(int).tolist()))
    is_dada = folder.name.startswith('DADA_')
    sid = folder.name if is_dada else 'MMAU_CAP_8_009509'
    samples = []
    for idx in indexes:
        p = files[idx]
        rgb = np.array(Image.open(p).convert('RGB'))
        reference.append(signature(rgb, sid, 'dada' if is_dada else 'mmau', idx, rel(p)))
        samples.append(dict(path=rel(p), zero_based_sorted_position=idx, native_filename=p.name, file_sha256=sha(p.read_bytes())))
    sources.append(dict(case=sid, directory=rel(folder), available_images=len(files), sampled_images=samples))

print('frames', len(ccd), len(reference), 'prior_clips', len(sources), flush=True)
exact_rgb = []
nearest = []
for c in selection:
    best = None
    for x in ccd:
        if x['case'] != c['ID']:
            continue
        for y in reference:
            if x['width'] == y['width'] and x['height'] == y['height'] and x['rgb_sha256'] == y['rgb_sha256']:
                exact_rgb.append(dict(left=descriptor(x), right=descriptor(y)))
            d, variant = distance(x, y)
            if best is None or d < best[0]:
                best = (d, variant, x, y)
    d, variant, x, y = best
    pair_path = EVIDENCE / f'{c["ID"]}.jpg'
    canvas = Image.new('RGB', (768, 252), 'white')
    canvas.paste(Image.fromarray(x['thumb']), (0, 36)); canvas.paste(Image.fromarray(y['thumb']), (384, 36))
    draw = ImageDraw.Draw(canvas)
    draw.text((4, 4), f'{x["case"]} idx {x["frame"]} | {y["case"]} {Path(y["path"]).name}', fill='black')
    draw.text((4, 19), f'd={d}/63 {variant}; CCD frame index is zero based', fill='black')
    canvas.save(pair_path, quality=90)
    nearest.append(dict(case=c['ID'], distance=d, variant=variant, left=descriptor(x), right=descriptor(y), image=rel(pair_path)))

for i in range(0, len(nearest), 6):
    sheet = Image.new('RGB', (768, 252*6), 'white')
    for j, pair in enumerate(nearest[i:i+6]):
        sheet.paste(Image.open(ROOT / pair['image']), (0, 252*j))
    sheet.save(EVIDENCE / f'nearest_{i//6:02d}.jpg', quality=90)

result = dict(status='computed_visual_review_pending',
    scope=dict(ccd_videos=len(selection), ccd_frames=len(ccd), prior_image_clips=len(sources), prior_sampled_images=len(reference), dada_clips=16, mmau_clips=1),
    sources=sources, nearest_per_ccd_case=nearest, sampled_exact_rgb_matches=exact_rgb,
    method='Same signature/distance as ccd_public_overlap.py: 63 AC DCT median hash with full, horizontal mirror, central80pct crop. All CCD600 frames versus12 uniformly spaced frames from each existing DADA16 and MM-AU1 sequence. Ordinal indexes zero based; native filenames retained.',
    binding=dict(previous_audit_script_sha256=sha(prior_script.read_bytes()), this_script_sha256=sha(Path(__file__).read_bytes()), selection_sha256=sha((INTAKE/'selection.json').read_bytes())),
    limitation='Frame subsampling and fixed transforms can miss source overlap. No native time reconstruction, temporal alignment, incident annotation, model run, prediction or GT reading. Dataset names alone do not establish independence.')
(OUT / 'ccd_prior_images_overlap.json').write_text(json.dumps(result, ensure_ascii=False, indent=2))
print(json.dumps(dict(exact_rgb=len(exact_rgb), nearest=[(x['case'],x['right']['case'],x['distance']) for x in nearest]), ensure_ascii=False), flush=True)
