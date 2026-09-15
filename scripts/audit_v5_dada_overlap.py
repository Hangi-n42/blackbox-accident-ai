"""Limited image similarity audit, not a proof of source independence."""
from pathlib import Path
import json, hashlib
import cv2
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'research/v5_external/dada'
cv2.setNumThreads(2)
def descriptor(path):
    b=np.fromfile(path,dtype=np.uint8)
    im=cv2.imdecode(b,cv2.IMREAD_COLOR)
    if im is None: raise ValueError(str(path))
    gray=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    dct=cv2.dct(cv2.resize(gray,(32,32)).astype(np.float32))[:8,:8].ravel()[1:]
    return dct>np.median(dct),hashlib.sha256(im.tobytes()).hexdigest()
pub=[]
for p in sorted((ROOT/'research/v5_inputs/canonical/images').glob('*/*.png')):
    h,s=descriptor(p);pub.append((str(p.relative_to(ROOT)),h,s))
rows=[]
for folder in sorted((BASE/'inputs/images').iterdir()):
    if not folder.is_dir(): continue
    paths=sorted(folder.glob('*.png'))
    selected=[paths[i] for i in sorted(set(np.linspace(0,len(paths)-1,24).round().astype(int)))]
    best=(64,None,None); exact=[]
    for p in selected:
        h,s=descriptor(p)
        for q,ph,ps in pub:
            d=int(np.count_nonzero(h!=ph))
            if d<best[0]: best=(d,p.name,q)
            if s==ps: exact.append([p.name,q])
    rows.append({'ID':folder.name,'sampled':len(selected),'closest_phash_hamming_63bit':best[0],
        'closest_external_frame':best[1],'closest_public_frame':best[2],'sampled_exact_rgb_matches':exact})
report={'scope':'24 uniformly spaced images per external clip versus all250 public PNG frames; no crop/mirror/temporal alignment search.',
        'limitation':'No exact match or large nearest distance does not prove no common camera/source or hidden duplicate.', 'clips':rows}
(BASE/'overlap_audit.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(report,ensure_ascii=False))
