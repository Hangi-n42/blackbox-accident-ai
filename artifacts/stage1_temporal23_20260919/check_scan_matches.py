import sys,json
from pathlib import Path
import cv2
sys.path.insert(0,'scripts/data')
from prepare_stage1_road_pairs import match
ROOT=Path.cwd();OUT=ROOT/'artifacts/stage1_temporal23_20260919';OLD=ROOT/'artifacts/stage1_road_pairs_20260918'
cv2.setNumThreads(2)
paired=[]
for p in sorted((OUT/'scan/reds').glob('*.png')):
 a=cv2.imread(str(p));b=cv2.imread(str(OUT/'scan/vd'/p.with_suffix('.jpg').name));h,m=match(a,b);paired.append({'source':p.stem,**m})
(OUT/'scan_correspondence.json').write_text(json.dumps(paired,indent=2))
print('pairs',len(paired),'min_inliers',min(r['inliers'] for r in paired))
