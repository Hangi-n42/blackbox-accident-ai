"""Real-file CPU tracking workload, synthetic repeated image; not accuracy."""
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import time
from unittest.mock import patch

for name in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS"):
    os.environ[name]="2"
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import cv2
import numpy as np
from PIL import Image
from solution import stage2_v5 as c

cv2.setNumThreads(2)
original_gray=c._gray
decoded=0
def gray(path):
    global decoded
    decoded+=1
    return original_gray(path)

output=ROOT/"research/v5_stage2/tracking_cpu_stress.json"
assert not output.exists()
rng=np.random.default_rng(416)
# Repeated textured image forces accepted tracks for all 300 adjacent steps.
texture=rng.integers(0,256,(180,320,3),dtype=np.uint8)
rows=[]
with tempfile.TemporaryDirectory(prefix="v5_tracking_",dir=ROOT/"research/v5_stage2") as temp:
    paths=[]
    for index in range(302):
        path=Path(temp)/f"frame_{index}.png"
        Image.fromarray(texture).save(path)
        paths.append(path)
    for count in (301,302):
        decoded=0
        start=time.perf_counter()
        with patch.object(c,"_gray",side_effect=gray):
            boxes,events=c._track_rois(paths[:count],150,[.2,.2,.6,.7])
        rows.append(dict(real_png_count=count,decode_calls=decoded,tracked_frames=len(boxes),
                         seconds=time.perf_counter()-start,events=events))
assert rows[0]["tracked_frames"]==301 and rows[0]["decode_calls"]==302
assert rows[1]["tracked_frames"]==1 and rows[1]["decode_calls"]==0
output.write_text(json.dumps(dict(cpu_threads=2,GPU_used=False,passed=True,
    synthetic_repeated_texture=True,not_representative_video_runtime=True,
    candidate_sha256=hashlib.sha256((ROOT/"solution/stage2_v5.py").read_bytes()).hexdigest(),rows=rows),indent=2),encoding="utf-8")
print(output.read_text())
