"""Public-example motion ablation / image preparation, not a hidden-score estimate."""
from pathlib import Path
import json
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import cv2
import pandas as pd
from PIL import Image
from solution.stage2 import _motion_scan, _overview_indices, _contact_sheet, _frame_number

rows = []
labels = pd.read_csv(ROOT / 'Baseline/data/stage2/labels.csv')
for row in labels.itertuples():
    folder = ROOT / 'research/stage2_public/images' / row.ID
    folder.mkdir(parents=True, exist_ok=True)
    capture = cv2.VideoCapture(str(ROOT / 'Baseline/data/stage2' / row.path))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    index = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).save(folder / f'frame_{index:06d}.jpg')
        index += 1
    capture.release()
    start = time.perf_counter()
    paths, scores, side = _motion_scan(sorted(folder.glob('*.jpg')))
    prediction = _frame_number(paths[int(scores.argmax())])
    overview = _overview_indices(scores)
    _contact_sheet(paths, overview).save(ROOT / 'research/stage2_public' / f'{row.ID}_overview.jpg')
    rows.append(dict(ID=row.ID, decoded_frames=index, fps=fps,
                     collision_truth=int(row.t_collision), motion_collision=prediction,
                     abs_error_frames=abs(prediction-int(row.t_collision)),
                     abs_error_seconds=abs(prediction-int(row.t_collision))/fps if fps > 0 else None,
                     motion_seconds=time.perf_counter()-start,
                     weak_motion_side=side, overview_frames=[_frame_number(paths[i]) for i in overview]))
report = {'scope':'public five examples, untrained motion-only ablation', 'rows':rows}
(ROOT / 'research/stage2_public/motion_report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
print(json.dumps(report, indent=2))
