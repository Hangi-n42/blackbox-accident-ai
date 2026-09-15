"""Freeze source-level Stage2 inputs for paired V5 development diagnostics.

Lossless decoding prevents accidental JPEG changes between experiment and ZIP
verification. Mirroring is a stress test of the same sources, not extra GT.
"""
from pathlib import Path
import hashlib
import json
from datetime import datetime, timezone
import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'Baseline/data/stage2'
OUT = ROOT / 'research/v5_inputs'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    if OUT.exists():
        raise FileExistsError(f'Refusing to replace frozen diagnostic inputs: {OUT}')
    OUT.mkdir(parents=True)
    labels = pd.read_csv(SOURCE / 'labels.csv')
    manifest = {
        'created_at': datetime.now(timezone.utc).isoformat(),
        'scope': 'Five reused public source videos; paired representation and mirror diagnostics only. Not independent validation.',
        'source_labels_sha256': sha(SOURCE / 'labels.csv'),
        'preparer_sha256': sha(Path(__file__)),
        'protocol': {
            'canonical': 'OpenCV sequential BGR decode -> lossless PNG, no resizing.',
            'mirror': 'Horizontal reflection of canonical decoded BGR. Original frame numbers and chronology preserved.',
            'existing_jpeg': 'Read-only audit of artifacts/public_eval/stage2/images.',
            'interpretation': 'Mirror temporal/space consistency and LEFT/RIGHT reversal are diagnostic hypotheses; not additional independent examples or inferred missing official labels.',
            'prediction_policy': 'No candidate output or official hidden data used in constructing inputs.',
        },
        'videos': [],
    }
    for row in labels.itertuples():
        source = SOURCE / row.path
        destinations = {v: OUT / v / 'images' / row.ID for v in ['canonical', 'mirror']}
        for folder in destinations.values():
            folder.mkdir(parents=True)
        cap = cv2.VideoCapture(str(source))
        observed_fps = cap.get(cv2.CAP_PROP_FPS)
        frames = []
        while True:
            ok, bgr = cap.read()
            if not ok:
                break
            index = len(frames)
            versions = {'canonical': bgr, 'mirror': np.ascontiguousarray(bgr[:, ::-1])}
            entry = {'number': index, 'decoded_shape': list(bgr.shape),
                     'observed_pos_msec': cap.get(cv2.CAP_PROP_POS_MSEC), 'versions': {}}
            for variant, pixels in versions.items():
                target = destinations[variant] / f'frame_{index:06d}.png'
                ok, encoded = cv2.imencode('.png', pixels)
                if not ok:
                    raise RuntimeError(f'PNG encoding failed: {row.ID}/{index}')
                encoded.tofile(str(target))
                readback = cv2.imdecode(np.fromfile(str(target), dtype=np.uint8), cv2.IMREAD_COLOR)
                if not np.array_equal(readback, pixels):
                    raise AssertionError(f'Lossless round trip differs: {target}')
                entry['versions'][variant] = {'path': str(target.relative_to(ROOT)),
                    'file_sha256': sha(target), 'bgr_sha256': hashlib.sha256(pixels.tobytes()).hexdigest()}
            jpeg = ROOT / 'artifacts/public_eval/stage2/images' / row.ID / f'frame_{index:06d}.jpg'
            pixels = cv2.imdecode(np.fromfile(str(jpeg), dtype=np.uint8), cv2.IMREAD_COLOR)
            if pixels is None or pixels.shape != bgr.shape:
                raise AssertionError(f'JPEG shape mismatch: {jpeg}')
            entry['versions']['existing_jpeg'] = {'path': str(jpeg.relative_to(ROOT)),
                'file_sha256': sha(jpeg), 'bgr_sha256': hashlib.sha256(pixels.tobytes()).hexdigest(),
                'mean_absolute_difference_from_canonical': float(np.abs(pixels.astype(np.int16)-bgr.astype(np.int16)).mean())}
            frames.append(entry)
        cap.release()
        if not frames:
            raise RuntimeError(f'No decoded frames: {source}')
        manifest['videos'].append({'ID': row.ID, 'source_path': str(source.relative_to(ROOT)),
            'source_sha256': sha(source), 'observed_metadata_fps': observed_fps,
            'frame_count': len(frames), 'frames': frames})
        print(f'{row.ID}: {len(frames)} canonical and mirror frames verified', flush=True)
    (OUT / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'Frozen {len(manifest["videos"])} source groups at {OUT}', flush=True)


if __name__ == '__main__':
    main()
