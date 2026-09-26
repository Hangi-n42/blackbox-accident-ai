"""Source-only native-PTS review images; contains no annotations or predictions."""
import argparse
import hashlib
import json
from pathlib import Path

import av
from PIL import Image, ImageDraw, ImageOps

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent / 'acquisition'


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def extract(case_id, out, wanted=None):
    source, pts_path = SOURCE / (case_id + '.mp4'), SOURCE / (case_id + '.pts.json')
    pts = json.loads(pts_path.read_text())
    assert sha(source) == pts['source_sha256']
    mapping = pts['mapping']
    overview = wanted is None
    if overview:
        wanted, next_time = [], 0.
        for row in mapping:
            if row['time_s'] + 1e-9 >= next_time:
                wanted.append(row['frame_id'])
                next_time += .5
        if wanted[-1] != mapping[-1]['frame_id']:
            wanted.append(mapping[-1]['frame_id'])
    wanted = sorted(set(wanted))
    assert wanted and min(wanted) >= 0 and max(wanted) < len(mapping)
    out.mkdir(parents=True, exist_ok=True)
    selected, records = set(wanted), []
    tile_w, tile_h, page = 384, 244, None
    with av.open(str(source)) as container:
        stream = container.streams.video[0]
        stream.thread_count = 2
        for index, frame in enumerate(container.decode(stream)):
            if index not in selected:
                continue
            assert index == mapping[index]['frame_id']
            assert frame.pts == mapping[index]['native_pts'] and str(frame.time_base) == mapping[index]['time_base']
            im = frame.to_image().convert('RGB')
            position = len(records) % 16
            if position == 0:
                page = Image.new('RGB', (tile_w * 4, tile_h * 4), '#101010')
            thumb = ImageOps.contain(im, (tile_w, tile_h - 28))
            x, y = (position % 4) * tile_w, (position // 4) * tile_h
            page.paste(thumb, (x + (tile_w - thumb.width) // 2, y + 28))
            ImageDraw.Draw(page).text((x + 4, y + 5), f'{case_id} f{index} t={mapping[index]["time_s"]:.6f}s', fill='white')
            row = dict(frame=index, pts_seconds=mapping[index]['time_s'], rgb_sha256=hashlib.sha256(im.tobytes()).hexdigest())
            if not overview:
                path = out / f'frame_{index:06d}.png'
                assert not path.exists()
                im.save(path, compress_level=1)
                row.update(native_image=str(path), image_sha256=sha(path))
            records.append(row)
            if position == 15 or index == wanted[-1]:
                path = out / f'sheet_{(len(records) - 1) // 16:02d}.jpg'
                assert not path.exists()
                page.save(path, quality=95)
            if index == wanted[-1]:
                break
    assert len(records) == len(wanted)
    record = dict(ID=case_id, source_sha256=pts['source_sha256'], pts_sha256=sha(pts_path),
                  scope='fixed 2Hz full-timeline overview' if overview else 'requested native frames',
                  model_predictions_used=False, previous_annotations_used=False, frames=records)
    with (out / 'manifest.json').open('x') as stream:
        json.dump(record, stream, ensure_ascii=False, indent=2)
    print(json.dumps(dict(ID=case_id, selected=len(records), output=str(out))))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--id', required=True, choices=[f'{i:05d}' for i in range(24, 30)])
    parser.add_argument('--frames', nargs='+', type=int)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    extract(args.id, args.output.resolve(), args.frames)
