"""Retrieve only preselected official ZIP members; preserve CRC, source IDs, native PTS."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import struct
import zlib

import av
from PIL import Image, ImageDraw
import requests

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]


def sha(data):
    return hashlib.sha256(data).hexdigest()


def write(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2))


plan = json.loads((HERE / 'selection.json').read_text())
access = json.loads((HERE / 'zip_range_access.json').read_text())
total = int(access['headers']['Content-Range'].split('/')[-1])


def acquire(case):
    sid, member = case['ID'], case['member']
    out = HERE / sid
    out.mkdir(exist_ok=False)
    start = member['header_offset']
    end = start + 30 + 65535 + member['compressed_size'] - 1
    end = min(end, total - 1)
    with requests.get(access['url'], headers={'Range': f'bytes={start}-{end}', 'Accept-Encoding': 'identity'},
                      timeout=(10, 60), stream=True) as response:
        response.raise_for_status()
        assert response.status_code == 206
        assert response.headers['Content-Range'] == f'bytes {start}-{end}/{total}'
        expected = end - start + 1
        assert int(response.headers.get('Content-Length', expected)) == expected
        raw = response.raw.read(expected + 1)
        http = dict(status=response.status_code, content_range=response.headers['Content-Range'], transfer_limit=expected)
    assert len(raw) == end - start + 1 and raw[:4] == b'PK\x03\x04'
    name_size, extra_size = struct.unpack_from('<HH', raw, 26)
    assert raw[30:30 + name_size].decode() == member['name']
    offset = 30 + name_size + extra_size
    compressed = raw[offset:offset + member['compressed_size']]
    assert member['method'] == 8
    video = zlib.decompress(compressed, -15)
    assert len(video) == member['size'] and zlib.crc32(video) == member['crc']
    path = out / (sid + '.mp4')
    path.write_bytes(video)
    images, mapping, tiles = [], [], []
    (out / 'images').mkdir()
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        stream.thread_count = 2
        for index, frame in enumerate(container.decode(stream)):
            assert frame.pts is not None
            rgb = frame.to_image().convert('RGB')
            img_path = out / 'images' / f'frame_{index:06d}.png'
            rgb.save(img_path)
            pts = dict(frame_id=index, native_pts=frame.pts, time_base=str(frame.time_base), time_s=float(frame.pts * frame.time_base))
            assert not mapping or pts['time_s'] > mapping[-1]['time_s']
            mapping.append(pts)
            images.append(dict(path=str(img_path.relative_to(ROOT)), frame=index, pts_seconds=pts['time_s'],
                               sha256=sha(img_path.read_bytes()), rgb_sha256=sha(rgb.tobytes()), width=rgb.width, height=rgb.height))
            tile = Image.new('RGB', (384, 242))
            rgb.thumbnail((384, 216))
            tile.paste(rgb, (0, 0))
            ImageDraw.Draw(tile).text((4, 220), f'{sid} f{index} t={pts["time_s"]:.3f}', fill='white')
            tiles.append(tile)
    assert len(mapping) == len(case['labels']) == 50
    for page, offset in enumerate(range(0, len(tiles), 16)):
        canvas = Image.new('RGB', (1536, 968))
        for k, tile in enumerate(tiles[offset:offset + 16]):
            canvas.paste(tile, ((k % 4) * 384, (k // 4) * 242))
        canvas.save(out / f'sheet_{page:02d}.jpg', quality=95)
    pts_path = out / 'pts.json'
    write(pts_path, dict(source_sha256=sha(video), mapping=mapping))
    source = dict(ID=sid, source_group=case['source_group'], source_video=str(path.relative_to(ROOT)),
                  source_sha256=sha(video), pts_source=str(pts_path.relative_to(ROOT)), pts_sha256=sha(pts_path.read_bytes()),
                  group='ccd_new_source_first_frame_policy', split='preselected_new_groups', images=images,
                  original_source_startframe=case['original_startframe'],
                  frame_number_definition='0-based ordinal of provided CCD clip, not original YouTube frame index',
                  source_incident_independence='source groups unique; perceptual/visual duplicate review pending',
                  transfer_bytes=len(raw), video_bytes=len(video), zip_crc_verified=True, http_range=http)
    write(out / 'input_manifest.json', source)
    print(json.dumps(dict(ID=sid, frames=len(images), bytes=len(video))), flush=True)
    return source


if __name__ == '__main__':
    with ThreadPoolExecutor(max_workers=3) as pool:
        cases = list(pool.map(acquire, plan['selected']))
    write(HERE / 'inputs.json', cases)
    write(HERE / 'acquisition.json', dict(status='complete', created_utc=datetime.now(timezone.utc).isoformat(),
          source='Official CCD author-linked Google Drive Crash-1500.zip', selection_sha256=sha((HERE / 'selection.json').read_bytes()),
          script_sha256=sha(Path(__file__).read_bytes()), transfer_bytes=sum(c['transfer_bytes'] for c in cases),
          decoded_frames=sum(len(c['images']) for c in cases), cases=len(cases), model_calls=0,
          supplied_ego_yes=True, selection_fixed_before_visual_review=True))
