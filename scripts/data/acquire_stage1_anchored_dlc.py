"""Acquire frozen DLC selections with verified partial HTTP reads and CRCs."""
import argparse
import concurrent.futures
import hashlib
import json
import struct
import time
import zlib
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'artifacts/stage1_anchored_head_20260919'
OLD = ROOT / 'research/stage1/dlc_subset'


def write(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False))


def acquire(split):
    if split == 'confirmation':
        assert json.loads((OUT / 'selection.json').read_text())['gate_pass']
    plan = json.loads((OUT / 'dlc_plan.json').read_text())
    for rec in (7467028, 6466770):
        data = (OLD / f'license_{rec}.txt').read_bytes()
        assert b'creativecommons.org/licenses/by-sa/2.5' in data
        (OUT / f'license_{rec}.txt').write_bytes(data)
    previous = {r['archive_path']: r for r in json.loads((OLD / 'acquisition_manifest.json').read_text())}
    manifest = OUT / f'dlc_{split}_acquired.json'
    completed = json.loads(manifest.read_text()) if manifest.exists() else []
    done = {r['archive_path'] for r in completed}
    jobs = [(v, f) for v in plan['videos'] if v['split'] == split for f in v['frames'] if f['path'] not in done]

    def fetch(job):
        v, entry = job
        target = OUT / 'dlc_images' / entry['path']
        rec = 7467028 if v['label'] == 'or' else 6466770
        disk = int(entry['start_disk'])
        archive = 'or.zip' if v['label'] == 'or' else ('re.zip' if disk == 8 else f're.z{disk+1:02}')
        url = f'https://zenodo.org/records/{rec}/files/{archive}?download=1'
        offset, n = int(entry['local_header_offset']), int(entry['compressed_bytes'])
        history = []
        prior = previous.get(entry['path'])
        if prior:
            source = ROOT / 'research/stage1' / prior['local_path'].replace('\\', '/')
            data = source.read_bytes()
            assert hashlib.sha256(data).hexdigest() == prior['sha256']
            target = source
        elif target.exists():
            data = target.read_bytes()
        else:
            for attempt in range(3):
                try:
                    time.sleep(1.1)
                    end = offset + n + 1023
                    with requests.get(url + f'&anchor_range={offset}&attempt={attempt}',
                                      headers={'Range': f'bytes={offset}-{end}'}, stream=True, timeout=40) as r:
                        if r.status_code == 429:
                            delay = max(60, int(r.headers.get('Retry-After', '60')))
                            print('Rate limited; waiting', delay, 'seconds', flush=True)
                            time.sleep(delay)
                        r.raise_for_status()
                        assert r.status_code == 206, ('HTTP Range unsupported; body not read', r.status_code)
                        assert r.headers.get('Content-Range', '').startswith(f'bytes {offset}-{end}/')
                        raw = r.raw.read(n + 1025)
                        history.append({'attempt': attempt, 'bytes': len(raw), 'range': r.headers['Content-Range']})
                        assert len(raw) == n + 1024
                    header = struct.unpack_from('<4s5H3I2H', raw)
                    nl, xl = header[-2:]
                    assert header[0] == b'PK\x03\x04' and not (header[2] & 1)
                    assert raw[30:30+nl].decode() == entry['path'] and 30+nl+xl <= 1024
                    packed = raw[30+nl+xl:30+nl+xl+n]
                    assert int(entry['method']) in (0, 8)
                    data = packed if int(entry['method']) == 0 else zlib.decompress(packed, -15)
                    break
                except Exception as exc:
                    history.append({'attempt': attempt, 'failure': str(exc), 'unlogged_transfer_upper_bound': n+1025})
                    if attempt == 2:
                        return {'failed': entry['path'], 'history': history}
                    time.sleep(attempt + 1)
        assert len(data) == int(entry['uncompressed_bytes'])
        assert f'{zlib.crc32(data):08x}' == entry['crc32']
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
        return {k: v[k] for k in ('split', 'document_id', 'document_type', 'video_id', 'label', 'camera', 'condition_or_display')} | {
            'archive_path': entry['path'], 'path': str(target.relative_to(ROOT)), 'sha256': hashlib.sha256(data).hexdigest(),
            'crc32': entry['crc32'], 'source_url': url, 'history': history, 'license': 'CC BY-SA 2.5', 'reused_old': bool(prior)}

    failures = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        for r in pool.map(fetch, jobs):
            if 'failed' in r:
                failures.append(r)
                write(OUT / f'dlc_{split}_failures.json', failures)
            else:
                completed.append(r)
                write(manifest, completed)
            if (len(completed)+len(failures)) % 24 == 0:
                print(split, 'acquired', len(completed), 'failed', len(failures), flush=True)
    assert not failures, f'{len(failures)} acquisition failures; fixed selections retained'
    expected = sum(12 for v in plan['videos'] if v['split'] == split)
    assert len(completed) == expected
    for r in completed:
        assert hashlib.sha256((ROOT / r['path']).read_bytes()).hexdigest() == r['sha256']
    print('VERIFIED', split, expected, 'frames', flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('split', choices=['train', 'development', 'confirmation'])
    acquire(p.parse_args().split)
