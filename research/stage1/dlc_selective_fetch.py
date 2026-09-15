"""Bounded official DLC archive reads; never accept HTTP 200 for a Range request."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import re
import struct
import urllib.request
import zlib

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'dlc_subset'
META = ROOT / 'dlc_metadata'
OUT.mkdir(exist_ok=True)
LIMIT = 200_000_000
LOG = OUT / 'network_log.json'
HISTORY = json.loads(LOG.read_text()) if LOG.exists() else []


def save_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def fetch(url, cap, byte_range=None):
    if sum(x['bytes'] for x in HISTORY) + cap > LIMIT:
        raise RuntimeError('200 MB total response budget exceeded')
    headers = {'User-Agent': 'DLC-bounded-research/1.0'}
    if byte_range:
        headers['Range'] = 'bytes=' + byte_range
    with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=40) as r:
        if byte_range and r.status != 206:
            raise RuntimeError(f'Range not honored: {r.status}; body not read')
        content_range = r.headers.get('Content-Range')
        if byte_range:
            match = re.fullmatch(r'bytes (\d+)-(\d+)/(\d+)', content_range or '')
            if not match:
                raise RuntimeError('Invalid Content-Range; body not read')
            start, end, total = map(int, match.groups())
            expected = ((total - int(byte_range[1:]), total - 1) if byte_range.startswith('-')
                        else tuple(map(int, byte_range.split('-'))))
            if (start, end) != expected or end - start + 1 > cap:
                raise RuntimeError('Unexpected response range; body not read')
        length = int(r.headers.get('Content-Length', cap + 1))
        if length > cap:
            raise RuntimeError('Content-Length exceeds budget; body not read')
        data = r.read(cap + 1)
        if len(data) != length or len(data) > cap:
            HISTORY.append({'url':url, 'range':byte_range, 'status':r.status,
                            'content_range':content_range, 'bytes':len(data),
                            'error':f'Length mismatch: received {len(data)}, declared {length}, cap {cap}'})
            save_json(LOG,HISTORY)
            raise RuntimeError(HISTORY[-1]['error'])
        HISTORY.append({'url': url, 'range': byte_range, 'status': r.status,
                        'content_range': content_range, 'bytes': len(data),
                        'sha256': hashlib.sha256(data).hexdigest()})
        save_json(LOG, HISTORY)
    return data, content_range


def file_url(record, name):
    return f'https://zenodo.org/records/{record}/files/{name}?download=1'


def parse_index(data):
    entries, i = [], 0
    while i < len(data):
        h = struct.unpack_from('<4s6H3I5H2I', data, i)
        if h[0] != b'PK\x01\x02':
            raise ValueError(f'Invalid central directory at {i}')
        nl, xl, cl = h[10:13]
        name = data[i+46:i+46+nl].decode('utf-8')
        unpacked, compressed, offset, disk = h[9], h[8], h[16], h[13]
        extra = data[i+46+nl:i+46+nl+xl]
        j = 0
        while j + 4 <= len(extra):
            kind, length = struct.unpack_from('<HH', extra, j)
            value = extra[j+4:j+4+length]
            if kind == 1:
                p = 0
                if unpacked == 0xffffffff:
                    unpacked = struct.unpack_from('<Q', value, p)[0]; p += 8
                if compressed == 0xffffffff:
                    compressed = struct.unpack_from('<Q', value, p)[0]; p += 8
                if offset == 0xffffffff:
                    offset = struct.unpack_from('<Q', value, p)[0]; p += 8
                if disk == 0xffff:
                    disk = struct.unpack_from('<I', value, p)[0]
            j += 4 + length
        entries.append({'path': name, 'compressed_bytes': compressed,
                        'uncompressed_bytes': unpacked, 'start_disk': disk,
                        'local_header_offset': offset, 'method': h[4], 'crc32': f'{h[7]:08x}'})
        i += 46 + nl + xl + cl
    return entries


def metadata():
    for rec in [7467028, 6466770]:
        data, _ = fetch(file_url(rec, 'license.txt'), 10000)
        (OUT / f'license_{rec}.txt').write_bytes(data)
        if 'creativecommons.org/licenses/by-sa/2.5' not in data.decode('utf-8'):
            raise RuntimeError('Expected data license not found; inspect before payload')
    data, _ = fetch(file_url(7467028, 'dlc-2021.csv'), 100000)
    (OUT / 'dlc-2021_corrected.csv').write_bytes(data)
    url = file_url(7467028, 'or.zip')
    tail, cr = fetch(url, 65557, '-65557')
    size = int(cr.split('/')[-1]); base = size - len(tail)
    i = tail.rfind(b'PK\x05\x06')
    eocd = struct.unpack_from('<4s4H2IH', tail, i)
    cd_size, cd_offset = eocd[5:7]
    locator = tail.rfind(b'PK\x06\x07', 0, i)
    if locator >= 0:
        _, _, zoff, _ = struct.unpack_from('<4sIQI', tail, locator)
        z = tail[zoff-base:zoff-base+56] if zoff >= base else fetch(url, 56, f'{zoff}-{zoff+55}')[0]
        zh = struct.unpack('<4sQ2H2I4Q', z)
        cd_size, cd_offset = zh[-2:]
    if cd_size > 10_000_000:
        raise RuntimeError('Unexpectedly large metadata')
    central, _ = fetch(url, cd_size, f'{cd_offset}-{cd_offset+cd_size-1}')
    entries = parse_index(central)
    with (OUT/'or_archive_index.csv').open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, entries[0].keys()); w.writeheader(); w.writerows(entries)
    save_json(OUT/'or_archive_summary.json', {'record':7467028, 'archive_size':size,
              'central_size':cd_size, 'entries':len(entries)})
    print('metadata complete', len(entries), 'entries', flush=True)


def selection():
    rows = list(csv.reader((OUT/'dlc-2021_corrected.csv').open(), delimiter=';'))
    groups = {}
    for ident, camera, condition in rows:
        doc, suffix = ident.split('.')
        label = suffix[:2]
        if label in ['or', 're']:
            groups.setdefault(doc, {}).setdefault((label,camera), []).append((ident, condition))
    eligible = [g for g,v in groups.items() if all((c,d) in v for c in ['or','re'] for d in ['iphone','android'])]
    docs = sorted(eligible, key=lambda s: hashlib.sha256(s.encode()).hexdigest())[:6]
    indices = {}
    for cls, path in [('or', OUT/'or_archive_index.csv'), ('re', META/'re_archive_index.csv')]:
        indices[cls] = list(csv.DictReader(path.open()))
    chosen = []
    for doc in docs:
        for cls in ['or','re']:
            for cam in ['iphone','android']:
                ident, condition = sorted(groups[doc][cls,cam], key=lambda v:hashlib.sha256(v[0].encode()).hexdigest())[0]
                folder = doc.split('/')[0] + '/' + ident.split('/')[1]
                frames = sorted([e for e in indices[cls] if '/'+folder+'/' in e['path'] and e['path'].endswith('.jpg')],
                                key=lambda e:int(Path(e['path']).stem))
                if not frames:
                    raise RuntimeError(f'No indexed frames for {ident}')
                nums = sorted({round(i*(len(frames)-1)/7) for i in range(min(8,len(frames)))})
                chosen.append({'document_id':doc, 'video_id':ident, 'label':cls, 'camera':cam,
                               'condition_or_display':condition, 'available_frames':len(frames),
                               'frames':[frames[i] for i in nums]})
    payload = sum(int(f['compressed_bytes'])+1024 for v in chosen for f in v['frames'])
    if payload + sum(x['bytes'] for x in HISTORY) > LIMIT:
        raise RuntimeError('Selected subset exceeds budget; do not alter based on outcomes')
    plan = {'selection':'first 6 eligible document IDs in SHA256(document ID) ascending order; one video per label/camera by SHA256(video ID); at most 8 evenly spaced distributed JPEGs',
            'eligible_documents':len(eligible), 'documents':docs, 'videos':chosen,
            'maximum_payload_and_header_bytes':payload, 'frozen_before_model_results':True}
    save_json(OUT/'selection_plan.json', plan)
    print('selection', docs, len(chosen), 'videos', payload, 'bytes', flush=True)


def download():
    plan = json.loads((OUT/'selection_plan.json').read_text())
    manifest_path = OUT/'acquisition_manifest.json'
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else []
    done = {m['archive_path'] for m in manifest}
    for v in plan['videos']:
        for e in v['frames']:
            if e['path'] in done:
                continue
            if v['label'] == 'or':
                record, archive = 7467028, 'or.zip'
            else:
                record = 6466770; disk=int(e['start_disk'])
                archive = 're.zip' if disk == 8 else f're.z{disk+1:02}'
            offset, n = int(e['local_header_offset']), int(e['compressed_bytes'])
            # All selected items must stay inside one archive part; no implicit full fetch.
            for attempt in range(3):
                try:
                    raw, cr = fetch(file_url(record,archive), n+1024, f'{offset}-{offset+n+1023}')
                    break
                except RuntimeError as exc:
                    if not str(exc).startswith('Length mismatch:') or attempt == 2:
                        raise
                    print('retry same range after logged truncated response', e['path'], flush=True)
            h = struct.unpack_from('<4s5H3I2H',raw)
            if h[0] != b'PK\x03\x04':
                raise RuntimeError('Bad local header')
            nl,xl=h[-2:]; name=raw[30:30+nl].decode('utf-8')
            if name != e['path'] or 30+nl+xl > 1024 or h[2]&1:
                raise RuntimeError('Local header identity/bounds/encryption mismatch')
            packed=raw[30+nl+xl:30+nl+xl+n]
            if len(packed)!=n:
                raise RuntimeError('Split boundary requires separate handling; stopped')
            decoded=packed if int(e['method'])==0 else zlib.decompress(packed,-15) if int(e['method'])==8 else None
            if decoded is None or len(decoded)!=int(e['uncompressed_bytes']) or f'{zlib.crc32(decoded):08x}'!=e['crc32']:
                raise RuntimeError('CRC or size mismatch')
            path=OUT/'images'/e['path']
            path.parent.mkdir(parents=True, exist_ok=True);path.write_bytes(decoded)
            manifest.append({k:v[k] for k in ['document_id','video_id','label','camera','condition_or_display']} |
                            {'archive_path':e['path'],'local_path':str(path.relative_to(ROOT)), 'source_url':file_url(record,archive),
                             'http_range':cr,'crc32':e['crc32'],'sha256':hashlib.sha256(decoded).hexdigest(),
                             'bytes':len(decoded),'license':'CC BY-SA 2.5','license_file':f'license_{record}.txt'})
            save_json(manifest_path,manifest)
        print('acquired',v['video_id'],v['camera'], 'total',len(manifest),flush=True)
    print('completed response bytes',sum(x['bytes'] for x in HISTORY),flush=True)


if __name__ == '__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['metadata','select','download'])
    args=parser.parse_args()
    {'metadata':metadata,'select':selection,'download':download}[args.mode]()
