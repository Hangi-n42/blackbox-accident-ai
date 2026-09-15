"""Bounded reads of public DADA release metadata; never download the full archive."""
import hashlib
import json
from pathlib import Path
import re
import struct
from html import unescape
import requests

OUT = Path(__file__).resolve().parents[1] / 'research/v5_external/dada'
BASE = 'https://drive.usercontent.google.com/download'


def get_range(identifier, byte_range, maximum):
    params = {'id': identifier, 'export': 'download'}
    for attempt in range(2):
        with requests.get(BASE, params=params, headers={'Range': byte_range},
                          timeout=30, stream=True) as r:
            r.raise_for_status()
            ctype = r.headers.get('Content-Type', '')
            if 'text/html' in ctype:
                body = next(r.iter_content(16384)).decode('utf-8')
                match = re.search(r'<form[^>]+action="([^"]+)"[^>]*>(.*?)</form>', body, re.S)
                if attempt or not match or unescape(match.group(1)) != BASE:
                    raise RuntimeError('Public download did not expose the expected confirmation form')
                fields = dict(re.findall(r'name="([^"]+)"\s+value="([^"]*)"', match.group(2)))
                if fields.get('id') != identifier or set(fields) - {'id','export','confirm','uuid'}:
                    raise RuntimeError('Unexpected public-download fields')
                params = {k: unescape(v) for k,v in fields.items()}
                continue
            length = int(r.headers.get('Content-Length', maximum+1))
            if r.status_code != 206 and length > maximum:
                raise RuntimeError('Server ignored bounded range; full download refused')
            data = bytearray()
            for chunk in r.iter_content(65536):
                data.extend(chunk)
                if len(data) > maximum:
                    raise RuntimeError('Bounded download limit exceeded')
            return bytes(data), {'id': identifier, 'range': byte_range,
                'status': r.status_code, 'content_range': r.headers.get('Content-Range'),
                'bytes_received': len(data), 'sha256': hashlib.sha256(data).hexdigest()}
    raise RuntimeError('No binary response')


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    ann = OUT / 'official_annotation.xlsx'
    if not ann.exists():
        data, meta = get_range('1ZLKi6ZtJacBer_xCn-Xs5PS07WrNPsBV', 'bytes=0-', 1024*1024)
        if not data.startswith(b'PK'):
            raise RuntimeError('Annotation is not an XLSX archive')
        ann.write_bytes(data)
        (OUT/'annotation_download.json').write_text(json.dumps(meta, indent=2),encoding='utf-8')
        print(json.dumps({'annotation': meta}), flush=True)
    tail = OUT / 'archive_tail.bin'
    if not tail.exists():
        data, meta = get_range('1QrrSZECLBpBpzhLGa7Lw0YiQ1YRR3P7S', 'bytes=-2097152', 2097152)
        tail.write_bytes(data)
        (OUT/'archive_tail_download.json').write_text(json.dumps(meta, indent=2),encoding='utf-8')
    else:
        data = tail.read_bytes()
        meta = json.loads((OUT/'archive_tail_download.json').read_text(encoding='utf-8'))
    pos = data.rfind(b'PK\x05\x06')
    result = {'tail':meta, 'eocd_found':pos>=0}
    if pos>=0:
        fields = struct.unpack_from('<4s4H2LH', data, pos)
        result['eocd'] = dict(zip(['disk','central_start_disk','disk_records','total_records','central_bytes','central_offset','comment_bytes'],fields[1:]))
    names = []
    for m in re.finditer(b'PK\x01\x02',data):
        p=m.start()
        if p+46<=len(data):
            n=struct.unpack_from('<H',data,p+28)[0]
            names.append(data[p+46:p+46+n].decode('utf-8',errors='replace'))
    result['tail_central_names_count']=len(names)
    result['tail_central_names_sample']=names[:12]+names[-12:]
    (OUT/'probe_report.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(result,ensure_ascii=True),flush=True)


if __name__=='__main__':
    main()
