"""Extract only frozen primary RGB clips from bounded public split-ZIP ranges."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import re
import struct
import zlib
from probe_v5_dada import OUT, get_range

DISKS = {
    0:'1q2wU6zZJekNEjgCeJGb9wS2EQwICswLV',
    1:'1LWWBFqYKiiE_pEEo090G-OjrG4oQAsOL',
    2:'1AydgwavRikuXra_e65eOOVpPTvdafP8u',
    3:'1HUV5Qr2KPXDmxpNeVsCZT9gmHIp4RAUP',
    4:'1GDCuY157lpmSqgCiV4FPSMItkXvEgbJt',
    5:'1QrrSZECLBpBpzhLGa7Lw0YiQ1YRR3P7S',
}


def extract(source, members):
    name='DADA_'+source['source_key'].replace('/','_')
    target=OUT/'inputs/images'/name
    report_path=OUT/'extraction_reports'/f'{name}.json'
    if report_path.exists():
        report=json.loads(report_path.read_text(encoding='utf-8'))
        for row in report['frames']:
            if hashlib.sha256((target/row['filename']).read_bytes()).hexdigest()!=row['sha256']:
                raise AssertionError('Previously extracted file changed')
        return {'source':name,'status':'verified_existing','frames':len(report['frames'])}
    disks={m['disk'] for m in members}
    if len(disks)!=1:
        raise RuntimeError('A selected clip crosses ZIP volumes; bounded extraction requires explicit handling')
    disk=next(iter(disks))
    start=min(m['local_offset'] for m in members)
    end=max(m['local_offset']+m['compressed_bytes']+1024 for m in members)
    if end-start>96*1024*1024:
        raise RuntimeError('Per-clip acquisition budget exceeded')
    cache=OUT/'ranges'/f'{name}.bin'
    cache.parent.mkdir(parents=True,exist_ok=True)
    if cache.exists():
        data=cache.read_bytes()
        download=json.loads(cache.with_suffix('.json').read_text(encoding='utf-8'))
        if hashlib.sha256(data).hexdigest()!=download['sha256']:
            raise AssertionError('Cached range changed')
    else:
        data,download=get_range(DISKS[disk],f'bytes={start}-{end-1}',96*1024*1024)
        expected=f'bytes {start}-{end-1}/'
        if not (download['content_range'] or '').startswith(expected) or len(data)!=end-start:
            raise AssertionError('Range response does not match requested member offsets')
        cache.write_bytes(data)
        cache.with_suffix('.json').write_text(json.dumps(download,indent=2),encoding='utf-8')
    target.mkdir(parents=True,exist_ok=True)
    frames=[]
    for member in members:
        if not re.fullmatch(r'\d+\.(png|jpg|jpeg)',member['filename'],flags=re.I):
            raise ValueError('Unexpected RGB member filename')
        p=member['local_offset']-start
        header=struct.unpack_from('<4s5H3L2H',data,p)
        if header[0]!=b'PK\x03\x04' or header[2]&1:
            raise ValueError('Unexpected or encrypted local ZIP member')
        local_name=data[p+30:p+30+header[-2]].decode('utf-8' if header[2]&0x800 else 'cp437')
        if local_name!=member['path'] or header[3]!=member['method']:
            raise AssertionError('Central/local member mismatch')
        begin=p+30+header[-2]+header[-1]
        compressed=data[begin:begin+member['compressed_bytes']]
        if member['method']==8:
            pixels=zlib.decompress(compressed,-15)
        elif member['method']==0:
            pixels=compressed
        else:
            raise ValueError('Unsupported ZIP compression')
        if len(pixels)!=member['uncompressed_bytes'] or zlib.crc32(pixels)&0xffffffff!=member['crc32']:
            raise AssertionError('Extracted RGB file CRC/size mismatch')
        output=target/member['filename']
        digest=hashlib.sha256(pixels).hexdigest()
        if output.exists() and hashlib.sha256(output.read_bytes()).hexdigest()!=digest:
            raise FileExistsError('Refusing to replace a differing RGB file')
        if not output.exists():
            output.write_bytes(pixels)
        frames.append({'filename':member['filename'],'number':int(Path(member['filename']).stem),
            'sha256':digest,'bytes':len(pixels),'crc32':member['crc32'],'original_member':member['path']})
    report={'source':source,'ID':name,'download':download,'disk':disk,'frames':frames,
        'scope':'Author-distributed RGB files, original numbering. No resampling, labels not passed to inference.'}
    report_path.parent.mkdir(parents=True,exist_ok=True)
    report_path.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    return {'source':name,'status':'extracted_crc_pass','frames':len(frames),'downloaded_bytes':len(data)}


def main():
    selection=json.loads((OUT/'selection_frozen.json').read_text(encoding='utf-8'))
    index=json.loads((OUT/'selected_members.json').read_text(encoding='utf-8'))['members_by_source']
    primary=[s for s in selection['sources'] if s['role']=='primary']
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures=[pool.submit(extract,s,index[s['source_key']]) for s in primary]
        for f in futures:
            print(json.dumps(f.result()),flush=True)


if __name__=='__main__':
    main()
