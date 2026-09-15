"""Bounded official metadata and compressed-prefix feasibility probe; no inference."""
from pathlib import Path
import hashlib, io, json, tarfile
import requests

OUT=Path(__file__).parent/'mmau_probe'
OUT.mkdir(exist_ok=True)
records=[]
def get(url,name,limit,range_request=False):
    headers={'Range':f'bytes=0-{limit-1}','Accept-Encoding':'identity'} if range_request else {}
    with requests.get(url,headers=headers,stream=True,timeout=(15,45)) as r:
        r.raise_for_status()
        if range_request and r.status_code!=206: raise RuntimeError('Server ignored bounded range')
        data=bytearray()
        for chunk in r.iter_content(65536):
            if len(data)+len(chunk)>limit: raise RuntimeError('Body exceeds predeclared metadata/probe limit')
            data.extend(chunk)
        b=bytes(data)
        (OUT/name).write_bytes(b)
        records.append(dict(url=url,path=name,bytes=len(b),status=r.status_code,
            content_range=r.headers.get('Content-Range'),sha256=hashlib.sha256(b).hexdigest()))
        return b

info=json.loads(get('https://huggingface.co/api/datasets/JeffreyChou/MM-AU','hf_info.json',1_000_000))
rev=info['sha']
git=json.loads(get('https://api.github.com/repos/jeffreychou777/LOTVS-MM-AU/commits/main','github_revision.json',1_000_000))
gitrev=git['sha']
get(f'https://huggingface.co/datasets/JeffreyChou/MM-AU/raw/{rev}/README.md','hf_README.md',100_000)
get(f'https://raw.githubusercontent.com/jeffreychou777/LOTVS-MM-AU/{gitrev}/README.md','github_README.md',100_000)
get(f'https://raw.githubusercontent.com/jeffreychou777/LOTVS-MM-AU/{gitrev}/video_metadata.json','video_metadata.json',30_000_000)
prefix=get(f'https://huggingface.co/datasets/JeffreyChou/MM-AU/resolve/{rev}/CAP-DATA_chunks/1-10/1-10.part_aa?probe=prefix16m',
           'cap_1_10_prefix16m.bin',16*1024*1024,True)
entries=[]
try:
    with tarfile.open(fileobj=io.BytesIO(prefix),mode='r|gz') as archive:
        for member in archive:
            entries.append(dict(name=member.name,size=member.size,type=member.type.decode(errors='replace')))
            if len(entries)>=200:break
except (EOFError,tarfile.TarError) as error:
    entries.append(dict(prefix_end=f'{type(error).__name__}: {error}'))
result=dict(hf_revision=rev,github_revision=gitrev,downloaded_bytes=sum(x['bytes'] for x in records),records=records,tar_entries=entries)
(OUT/'probe_report.json').write_text(json.dumps(result,indent=2),encoding='utf8')
print(json.dumps(result,indent=2))
