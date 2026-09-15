"""Extract only first CAP source sequence from <=64 MiB official gzip prefix."""
from pathlib import Path, PurePosixPath
import io, json, tarfile, hashlib
import requests
from PIL import Image

ROOT=Path(__file__).resolve().parents[2]
OUT=Path(__file__).parent/'mmau_probe'
old=(OUT/'cap_1_10_prefix16m.bin').read_bytes()
rev=json.loads((OUT/'probe_report.json').read_text())['hf_revision']
start,end=len(old),64*1024*1024-1
url=f'https://huggingface.co/datasets/JeffreyChou/MM-AU/resolve/{rev}/CAP-DATA_chunks/1-10/1-10.part_aa?probe=remaining48m'
with requests.get(url,headers={'Range':f'bytes={start}-{end}','Accept-Encoding':'identity'},stream=True,timeout=(15,45)) as response:
    response.raise_for_status()
    assert response.status_code==206 and response.headers.get('Content-Range')==f'bytes {start}-{end}/2147483648'
    extra=bytearray()
    for chunk in response.iter_content(65536):
        if len(extra)+len(chunk)>end-start+1:raise RuntimeError('Transfer limit')
        extra.extend(chunk)
    header=response.headers.get('Content-Range')
assert len(extra)==end-start+1
(OUT/'cap_1_10_remaining48m.bin').write_bytes(extra)
prefix=old+extra
case='CAP-DATA/1-10/8/009509/'
saved=[]; next_source=None
with tarfile.open(fileobj=io.BytesIO(prefix),mode='r|gz') as archive:
    for member in archive:
        if not member.isfile():continue
        if not member.name.startswith(case):
            next_source=member.name
            break
        relative=PurePosixPath(member.name[len(case):])
        assert not relative.is_absolute() and '..' not in relative.parts
        destination=OUT/'cap_8_009509'/Path(*relative.parts)
        destination.parent.mkdir(parents=True,exist_ok=True)
        content=archive.extractfile(member).read()
        assert len(content)==member.size
        with destination.open('xb') as stream:stream.write(content)
        saved.append(dict(path=destination.relative_to(ROOT).as_posix(),archive_name=member.name,
            bytes=len(content),sha256=hashlib.sha256(content).hexdigest()))
assert next_source is not None,'Prefix did not finish entire first source'
images=[x for x in saved if x['path'].endswith('.jpg')]
numbers=sorted(int(Path(x['path']).stem) for x in images)
assert numbers==list(range(1,224))
for row in images:
    with Image.open(ROOT/row['path']) as image:
        image.verify()
metadata=json.loads((OUT/'video_metadata.json').read_text(encoding='utf8'))
matches=[dict(video_hashcode=k,**v) for k,v in metadata.items() if v['video_name']=='8_9509']
assert len(matches)==1 and matches[0]['total_frames']=='223'
(OUT/'cap_8_009509'/'official_annotation.json').write_text(json.dumps(matches[0],indent=2),encoding='utf8')
result=dict(scope='Acquisition and file integrity only; no viewing/annotation/model inference',
    source_selected_before_predictions='First complete tar source, not random representative sample',
    additional_transfer_bytes=len(extra),content_range=header,url=url,sha256=hashlib.sha256(extra).hexdigest(),
    entire_prefix_bytes=len(prefix),complete_source=True,next_source_seen=next_source,
    files=saved,images=len(images),frame_min=min(numbers),frame_max=max(numbers),
    official_annotation=matches[0],official_seconds_mapping_available=False)
(OUT/'extraction_report.json').write_text(json.dumps(result,indent=2),encoding='utf8')
print(json.dumps({k:v for k,v in result.items() if k!='files'},indent=2))
