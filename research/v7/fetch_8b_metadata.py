"""Metadata only. Never request any tensor payload; pin all subsequent URLs."""
import hashlib,json,urllib.request
from pathlib import Path
from datetime import datetime,timezone
HERE=Path(__file__).resolve().parent
OUT=HERE/'qwen3_vl_8b_metadata'
OUT.mkdir(exist_ok=True)
assert not any(OUT.iterdir()), 'Metadata directory must be empty'
REPO='Qwen/Qwen3-VL-8B-Instruct'
MAX=2_000_000
def get(url):
    with urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':'Stage2-metadata-audit'}),timeout=45) as r:
        data=r.read(MAX+1)
        if len(data)>MAX:raise ValueError('Metadata exceeded size limit')
        return data
def digest(data):return hashlib.sha256(data).hexdigest()
info_url=f'https://huggingface.co/api/models/{REPO}?blobs=true'
raw=get(info_url);(OUT/'api_model_info.json').write_bytes(raw);info=json.loads(raw)
revision=info['sha'];records=[]
small={'config.json','preprocessor_config.json','video_preprocessor_config.json','tokenizer_config.json',
       'generation_config.json','model.safetensors.index.json','README.md','LICENSE','NOTICE','chat_template.json','chat_template.jinja'}
downloaded=[]
for item in info['siblings']:
    name=item['rfilename'];url=f'https://huggingface.co/{REPO}/resolve/{revision}/{name}'
    row=dict(name=name,bytes=item.get('size'),git_blob_id=item.get('blobId'),
             lfs_sha256=item.get('lfs',{}).get('sha256'),url=url,downloaded=False)
    if name in small:
        data=get(url);assert len(data)==row['bytes']
        (OUT/name).write_bytes(data);row.update(downloaded=True,local_sha256=digest(data));downloaded.append(len(data))
    records.append(row)
manifest=dict(created_utc=datetime.now(timezone.utc).isoformat(),repository=REPO,revision=revision,
              api_url=info_url,api_response_sha256=digest(raw),gated=info.get('gated'),private=info.get('private'),
              license=info.get('cardData',{}).get('license'),safetensors_metadata=info.get('safetensors'),
              files=records,safetensor_file_bytes=sum(r['bytes'] for r in records if r['name'].endswith('.safetensors')),
              all_repository_file_bytes=sum(r['bytes'] for r in records),metadata_transfer_bytes=len(raw)+sum(downloaded),
              tensor_payload_downloaded=False,model_or_GPU_execution=False)
(HERE/'model_capacity_source_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({k:v for k,v in manifest.items() if k!='files'},ensure_ascii=False))
