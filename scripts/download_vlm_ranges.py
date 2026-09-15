"""Download verified byte ranges when the large-file transfer does not stream."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib,json,time
import requests
from huggingface_hub import HfApi

def main():
    repo='Qwen/Qwen3-VL-2B-Instruct'
    info=HfApi(token=False).model_info(repo,files_metadata=True)
    entry=next(x for x in info.siblings if x.rfilename=='model.safetensors')
    total=entry.size
    sha=entry.lfs.sha256
    target=Path('artifacts/model/stage2/vlm/model.safetensors')
    parts=Path('artifacts/downloads/vlm_parts');parts.mkdir(parents=True,exist_ok=True)
    chunk=16*1024*1024
    def download(i):
        start=i*chunk;end=min(total,start+chunk)-1
        path=parts/f'{i:04d}.part'
        if path.exists() and path.stat().st_size==end-start+1:return i
        for attempt in range(3):
            try:
                url=f'https://huggingface.co/{repo}/resolve/{info.sha}/model.safetensors'
                with requests.get(url,params={'download':'true','part':i},
                                  headers={'Range':f'bytes={start}-{end}'},timeout=(20,60)) as r:
                    r.raise_for_status()
                    expected=f'bytes {start}-{end}/{total}'
                    if r.status_code!=206 or r.headers.get('Content-Range')!=expected:
                        raise ValueError(f'Range mismatch for {i}: {r.status_code} {r.headers.get("Content-Range")}')
                    data=r.content
                    if len(data)!=end-start+1:raise ValueError('Truncated range')
                    path.write_bytes(data)
                    return i
            except Exception:
                if attempt==2:raise
                time.sleep(2)
    n=(total+chunk-1)//chunk
    with ThreadPoolExecutor(max_workers=4) as pool:
        for done,future in enumerate(as_completed([pool.submit(download,i) for i in range(n)]),1):
            future.result()
            if done%8==0 or done==n:print(f'Parts {done}/{n}',flush=True)
    temp=target.with_suffix('.assembling');digest=hashlib.sha256()
    with temp.open('wb') as out:
        for i in range(n):
            data=(parts/f'{i:04d}.part').read_bytes();digest.update(data);out.write(data)
    if digest.hexdigest()!=sha:raise ValueError('Model SHA256 verification failed')
    temp.replace(target)
    record={'repository':repo,'revision':info.sha,'source':f'https://huggingface.co/{repo}',
            'license':'Apache-2.0','model_sha256':sha,'bytes':total,
            'purpose':'Frozen offline visual inference, no evaluation-data training'}
    Path('artifacts/vlm_provenance.json').write_text(json.dumps(record,indent=2),encoding='utf-8')
    print(json.dumps(record),flush=True)

if __name__=='__main__':main()
