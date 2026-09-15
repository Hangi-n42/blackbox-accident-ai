"""Download an official Qwen candidate into a separate, checksum-verified folder."""
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import hashlib
import json
import time
import requests
from huggingface_hub import HfApi


def sha256(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    repo = 'Qwen/Qwen3-VL-4B-Instruct'
    info = HfApi(token=False).model_info(repo, files_metadata=True)
    target = Path('artifacts/candidates/qwen3_vl_4b')
    target.mkdir(parents=True, exist_ok=True)
    record = {'repository': repo, 'revision': info.sha,
              'source': f'https://huggingface.co/{repo}', 'files': []}
    names = {'config.json', 'generation_config.json', 'preprocessor_config.json',
             'video_preprocessor_config.json', 'tokenizer_config.json',
             'tokenizer.json', 'vocab.json', 'merges.txt', 'chat_template.json',
             'model.safetensors.index.json', 'README.md', 'LICENSE'}
    for item in info.siblings:
        name = item.rfilename
        if name not in names and not (name.startswith('model-') and name.endswith('.safetensors')):
            continue
        assert '/' not in name and '\\' not in name
        path = target / name
        url = f'https://huggingface.co/{repo}/resolve/{info.sha}/{name}'
        expected_sha = item.lfs.sha256 if item.lfs else None
        if path.exists() and path.stat().st_size == item.size and (not expected_sha or sha256(path) == expected_sha):
            print(f'Already verified: {name}', flush=True)
        elif name.endswith('.safetensors'):
            total = item.size
            chunk = 16 * 1024 * 1024
            parts = target / '.parts' / name
            parts.mkdir(parents=True, exist_ok=True)
            count = (total + chunk - 1) // chunk

            def download(index):
                start = index * chunk
                end = min(total, start + chunk) - 1
                part = parts / f'{index:04d}.part'
                if part.exists() and part.stat().st_size == end - start + 1:
                    return
                for attempt in range(3):
                    try:
                        with requests.get(url, params={'download': 'true', 'part': index},
                                          headers={'Range': f'bytes={start}-{end}'},
                                          timeout=(20, 60)) as response:
                            response.raise_for_status()
                            if response.status_code != 206 or response.headers.get('Content-Range') != f'bytes {start}-{end}/{total}':
                                raise ValueError('Unexpected byte range')
                            data = response.content
                        if len(data) != end - start + 1:
                            raise ValueError('Truncated byte range')
                        part.write_bytes(data)
                        return
                    except Exception:
                        if attempt == 2:
                            raise
                        time.sleep(2)

            with ThreadPoolExecutor(max_workers=4) as pool:
                for done, future in enumerate(as_completed(pool.submit(download, i) for i in range(count)), 1):
                    future.result()
                    if done % 16 == 0 or done == count:
                        print(f'{name}: {done}/{count} parts', flush=True)
            temporary = path.with_suffix('.assembling')
            with temporary.open('wb') as stream:
                for index in range(count):
                    stream.write((parts / f'{index:04d}.part').read_bytes())
            if not expected_sha or sha256(temporary) != expected_sha:
                raise ValueError(f'Checkpoint SHA256 mismatch: {name}')
            temporary.replace(path)
        else:
            response = requests.get(url, timeout=(20, 60))
            response.raise_for_status()
            if len(response.content) != item.size:
                raise ValueError(f'File size mismatch: {name}')
            path.write_bytes(response.content)
        record['files'].append({'name': name, 'bytes': path.stat().st_size,
                                'sha256': sha256(path), 'expected_lfs_sha256': expected_sha})
        (target / 'download_manifest.json').write_text(json.dumps(record, indent=2), encoding='utf-8')
    print(json.dumps({'revision': info.sha, 'total_bytes': sum(x['bytes'] for x in record['files'])}), flush=True)


if __name__ == '__main__':
    main()
