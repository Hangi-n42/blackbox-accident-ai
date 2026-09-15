"""Fetch only the ZIP central directory for bounded source-level selection."""
import json
from pathlib import Path
from probe_v5_dada import OUT, get_range

def main():
    target = OUT/'central_directory.bin'
    if target.exists():
        raise FileExistsError('Existing directory is preserved; inspect its report instead')
    start, size = 18035845548, 482459595
    data, metadata = get_range('1QrrSZECLBpBpzhLGa7Lw0YiQ1YRR3P7S',
        f'bytes={start}-{start+size-1}', size)
    if len(data)!=size or not data.startswith(b'PK\x01\x02'):
        raise AssertionError('Central-directory range did not match the observed ZIP64 record')
    target.write_bytes(data)
    (OUT/'central_directory_download.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
    print(json.dumps(metadata),flush=True)

if __name__=='__main__':
    main()
