"""Acquire the official Apache-2.0 model for offline competition inference."""
import os
os.environ['HF_HUB_DISABLE_XET'] = '1'
os.environ['HF_HUB_DISABLE_IMPLICIT_TOKEN'] = '1'
from pathlib import Path
import json
from huggingface_hub import HfApi, snapshot_download

if __name__ == '__main__':
    repo = 'Qwen/Qwen3-VL-2B-Instruct'
    info = HfApi(token=False).model_info(repo)
    target = Path('artifacts/model/stage2/vlm')
    target.mkdir(parents=True, exist_ok=True)
    snapshot_download(repo, revision=info.sha, local_dir=target, token=False,
                      allow_patterns=['*.json', '*.safetensors', '*.txt', '*.jinja', 'LICENSE', 'README.md'],
                      max_workers=3)
    provenance = {'repository': repo, 'revision': info.sha,
                  'source': f'https://huggingface.co/{repo}', 'license': 'Apache-2.0',
                  'purpose': 'Frozen local visual inference; no private-data training'}
    Path('artifacts/vlm_provenance.json').write_text(json.dumps(provenance, indent=2), encoding='utf-8')
    print(json.dumps(provenance), flush=True)
