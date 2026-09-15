"""Create relative asset inventory from Git exclusions; no assets are modified."""
import collections
import hashlib
import json
import os
import subprocess
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]

def main():
    dest=ROOT/'docs/migration/external-assets.jsonl'
    if dest.exists():raise FileExistsError('Preserve the captured inventory')
    result=subprocess.run(['git','-c',f'safe.directory={ROOT.as_posix()}','ls-files','--others','--ignored','--exclude-standard','-z'],cwd=ROOT,check=True,capture_output=True)
    names=sorted(os.fsdecode(p).replace('\\','/') for p in result.stdout.split(b'\0') if p)
    summaries=collections.defaultdict(lambda:{'files':0,'bytes':0,'sha256_verified_files':0})
    skip_parts={'.venv','venv','__pycache__','node_modules','.cache','.parts','.openai','.agents','.codex'}
    dest.parent.mkdir(parents=True,exist_ok=True)
    with dest.open('x',encoding='utf-8') as out:
        for name in names:
            p=ROOT/name
            if set(Path(name).parts)&skip_parts or name.startswith('artifacts/migration/') or p.suffix.lower() in {'.pyc','.log','.part','.incomplete','.pem','.key','.pfx','.p12'}:continue
            critical=(name.startswith(('Baseline/data/','external_data/','model/','research/')) or name=='artifacts/submissions/submit_v6.zip')
            # Raw archive fragments/central directories are optional download archaeology.
            if '/ranges/' in name or name.endswith('central_directory.bin'):critical=False
            tier='resume-evidence' if critical else 'archive-or-redownload'
            size=p.stat().st_size
            digest=None
            if critical:
                with p.open('rb') as f:digest=hashlib.file_digest(f,'sha256').hexdigest()
            row={'path':name,'bytes':size,'sha256':digest,'tier':tier}
            out.write(json.dumps(row,ensure_ascii=False)+'\n')
            s=summaries[tier];s['files']+=1;s['bytes']+=size;s['sha256_verified_files']+=int(digest is not None)
    summary={'groups':dict(summaries),'manifest_sha256':hashlib.sha256(dest.read_bytes()).hexdigest(),
             'destination_rule':'Same relative path under clone root',
             'not_transferred':['Windows virtualenv/native binaries','download fragments/caches','machine-local hosting state','credentials','logs'],
             'optional_sha_limitation':'Archive/redownload entries record size only, not a content-integrity proof.'}
    (dest.parent/'external-assets-summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=True))

if __name__=='__main__':main()
