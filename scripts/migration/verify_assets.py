"""Check copied assets against the immutable Windows capture; never overwrite."""
import argparse
import hashlib
import json
from pathlib import Path
from paths import ROOT,resolve_recorded

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--all',action='store_true',help='Also check archive/redownload sizes')
    args=ap.parse_args();missing=[];different=[];checked=0
    for line in (ROOT/'docs/migration/external-assets.jsonl').read_text(encoding='utf-8').splitlines():
        r=json.loads(line)
        if r['tier']!='resume-evidence' and not args.all:continue
        p=resolve_recorded(r['path'])
        if not p.is_file():missing.append(r['path']);continue
        if p.stat().st_size!=r['bytes']:different.append(r['path']);continue
        if r['sha256']:
            with p.open('rb') as f:digest=hashlib.file_digest(f,'sha256').hexdigest()
            if digest!=r['sha256']:different.append(r['path']);continue
        checked+=1
    print(json.dumps({'checked':checked,'missing_count':len(missing),'different_count':len(different),'missing':missing[:30],'different':different[:30]},ensure_ascii=False,indent=2))
    if missing or different:raise SystemExit(1)

if __name__=='__main__':main()
