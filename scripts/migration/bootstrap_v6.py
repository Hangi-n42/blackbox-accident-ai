"""Restore the exact V6 source snapshot or a verified local V6 ZIP, without overwrite."""
import argparse
import hashlib
import json
import shutil
import zipfile
from pathlib import Path, PurePosixPath

ROOT=Path(__file__).resolve().parents[2]
ZIP_SHA='860f9d08d6013f090ae4bc710bdbbfeee06e01a0cd71f8a784b5689cdb4b7ee0'

def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def safe_member(name):
    p=PurePosixPath(name)
    if p.is_absolute() or '..' in p.parts or '\\' in name or ':' in name:raise ValueError('Unsafe archive member')
    return p

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--zip',type=Path)
    args=ap.parse_args()
    target=ROOT/'artifacts/submissions/verify_v6'
    if target.is_symlink():raise ValueError('Destination must not be a symlink')
    if args.zip:
        if sha(args.zip)!=ZIP_SHA:raise ValueError('V6 ZIP SHA mismatch; nothing extracted')
        manifest=json.loads((ROOT/'releases/v6/submission.manifest.json').read_text(encoding='utf-8-sig'))
        entries={r['path']:r for r in manifest['files']}
        with zipfile.ZipFile(args.zip) as z:
            names=[i.filename for i in z.infolist() if not i.is_dir()]
            if len(names)!=len(set(names)) or set(names)!=set(entries):raise ValueError('Unexpected ZIP members')
            for name in names:
                p=target.joinpath(*safe_member(name).parts)
                if not p.resolve().is_relative_to(target.resolve()):raise ValueError('Destination escapes release')
                if p.exists() and sha(p)!=entries[name]['sha256']:raise ValueError(f'Existing file differs, not overwritten: {name}')
            for name in names:
                p=target.joinpath(*safe_member(name).parts)
                if p.exists():continue
                p.parent.mkdir(parents=True,exist_ok=True)
                with p.open('xb') as out,z.open(name) as source:shutil.copyfileobj(source,out)
                if sha(p)!=entries[name]['sha256']:raise ValueError('Extracted file hash mismatch; keep evidence and investigate')
        print('Exact V6 release restored/verified. Stage2 still requires CUDA.')
    else:
        snapshot=ROOT/'releases/v6/source'
        entries=json.loads((ROOT/'releases/v6/source_manifest.json').read_text(encoding='utf-8'))['files']
        for r in entries:
            relative=safe_member(r['path'])
            source=snapshot.joinpath(*relative.parts);dest=target.joinpath(*relative.parts)
            if not dest.resolve().is_relative_to(target.resolve()):raise ValueError('Destination escapes release')
            if sha(source)!=r['sha256']:raise ValueError('Git source snapshot changed')
            if dest.exists() and sha(dest)!=r['sha256']:raise ValueError(f"Existing release differs: {r['path']}")
        for r in entries:
            dest=target/r['path']
            if not dest.exists():
                dest.parent.mkdir(parents=True,exist_ok=True)
                with dest.open('xb') as f:f.write((snapshot/r['path']).read_bytes())
        print('V6 source only restored. Models/data are NOT restored.')

if __name__=='__main__':main()
