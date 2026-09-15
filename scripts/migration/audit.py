"""Read-only source/config audit; findings never contain credential values."""
import argparse
import ast
import collections
import hashlib
import json
import os
import re
import subprocess
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TEXT = {'.py','.md','.txt','.json','.jsonl','.csv','.yml','.yaml','.toml','.ini','.cfg','.js','.cjs','.css','.html','.ipynb','.sh','.ps1','.bat','.cmd','.example'}
RULES = {
    'private_key': re.compile(r'-----BEGIN (?:RSA |EC |OPENSSH |DSA |ENCRYPTED )?PRIVATE KEY-----'),
    'github_token': re.compile(r'\b(?:gh[pousr]_[A-Za-z0-9]{30,255}|github_pat_[A-Za-z0-9_]{40,255})\b'),
    'huggingface_token': re.compile(r'\bhf_[A-Za-z0-9]{25,100}\b'),
    'openai_key': re.compile(r'\bsk-(?:proj-|svcacct-)?[A-Za-z0-9_-]{25,200}\b'),
    'aws_access_key': re.compile(r'\b(?:AKIA|ASIA)[A-Z0-9]{16}\b'),
    'google_key': re.compile(r'\bAIza[A-Za-z0-9_-]{30,45}\b'),
    'slack_token': re.compile(r'\bxox[baprs]-[A-Za-z0-9-]{20,200}\b'),
    'jwt': re.compile(r'\beyJ[A-Za-z0-9_-]{15,}\.[A-Za-z0-9_-]{15,}\.[A-Za-z0-9_-]{15,}\b'),
    'signed_url': re.compile(r'(?i)(?:[?&]|&amp;)(?:X-Amz-Signature|X-Goog-Signature|Signature|access_token|auth_token|token)=[A-Za-z0-9_%+/=-]{20,}'),
    'credential_assignment': re.compile(r'''(?ix)(?:api[_-]?key|access[_-]?token|auth[_-]?token|client[_-]?secret|secret[_-]?access[_-]?key|password)["']?\s*[:=]\s*["'](?!\$|\{|<|YOUR_|EXAMPLE|REDACTED|PLACEHOLDER)[A-Za-z0-9_./+@=-]{12,}["']'''),
}

def scan_text(text):
    findings=[]
    for rule, pattern in RULES.items():
        for match in pattern.finditer(text):
            findings.append({'rule':rule,'line':text.count('\n',0,match.start())+1})
    return findings

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--tracked',action='store_true')
    args=ap.parse_args()
    if args.tracked:
        result=subprocess.run(['git','-c',f'safe.directory={ROOT.as_posix()}','ls-files','-z'],cwd=ROOT,check=True,capture_output=True)
        files=[ROOT / os.fsdecode(p) for p in result.stdout.split(b'\0') if p]
    else:
        files=[]
        for current,dirs,names in os.walk(ROOT):
            dirs[:]=[d for d in dirs if d not in {'.git','.venv','venv','node_modules','__pycache__','.cache'}]
            if Path(current).is_relative_to(ROOT/'artifacts/migration'):
                dirs[:]=[]
                continue
            files.extend(Path(current)/n for n in names)
    secrets=[];windows=[];env=set();stats=collections.Counter();case_map=collections.defaultdict(list);crlf=[];links=[];large=[]
    for p in files:
        rel=p.relative_to(ROOT).as_posix()
        if p.is_symlink():links.append(rel)
        size=p.stat().st_size
        stats['files']+=1;stats['bytes']+=size
        case_map[unicodedata.normalize('NFC',rel).casefold()].append(rel)
        if size>=5_000_000:large.append({'path':rel,'bytes':size})
        if p.suffix.lower() not in TEXT and p.name not in {'.env','.gitignore','.gitattributes','LICENSE','Dockerfile'}:continue
        if size>20_000_000:stats['text_over_20MB_not_scanned']+=1;continue
        raw=p.read_bytes()
        try:text=raw.decode('utf-8-sig')
        except UnicodeDecodeError:stats['non_utf8_not_scanned']+=1;continue
        stats['text_files_scanned']+=1
        if b'\r\n' in raw:crlf.append(rel)
        for finding in scan_text(text):secrets.append({'path':rel,**finding})
        if re.search(r'(?i)\b[A-Z]:[\\/]|%USERPROFILE%|%APPDATA%',text):windows.append(rel)
        if p.suffix=='.py':
            try:tree=ast.parse(text)
            except SyntaxError:continue
            for node in ast.walk(tree):
                if isinstance(node,ast.Subscript) and isinstance(node.value,ast.Attribute) and node.value.attr=='environ' and isinstance(node.slice,ast.Constant):env.add(str(node.slice.value))
                if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute) and node.func.attr in {'getenv','get','setdefault'} and node.args and isinstance(node.args[0],ast.Constant):
                    prefix=ast.get_source_segment(text,node.func) or ''
                    if 'environ' in prefix or 'getenv' in prefix:env.add(str(node.args[0].value))
    report={'scope':'tracked_worktree' if args.tracked else 'project_without_environments_and_binary_caches',
            'statistics':dict(stats),'credential_findings':secrets,'windows_path_files':windows,
            'environment_names_only':sorted(env),'case_or_unicode_collisions':[v for v in case_map.values() if len(v)>1],
            'symlinks':links,'crlf_text_files':crlf,'large_files':large,
            'limitations':'Pattern-based detection is not a proof that all possible secrets are absent. No values emitted; dependencies/binaries and files over20MB are excluded from content scan.'}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x',encoding='utf-8') as f:json.dump(report,f,ensure_ascii=False,indent=2)
    print(json.dumps({'scope':report['scope'],'stats':dict(stats),'findings_count':len(secrets),'finding_locations':secrets[:35],'case_collisions':len(report['case_or_unicode_collisions'])},ensure_ascii=True))

if __name__=='__main__':main()
