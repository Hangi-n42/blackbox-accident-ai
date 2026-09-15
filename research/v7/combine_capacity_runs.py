"""Combine disjoint, identically frozen capacity batches without rerunning clips."""
import argparse, hashlib, json
from pathlib import Path

def read(p): return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):
    with p.open('rb') as f: return hashlib.file_digest(f,'sha256').hexdigest()

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--runs',nargs='+',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists(): raise ValueError('Preserve existing output')
    videos=[]; bindings={}; batches=[]; policy=None; runtimes=[]
    for folder in a.runs:
        report=read(folder/'report.json'); freeze=read(folder/'freeze.json')
        if report['status']!='complete' or report['network_attempts']!=0:
            raise ValueError('Only complete offline batches may be combined')
        current=(freeze['candidate'],report['runtime']['precision'],
                 report['runtime']['compute_dtype'],report['runtime']['double_quant'],
                 report.get('embedding_policy'))
        if policy is not None and current!=policy: raise ValueError('Runtime policy differs')
        policy=current
        runtimes.append(report['runtime'])
        for path,digest in freeze['files'].items():
            if path in bindings and bindings[path]!=digest: raise ValueError('Frozen file changed between batches')
            bindings[path]=digest
        videos.extend(report['videos'])
        batches.append({'run':str(folder.resolve()),'report_sha256':sha(folder/'report.json'),
                        'freeze_sha256':sha(folder/'freeze.json')})
    ids=[r['ID'] for r in videos]
    if len(ids)!=9 or set(ids)!={'00000','00003','00004','00005','00006','00007','00008','00010','00013'}:
        raise ValueError('Exactly nine unique frozen development clips required')
    # Each batch must bind the same inference runner and experiment protocol.
    frozen=[read(f/'freeze.json')['files'] for f in a.runs]
    for suffix in ('run_capacity.py','capacity_experiment_protocol.json'):
        matching=[{k:v for k,v in f.items() if k.endswith(suffix)} for f in frozen]
        if not matching[0] or any(m!=matching[0] for m in matching):
            raise ValueError('Different/missing runner or protocol')
    a.output.mkdir(parents=True)
    combined={'status':'complete','network_attempts':0,'source_role':'exposed_development',
              'batches':batches,'videos':sorted(videos,key=lambda x:x['ID']),
              'combiner_sha256':sha(Path(__file__)),'all_input_bindings':bindings,
              'runtime':runtimes[0],'batch_runtimes':runtimes,'embedding_policy':policy[-1]}
    combined_freeze={'candidate':policy[0],'files':bindings,'ids':sorted(ids),
                     'source_role':'exposed_development','batches':batches,
                     'note':'Disjoint pre-existing batches, no new inference; runtime stores first batch and all batch runtimes separately.'}
    (a.output/'freeze.json').write_text(json.dumps(combined_freeze,ensure_ascii=False,indent=2),encoding='utf8')
    (a.output/'report.json').write_text(json.dumps(combined,ensure_ascii=False,indent=2),encoding='utf8')
    print(json.dumps({'status':'complete','ids':sorted(ids),'batches':len(batches)}))

if __name__=='__main__': main()
