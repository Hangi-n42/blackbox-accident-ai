import sys,json,zipfile,hashlib,time
from pathlib import Path
ROOT=Path.cwd();sys.path.insert(0,str(ROOT/'research'))
from acquire_comma_subset import RemoteZipFile
out=ROOT/'artifacts/stage1_road_data_search_20260918'
url='https://huggingface.co/datasets/snah/REDS/resolve/main/train_sharp.zip?download=true&probe=20260918road'
r=RemoteZipFile(url,budget=20_000_000)
with zipfile.ZipFile(r) as z:
    names=z.namelist();(out/'reds_train_index.json').write_text(json.dumps(names))
    print('archive',r.size,'members',len(names),'examples',names[:8],flush=True)
    rows=[]
    for seq in [2,16,45]:
        candidates=[n for n in names if n.endswith('.png') and f'/{seq:03d}/' in n]
        assert candidates,seq
        name=sorted(candidates)[0];p=out/f'reds_{seq:03d}_first.png';p.write_bytes(z.read(name))
        rows.append({'sequence':seq,'member':name,'path':str(p.relative_to(ROOT)),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'source_url':url})
        print(name,flush=True)
    (out/'reds_source_probe.json').write_text(json.dumps({'frames':rows,'downloaded_bytes':r.used},indent=2))
