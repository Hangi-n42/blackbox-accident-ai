"""Read six original recapture frames from each of 20 source groups via ZIP ranges."""
import hashlib,json,sys,zipfile
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'research'))
from acquire_comma_subset import RemoteZipFile
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'artifacts/data_pilot_20260916/vdmoire'
URL='https://www.dropbox.com/scl/fo/bxelqebf0241n1a3z13dq/AE-NbNxIACNu6jtmcXRFNqs/iphone.zip?rlkey=2f1us04s3ytbf05kfflet1dqg&dl=1'

def main():
    OUT.mkdir(exist_ok=True)
    remote=RemoteZipFile(URL,budget=150000000)
    with zipfile.ZipFile(remote) as archive:
        groups={}
        for name in archive.namelist():
            if name.startswith('frames/train/') and name.endswith('.jpg') and 'not_used' not in name:
                groups.setdefault(str(Path(name).parent),[]).append(name)
        selected=sorted(groups,key=lambda s:hashlib.sha256(s.encode()).hexdigest())[:20]
        assert len(selected)==20
        rows=[]
        for group in selected:
            names=sorted(groups[group]);chosen=[names[round(i*(len(names)-1)/5)] for i in range(6)]
            for name in chosen:
                target=OUT/name;target.parent.mkdir(parents=True,exist_ok=True)
                if not target.exists():target.write_bytes(archive.read(name))
                rows.append({'source_group':group,'member':name,'path':str(target.relative_to(ROOT)),
                             'sha256':hashlib.sha256(target.read_bytes()).hexdigest(),'role':'recapture_only_qa','device':'iphone','paired_original':None})
            (OUT/'manifest.json').write_text(json.dumps({'source_url':URL,'frames':rows,'downloaded_bytes':remote.used,'status':'QA only; corresponding clean sources not acquired; no temporal FPS assumed'},indent=2))
            print(group,flush=True)
    print('frames',len(rows),'downloaded_bytes',remote.used,flush=True)
if __name__=='__main__':main()
