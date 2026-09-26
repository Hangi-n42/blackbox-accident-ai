"""Fetch clean counterparts using the provider's scene-to-filename mapping."""
import hashlib,json,sys,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'research'))
from acquire_comma_subset import RemoteZipFile
OUT=ROOT/'artifacts/data_pilot_20260916/vdmoire'
URL='https://www.dropbox.com/scl/fo/43k8ye706fdc973c1jrqb/APb4FVqdzo_ILHk5MTOUv7c/iphone.zip?rlkey=an6mb1r2h3en5kjuiui221anq&dl=1'

def main():
    raw=json.loads((OUT/'manifest.json').read_text())['frames']
    groups=sorted({r['source_group'] for r in raw})
    folders=['Reds','Moca','landscape','sports','daily','house','text','animal']
    remote=RemoteZipFile(URL,budget=70000000);rows=[]
    with zipfile.ZipFile(remote) as archive:
        names=set(archive.namelist())
        for group in groups:
            _,_,scene,video=group.split('/');k=folders.index(scene);number=int(video.split('_')[-1])
            prefix=f'v1{k}{number:03d}' if scene=='Reds' else f'v2{k-1}{number:03d}'
            for frame in (0,59):
                member=f'iphone/train/target/{prefix}_{frame:05d}.jpg'
                assert member in names,member
                path=OUT/'clean'/Path(member).name;path.parent.mkdir(exist_ok=True)
                if not path.exists():path.write_bytes(archive.read(member))
                rows.append({'source_group':group,'member':member,'path':str(path.relative_to(ROOT)),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'provider_role':'clean target','temporal_alignment_to_raw':'not verified; group correspondence only'})
            (OUT/'clean_manifest.json').write_text(json.dumps({'source_url':URL,'mapping_evidence':'official dataset_prepare/data_prepare.py folders and filename formulas','frames':rows,'downloaded_bytes':remote.used},indent=2))
            print(group,flush=True)
    assert len(rows)==40
    print('complete',len(rows),remote.used,flush=True)
if __name__=='__main__':main()
