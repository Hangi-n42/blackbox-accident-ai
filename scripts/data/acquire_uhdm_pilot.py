"""Read a bounded prefix of the official training archive for pilot inspection."""
import hashlib,html,json,re,tarfile
from pathlib import Path
import requests
OUT=Path(__file__).resolve().parents[2]/'artifacts/data_pilot_20260916/uhdm'
OUT.mkdir(exist_ok=True)
s=requests.Session()
page=s.get('https://drive.google.com/uc',params={'export':'download','id':'1XgWj0zPqUm-o83jMoBtvcZoZSCBNemEG'},timeout=30).text
form=re.search(r'<form[^>]+action="([^"]+)"',page)
if not form: raise RuntimeError('Official download confirmation unavailable')
fields=dict(re.findall(r'<input[^>]+name="([^"]+)"[^>]+value="([^"]*)"',page))
records=[]
with s.get(html.unescape(form[1]),params=fields,headers={'Range':'bytes=0-268435455'},stream=True,timeout=60) as r:
    r.raise_for_status()
    if 'text/html' in r.headers.get('content-type',''):
        detail=r.text
        (OUT/'pilot_download_error.html').write_text(detail)
        raise RuntimeError(f"Download returned HTML: {re.findall(r'<title>(.*?)</title>',detail)}")
    try:
        with tarfile.open(fileobj=r.raw,mode='r|gz') as archive:
            for item in archive:
                if not item.isfile() or not item.name.endswith('.jpg'):continue
                path=OUT/'images'/item.name
                if not path.resolve().is_relative_to((OUT/'images').resolve()):raise ValueError(item.name)
                data=archive.extractfile(item).read();path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(data)
                records.append({'member':item.name,'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()})
                (OUT/'pilot_manifest.json').write_text(json.dumps({'source_id':'1XgWj0zPqUm-o83jMoBtvcZoZSCBNemEG','selection':'first 40 JPEG members; archive order, not representative random sample','files':records},indent=2))
                print(item.name,len(data),flush=True)
                if len(records)>=40:break
    except (tarfile.ReadError,EOFError) as error:
        print('Bounded prefix ended:',error,flush=True)
print('Saved JPEGs:',len(records),flush=True)
