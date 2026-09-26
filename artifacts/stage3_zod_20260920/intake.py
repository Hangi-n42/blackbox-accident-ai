"""Read the user-authorized ZOD share using the public devkit's download identity."""
from pathlib import Path
import ast,json,requests,sys
O=Path(__file__).resolve().parent
URL='https://www.dropbox.com/scl/fo/q81qqpiqygaeys7mppgoe/AFuqa-QrSkGzHmnkhhpvbBE?rlkey=t6k2mq2lcgzdla1fvvjw8yj4a&dl=0'
def session():
    tree=ast.parse((O/'vendor/download.py').read_text());cfg={}
    for n in tree.body:
        if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name) and n.targets[0].id in ['APP_KEY','REFRESH_TOKEN']:cfg[n.targets[0].id]=ast.literal_eval(n.value)
    r=requests.post('https://api.dropboxapi.com/oauth2/token',data={'grant_type':'refresh_token','refresh_token':cfg['REFRESH_TOKEN'],'client_id':cfg['APP_KEY']},timeout=30)
    r.raise_for_status();s=requests.Session();s.headers['Authorization']='Bearer '+r.json()['access_token'];return s

def catalog(subset):
    s=session();r=s.post('https://api.dropboxapi.com/2/files/list_folder',json={'path':'/'+subset,'shared_link':{'url':URL}},timeout=30)
    if r.status_code!=200:raise RuntimeError(f'list_folder status {r.status_code}: '+r.text[:300])
    data=r.json();entries=data['entries']
    while data['has_more']:
        r=s.post('https://api.dropboxapi.com/2/files/list_folder/continue',json={'cursor':data['cursor']},timeout=30);r.raise_for_status();data=r.json();entries+=data['entries']
    out=[{k:e.get(k) for k in ['.tag','name','size','content_hash','server_modified']} for e in entries]
    (O/f'{subset}_catalog.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
if __name__=='__main__':catalog(sys.argv[1] if len(sys.argv)>1 else 'sequences')

def download(name):
    import hashlib,time
    catalog=json.loads((O/'sequences_catalog.json').read_text());entry=next(x for x in catalog if x['name']==name)
    dst=O/'archives'/name;dst.parent.mkdir(exist_ok=True)
    if dst.exists():raise FileExistsError(dst)
    s=session();r=s.post('https://content.dropboxapi.com/2/sharing/get_shared_link_file',headers={'Dropbox-API-Arg':json.dumps({'url':URL,'path':'/sequences/'+name})},stream=True,timeout=(30,90));r.raise_for_status();n=0;t=time.monotonic();last=t
    with dst.with_suffix(dst.suffix+'.partial').open('xb') as f:
        for b in r.iter_content(4*1024*1024):
            f.write(b);n+=len(b)
            if n>entry['size']:raise ValueError('size overflow')
            if time.monotonic()-last>20:print(name,n,'/',entry['size'],flush=True);last=time.monotonic()
    r.close();assert n==entry['size'];tmp=dst.with_suffix(dst.suffix+'.partial');blockhash=hashlib.sha256();full=hashlib.sha256()
    with tmp.open('rb') as f:
        while b:=f.read(4*1024*1024):blockhash.update(hashlib.sha256(b).digest());full.update(b)
    assert blockhash.hexdigest()==entry['content_hash'];tmp.rename(dst)
    (O/(name+'.receipt.json')).write_text(json.dumps({'bytes':n,'sha256':full.hexdigest(),'dropbox_content_hash':blockhash.hexdigest(),'verified':True,'seconds':time.monotonic()-t},indent=2));print('verified',name,n,flush=True)
