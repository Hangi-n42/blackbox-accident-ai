"""Download only the six frozen public source objects; never inspect or infer."""
import hashlib,json,shutil,time
from pathlib import Path
import requests
H=Path(__file__).resolve().parent;OUT=H/'new_validation_sources'
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def put(p,v):
    with p.open('x',encoding='utf8') as f:json.dump(v,f,ensure_ascii=False,indent=2)
def main():
    plan=json.loads((OUT/'selection_plan.json').read_text('utf8'))
    assert not (OUT/'acquisition.json').exists() and not list(OUT.glob('*.mp4')) and not list(OUT.glob('*.part'))
    assert len(plan['selected'])==6 and sum(x['bytes'] for x in plan['selected'])==plan['expected_bytes']<120000000
    for path,digest in plan['source_bindings'].items():assert sha(path)==digest
    for name in ('LICENSE','README.md','ATTRIBUTION.json'):
        source=next(Path(p) for p in plan['source_bindings'] if Path(p).name==name)
        shutil.copyfile(source,OUT/name)
    started=time.monotonic();report={'status':'running','records':[],'selection_plan_sha256':sha(OUT/'selection_plan.json'),'script_sha256':sha(Path(__file__)),'model_calls':0,'label_reads':0}
    try:
        for item in plan['selected']:
            assert time.monotonic()-started<600
            assert item['url']==f'https://huggingface.co/datasets/{plan["repository"]}/resolve/{plan["revision"]}/{item["relative_path"]}'
            assert item['relative_path']==f'train/positive/{item["ID"]}.mp4'
            partial=OUT/(item['ID']+'.part');final=OUT/(item['ID']+'.mp4');size=0
            with requests.get(item['url'],stream=True,timeout=(15,60)) as response:
                response.raise_for_status();assert response.status_code==200
                with partial.open('xb') as f:
                    for chunk in response.iter_content(1048576):
                        if not chunk:continue
                        assert time.monotonic()-started<600
                        size+=len(chunk);assert size<=item['bytes'];f.write(chunk)
            assert size==item['bytes'] and sha(partial)==item['sha256']
            partial.rename(final)
            report['records'].append({**item,'path':str(final)})
            print(json.dumps({'ID':item['ID'],'verified_bytes':size}),flush=True)
        report['status']='complete'
    except BaseException as error:report.update(status='failed_preserved',error=repr(error));raise
    finally:
        report['seconds']=time.monotonic()-started
        put(OUT/'acquisition.json',report)
if __name__=='__main__':main()
