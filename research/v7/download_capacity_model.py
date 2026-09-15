"""Pinned official 8B assets; bounded range download, verified assembly, no execution."""
import json,hashlib,time,shutil,threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor,as_completed
import requests
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
OUT=ROOT/'artifacts/candidates/qwen3_vl_8b_v7'
REV='0c351dd01ed87e9c1b53cbc748cba10e6187ff3b'
def digest(path):
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def put(path,obj):
    with path.open('x',encoding='utf8') as f:json.dump(obj,f,ensure_ascii=False,indent=2)
def main():
    source=HERE/'model_capacity_source_manifest.json';plan=json.loads(source.read_text('utf8'))
    assert plan['repository']=='Qwen/Qwen3-VL-8B-Instruct' and plan['revision']==REV and plan['license']=='apache-2.0'
    selected=[x for x in plan['files'] if x['name']!='.gitattributes']
    total=sum(x['bytes'] for x in selected)
    assert total<18000000000 and shutil.disk_usage(ROOT).free>45000000000
    assert not OUT.exists();OUT.mkdir()
    put(OUT/'acquisition_plan.json',dict(source_manifest_sha256=digest(source),script_sha256=digest(Path(__file__)),selected=selected,expected_bytes=total,wall_budget_seconds=1800,workers=4,range_bytes=16777216,max_attempts_per_part=2,role='Official model capacity experiment; no adoption or GPU execution',disk_free_before=shutil.disk_usage(ROOT).free))
    started=time.monotonic();records=[];lock=threading.Lock();wire=[0]
    report=dict(repository=plan['repository'],revision=REV,source=f'https://huggingface.co/{plan["repository"]}',status='running',files=records)
    def check_time():
        if time.monotonic()-started>1800:raise TimeoutError('Frozen acquisition wall budget exceeded')
    def receive(url,destination,expected,headers=None):
        check_time();size=0
        with requests.get(url,headers=headers,stream=True,timeout=(20,60)) as r:
            r.raise_for_status()
            if headers:
                assert r.status_code==206 and r.headers.get('Content-Range')==headers['Expected-Content-Range']
            else:assert r.status_code==200
            with destination.open('xb') as f:
                for chunk in r.iter_content(1024*1024):
                    if not chunk:continue
                    check_time();size+=len(chunk);assert size<=expected
                    with lock:
                        wire[0]+=len(chunk)
                        if wire[0]>36000000000:raise ValueError('Transfer budget exceeded')
                    f.write(chunk)
        assert size==expected
    try:
        for item in selected:
            name=item['name'];assert '/' not in name and '\\' not in name
            assert item['url']==f'https://huggingface.co/{plan["repository"]}/resolve/{REV}/{name}'
            path=OUT/name
            if name.endswith('.safetensors'):
                parts=OUT/(name+'.parts');parts.mkdir();size=item['bytes'];step=16777216;count=(size+step-1)//step
                def part(i):
                    start=i*step;end=min(size,start+step)-1;last=None
                    for attempt in range(2):
                        dest=parts/f'{i:04d}.{attempt}.part'
                        try:
                            receive(item['url']+f'?download=true&v7part={i}&attempt={attempt}',dest,end-start+1,{'Range':f'bytes={start}-{end}','Expected-Content-Range':f'bytes {start}-{end}/{size}'})
                            return i,dest
                        except Exception as e:last=e
                    raise last
                completed={}
                with ThreadPoolExecutor(max_workers=4) as pool:
                    futures=[pool.submit(part,i) for i in range(count)]
                    for n,future in enumerate(as_completed(futures),1):
                        i,dest=future.result();completed[i]=dest
                        if n%32==0 or n==count:print(json.dumps(dict(file=name,parts_complete=n,parts_total=count,seconds=round(time.monotonic()-started))),flush=True)
                assembling=path.with_suffix('.assembling')
                with assembling.open('xb') as f:
                    for i in range(count):
                        with completed[i].open('rb') as src:shutil.copyfileobj(src,f,1024*1024)
                assert assembling.stat().st_size==size and digest(assembling)==item['lfs_sha256']
                assembling.rename(path)
                # Only our redundant fragments, after verified complete assembly.
                for fragment in parts.iterdir():
                    assert fragment.resolve().is_relative_to(OUT.resolve()) and fragment.is_file() and fragment.name.endswith('.part')
                    fragment.unlink()
                parts.rmdir()
            else:
                receive(item['url'],path,item['bytes'])
                payload=path.read_bytes();blob=hashlib.sha1(b'blob '+str(len(payload)).encode()+b'\0'+payload).hexdigest()
                assert blob==item['git_blob_id'],'Pinned Git blob mismatch'
            record=dict(name=name,bytes=path.stat().st_size,sha256=digest(path),expected_lfs_sha256=item['lfs_sha256'])
            records.append(record);print(json.dumps(dict(verified=name,bytes=record['bytes'])),flush=True)
        report['status']='complete'
    except BaseException as e:report.update(status='failed_preserved',error=repr(e));raise
    finally:
        report.update(seconds=time.monotonic()-started,received_bytes=wire[0],weights_executed=False)
        put(OUT/'download_manifest.json',report)
if __name__=='__main__':main()
