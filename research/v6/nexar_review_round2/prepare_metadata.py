"""Metadata only: fixed revision, three next unused positive MP4 sources."""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
import hashlib,json,datetime,re
import requests
OUT=Path(__file__).resolve().parent
OLD=OUT.parent/'nexar_review_candidates'
REV='aa97deda5a59f00bb7187739053b7c72e14374df'
REPO='nexar-ai/nexar_collision_prediction'
PREFIX=f'https://huggingface.co/api/datasets/{REPO}/tree/{REV}/train/positive'
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def write(name,value):
    with (OUT/name).open('x',encoding='utf8') as f:json.dump(value,f,indent=2,ensure_ascii=False)
def main():
    old=json.loads((OLD/'acquisition.json').read_text(encoding='utf8'));prior=json.loads((OLD/'plan.json').read_text(encoding='utf8'))
    assert prior['revision']==REV
    used={Path(x['path']).name:x['sha256'] for x in old['records']}
    assert set(used)=={'00000.mp4','00003.mp4','00004.mp4','00005.mp4','00006.mp4','00007.mp4'}
    hashes={str(p):sha(p) for p in [OLD/'acquisition.json',OLD/'plan.json',Path(__file__)]}
    for name,h in used.items():assert sha(OLD/name)==h;hashes[str(OLD/name)]=h
    write('selection_policy_frozen.json',dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),revision=REV,repository=REPO,
      metadata_endpoint=PREFIX,inputs_sha256=hashes,used_sources=used,source_count=3,budget_bytes=60000000,
      rule='Fetch complete positive-directory metadata, sort MP4 paths lexicographically; exclude previously used filenames and any matching previously used SHA256; select next three unique SHA256 sources. No size-driven substitution.',
      metadata_limit_pages=10,metadata_limit_bytes=5000000,scope='Metadata only; no MP4 or labels.json read/download; no GPU/model/GT',
      independence='New exact object SHA only; same-incident/perceptual/pretraining independence unverified. Human review required.'))
    entries=[];pages=[];url=PREFIX+'?limit=1000';size=0
    session=requests.Session();session.trust_env=True
    try:
        while url:
            assert url.startswith(PREFIX+'?') and len(pages)<10
            response=session.get(url,timeout=(15,45),allow_redirects=False);response.raise_for_status()
            assert response.status_code==200
            size+=len(response.content);assert size<=5000000
            data=response.json();assert isinstance(data,list)
            entries.extend(data)
            pages.append(dict(url=url,bytes=len(response.content),sha256=hashlib.sha256(response.content).hexdigest(),count=len(data),link=response.headers.get('Link')))
            write(f'api_page_{len(pages):02d}.json',data)
            url=response.links.get('next',{}).get('url')
        eligible=sorted((x for x in entries if x['type']=='file' and x['path'].endswith('.mp4')),key=lambda x:x['path'])
        assert len({x['path'] for x in eligible})==len(eligible)
        selected=[];excluded=[];seen=set(used.values())
        for x in eligible:
            name=Path(x['path']).name;digest=x.get('lfs',{}).get('oid')
            assert digest and re.fullmatch('[0-9a-f]{64}',digest) and x['lfs']['size']==x['size']
            if name in used:excluded.append(dict(path=x['path'],reason='used_filename',sha256=digest));assert digest==used[name];continue
            if digest in seen:excluded.append(dict(path=x['path'],reason='duplicate_SHA256',sha256=digest));continue
            selected.append(dict(path=x['path'],bytes=x['size'],sha256=digest,git_blob_oid=x['oid'],xet_hash=x.get('xetHash'),
              source_uri=f'https://huggingface.co/datasets/{REPO}/resolve/{REV}/{x["path"]}',downloaded=False,human_review='not_started',independence_verified=False))
            seen.add(digest)
            if len(selected)==3:break
        assert len(selected)==3
        total=sum(x['bytes'] for x in selected)
        assert all(sha(p)==h for p,h in hashes.items())
        result=dict(status='metadata_selected_within_budget' if total<=60000000 else 'metadata_selected_over_budget_stop_no_substitution',revision=REV,repository=REPO,
          selected=selected,total_bytes=total,total_decimal_MB=total/1e6,total_MiB=total/1024**2,budget_bytes=60000000,budget_passed=total<=60000000,
          excluded_before_selection=excluded,metadata_pages=pages,metadata_bytes=size,eligible_mp4_count=len(eligible),
          input_assets_unchanged=True,downloaded_video_count=0,labels_json_reads=0,model_calls=0,
          note='Filename order convenience cohort; previously unseen bytes do not establish source/incident independence. No contact GT inferred from positive directory.')
        write('manifest.json',result);print(json.dumps(result,ensure_ascii=False),flush=True)
    except Exception as e:
        write('metadata_access_failure.json',dict(error=repr(e),completed_pages=len(pages),metadata_bytes=size,video_downloads=0));raise
if __name__=='__main__':main()
