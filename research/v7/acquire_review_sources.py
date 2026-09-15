"""Acquire three preselected public videos, no labels/model outputs/pixel inspection."""
import json,hashlib,shutil,requests,sys
from pathlib import Path
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parent/('fresh_sources_retry1' if sys.argv[1:]==['--retry-network-permission'] else 'fresh_sources')
REV='aa97deda5a59f00bb7187739053b7c72e14374df'
ITEMS=[('00014',12482330,'90f401dbf91ead5098e1629ebf384036e0d2ca7943269fd2b48327b3a558c7ee'),('00015',8094504,'929e9e6f2494f30c47f5567e95412d168b458336f31024ab1cd6dea4c9b2cb1a'),('00016',28279207,'87b96caf17ebd8a76ee7ccc93aec6f5932def39fd5f6d2ccaf65690b72e0f67e')]
def put(p,v):
    with p.open('x',encoding='utf8') as f:json.dump(v,f,ensure_ascii=False,indent=2)
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def main():
    if sys.argv[1:]:
        assert sys.argv[1:]==['--retry-network-permission']
        previous=OUT.parent/'fresh_sources'
        prior=json.loads((previous/'acquisition.json').read_text(encoding='utf8'))
        assert prior['status']=='failed_partial_preserved' and prior['records']==[]
        assert 'WinError 10013' in prior['error'] and not list(previous.glob('*.part')) and not list(previous.glob('*.mp4'))
    assert not OUT.exists(),'Preserve existing attempts'
    assert sum(x[1] for x in ITEMS)==48856041<60000000
    OUT.mkdir()
    records=[]
    plan=dict(revision=REV,selected=ITEMS,created_utc=datetime.now(timezone.utc).isoformat(),protocol_sha256=sha(OUT.parent/'experiment_protocol.json'),role='AI secondary review; not human GT',script_sha256=sha(Path(__file__)),automatic_retry=False)
    put(OUT/'plan.json',plan)
    for name in ('LICENSE','README.md','ATTRIBUTION.json'):
        shutil.copyfile(ROOT/'research/v6/nexar_review_round2'/name,OUT/name)
    report=dict(status='running',records=records)
    try:
        for ID,size,expected in ITEMS:
            url=f'https://huggingface.co/datasets/nexar-ai/nexar_collision_prediction/resolve/{REV}/train/positive/{ID}.mp4'
            path=OUT/f'{ID}.mp4';partial=OUT/f'{ID}.part';n=0;digest=hashlib.sha256()
            with requests.get(url,stream=True,timeout=(15,60)) as response:
                response.raise_for_status();assert response.status_code==200
                with partial.open('xb') as f:
                    for chunk in response.iter_content(1024*1024):
                        if not chunk:continue
                        n+=len(chunk);assert n<=size
                        f.write(chunk);digest.update(chunk)
            assert n==size and digest.hexdigest()==expected
            partial.rename(path)
            records.append(dict(ID=ID,path=str(path),url=url,bytes=n,sha256=expected))
            print(json.dumps(records[-1]),flush=True)
        report['status']='complete'
    except BaseException as e:report.update(status='failed_partial_preserved',error=repr(e));raise
    finally:put(OUT/'acquisition.json',report)
if __name__=='__main__':main()
