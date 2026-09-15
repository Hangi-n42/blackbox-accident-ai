"""Root-executed, fixed three-source download. No image inspection or labels.

Run: .venv/Scripts/python.exe -I -B research/v6/nexar_review_round2/download_sources.py
Refuses every existing output, including partial attempts; no automatic retries.
"""
import sys,os
sys.dont_write_bytecode=True
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='2'
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,math
import requests
import av

OUT=Path(__file__).resolve().parent
OLD=OUT.parent/'nexar_review_candidates'
MANIFEST=OUT/'manifest.json'
EXPECTED_MANIFEST_SHA='26f9316348a1f24f93714856afca5f1205dd90528e669f1024170ad67ac91aad'
REV='aa97deda5a59f00bb7187739053b7c72e14374df'
REPO='nexar-ai/nexar_collision_prediction'
EXPECTED_NAMES=['00008.mp4','00010.mp4','00013.mp4']

def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def write_new(path,value):
    with Path(path).open('x',encoding='utf8') as f:json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False)

def main():
    assert sha(MANIFEST)==EXPECTED_MANIFEST_SHA,'Frozen source manifest changed'
    plan=json.loads(MANIFEST.read_text(encoding='utf8'))
    assert plan['revision']==REV and plan['repository']==REPO and plan['budget_passed'] is True
    selected=plan['selected'];assert [Path(x['path']).name for x in selected]==EXPECTED_NAMES
    total=sum(x['bytes'] for x in selected)
    assert total==plan['total_bytes']==42322975 and total<=60000000
    for item in selected:
        name=Path(item['path']).name
        assert item['path']==f'train/positive/{name}'
        assert item['source_uri']==f'https://huggingface.co/datasets/{REPO}/resolve/{REV}/{item["path"]}'
    outputs=[OUT/'download_plan.json',OUT/'acquisition.json',OUT/'LICENSE',OUT/'README.md',OUT/'ATTRIBUTION.json']
    for name in EXPECTED_NAMES:
        p=OUT/name;outputs.extend([p,p.with_suffix('.part'),p.with_suffix('.mapping.json'),OUT/f'acquisition_{p.stem}.json'])
    assert not any(p.exists() for p in outputs),'Existing output/partial attempt: refusing overwrite or retry'
    bindings={str(MANIFEST):sha(MANIFEST),str(Path(__file__)):sha(__file__)}
    for name in ('LICENSE','README.md'):bindings[str(OLD/name)]=sha(OLD/name)
    write_new(OUT/'download_plan.json',dict(created_utc=datetime.now(timezone.utc).isoformat(),files_sha256=bindings,
      revision=REV,selected=selected,total_bytes=total,budget_bytes=60000000,
      scope='Only three fixed MP4 URLs. Copy existing same-revision attribution locally. Decode native PTS without exporting/viewing pixels. No model/GPU/labels requests.',
      overwrite=False,automatic_retry=False,failure_policy='Preserve partial bytes and completed per-source records; do not silently select replacement sources.'))
    records=[];active=None
    report=dict(status='in_progress',revision=REV,repository=REPO,manifest_sha256=EXPECTED_MANIFEST_SHA,
      records=records,video_requests=0,labels_loaded=False,model_calls=0,images_exported=0,ground_truth_eligible=False,
      source_incident_independence_verified=False)
    try:
        copied={}
        for name in ('LICENSE','README.md'):
            payload=(OLD/name).read_bytes()
            with (OUT/name).open('xb') as f:f.write(payload)
            assert sha(OUT/name)==bindings[str(OLD/name)]
            copied[name]=dict(copied_from=str(OLD/name),sha256=sha(OUT/name),same_revision_prior_source=True)
        write_new(OUT/'ATTRIBUTION.json',dict(repository=REPO,revision=REV,files=copied,
          note='Original source README/LICENSE copied byte-for-byte from the previously acquired same revision. The positive directory is not contact ground truth.'))
        with requests.Session() as session:
            for item in selected:
                path=OUT/Path(item['path']).name;partial=path.with_suffix('.part');active=path.name
                size=0;digest=hashlib.sha256();report['video_requests']+=1
                with session.get(item['source_uri'],stream=True,timeout=(15,60)) as response:
                    response.raise_for_status();assert response.status_code==200,'Expected complete source response'
                    with partial.open('xb') as f:
                        for chunk in response.iter_content(1024*1024):
                            if not chunk:continue
                            size+=len(chunk)
                            if size>item['bytes']:raise ValueError('Payload exceeds declared size; stop before writing oversized chunk')
                            f.write(chunk);digest.update(chunk)
                assert size==item['bytes'] and digest.hexdigest()==item['sha256'],'Downloaded size/SHA mismatch'
                # Windows rename refuses a destination created concurrently; no replace API.
                assert not path.exists();partial.rename(path)
                frames=[];native=[]
                with av.open(str(path)) as container:
                    stream=container.streams.video[0];stream.thread_count=2
                    for number,frame in enumerate(container.decode(stream)):
                        assert frame.pts is not None and frame.time_base is not None,'Missing native PTS'
                        seconds=float(frame.pts*frame.time_base)
                        assert math.isfinite(seconds) and seconds>=0 and (not frames or seconds>frames[-1]['pts_seconds']),'Nonmonotonic/nonfinite native PTS'
                        frames.append(dict(frame=number,pts_seconds=seconds))
                        native.append(dict(frame=number,native_pts=frame.pts,time_base=[frame.time_base.numerator,frame.time_base.denominator]))
                assert frames and sha(path)==item['sha256']
                mapping=path.with_suffix('.mapping.json')
                write_new(mapping,dict(frame_pts=frames,native_frames=native,source_video_sha256=item['sha256'],
                  time_origin='native source presentation timestamp seconds, no offset applied',method='PyAV native frame.pts * frame.time_base'))
                record=dict(path=str(path),source_uri=item['source_uri'],sha256=item['sha256'],bytes=size,frames=len(frames),
                  first_pts=frames[0]['pts_seconds'],last_pts=frames[-1]['pts_seconds'],mapping_sha256=sha(mapping),
                  human_review='pending',contact_ground_truth=None,source_group_id=None,independence_verified=False)
                records.append(record);write_new(OUT/f'acquisition_{path.stem}.json',record)
                print(json.dumps(dict(downloaded=path.name,bytes=size,frames=len(frames))),flush=True)
        assert all(sha(p)==h for p,h in bindings.items())
        report.update(status='three_round2_sources_acquired_not_validated',bytes=sum(r['bytes'] for r in records),input_assets_unchanged=True)
    except BaseException as error:
        report.update(status='failed_preserve_partial_do_not_retry_automatically',active_source=active,error=repr(error))
        raise
    finally:
        report['finished_utc']=datetime.now(timezone.utc).isoformat();write_new(OUT/'acquisition.json',report)
    print(json.dumps(dict(status=report['status'],source_count=len(records),bytes=report['bytes'])),flush=True)

if __name__=='__main__':main()
