"""Frozen small Nexar intake, reusing prior native-PTS/hash contract."""
import csv,hashlib,json,time
from pathlib import Path
import requests,av
from PIL import Image,ImageDraw
OUT=Path(__file__).resolve().parent;ROOT=OUT.parents[2]
def put(name,value):
    (OUT/name).write_text(json.dumps(value,ensure_ascii=False,indent=2))
def digest(path):
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def sheet(name,images):
    canvas=Image.new('RGB',(1280,203*((len(images)+3)//4)))
    for k,(index,pts,im) in enumerate(images):
        im.thumbnail((320,180));tile=Image.new('RGB',(320,203));tile.paste(im,(0,0))
        ImageDraw.Draw(tile).text((3,184),f'{name} f{index} PTS {pts:.6f}',fill='white')
        canvas.paste(tile,((k%4)*320,(k//4)*203))
    canvas.save(OUT/f'{name}.jpg',quality=95)
plan=json.loads((OUT/'selection_round2.json').read_text());assert len(plan['selected'])==2 and plan['expected_bytes']<100000000
metadata={Path(x['file_name']).stem:x for x in csv.DictReader((ROOT/'artifacts/data_pilot_20260916/nexar/provider_metadata.csv').open())}
report={'status':'running','selection_sha256':digest(OUT/'selection_round2.json'),'script_sha256':digest(Path(__file__)),'records':[],'model_calls':0,'human_reviews':0};start=time.monotonic()
try:
 for x in plan['selected']:
    sid=Path(x['path']).stem;path=OUT/f'{sid}.mp4';assert not path.exists()
    url=f'https://huggingface.co/datasets/{plan["repository"]}/resolve/{plan["revision"]}/{x["path"]}'
    size=0
    with requests.get(url,stream=True,timeout=(15,60)) as r:
      r.raise_for_status()
      with path.with_suffix('.part').open('xb') as f:
       for chunk in r.iter_content(1048576):
        size+=len(chunk);assert size<=x['size'];assert time.monotonic()-start<900;f.write(chunk)
    part=path.with_suffix('.part');assert size==x['size'] and digest(part)==x['lfs']['oid'];part.rename(path)
    event=float(metadata[sid]['time_of_event']);mapping=[];overview=[];around=[]
    # Obtain native timing and image content without touching any model prediction.
    with av.open(str(path)) as c:
      stream=c.streams.video[0];stream.thread_count=2
      for i,fr in enumerate(c.decode(stream)):
       assert fr.pts is not None
       pts=float(fr.pts*fr.time_base)
       assert not mapping or pts>mapping[-1]['time_s']
       mapping.append({'frame_id':i,'native_pts':fr.pts,'time_base':str(fr.time_base),'time_s':pts})
       if i%90==0:overview.append((i,pts,fr.to_image()))
       if event-1<=pts<=event+1 and i%3==0:around.append((i,pts,fr.to_image()))
    put(f'{sid}.pts.json',{'source_sha256':digest(path),'mapping':mapping})
    sheet(sid+'.overview',overview);sheet(sid+'.event',around)
    record={'id':sid,'source_video':str(path.relative_to(ROOT)),'source_url':url,'provider_sha256':x['lfs']['oid'],'local_sha256':digest(path),'bytes':size,'frames':len(mapping),'first_pts':mapping[0]['time_s'],'last_pts':mapping[-1]['time_s'],'provider_event_hint_only':event,'provider_metadata':metadata[sid],'pts_sha256':digest(OUT/f'{sid}.pts.json'),'model_prediction_seen':False,'ai_review_pending':True,'actual_ego_contact':None,'human_truth':False,'independent_evaluation_eligible':False}
    report['records'].append(record);put('acquisition_round2.json',report);print(json.dumps({'id':sid,'bytes':size,'frames':len(mapping),'provider_event_hint':event}),flush=True)
 report['status']='download_decode_hash_pts_complete_ai_review_pending'
except BaseException as e:
 report.update(status='failed_partial_preserved',error=repr(e));raise
finally:
 report['elapsed_seconds']=time.monotonic()-start;put('acquisition_round2.json',report)
