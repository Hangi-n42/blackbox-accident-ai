"""Render existing human-draft evidence; never edit the drafts or source frames."""
from pathlib import Path
import json,hashlib
from PIL import Image,ImageDraw
ROOT=Path(__file__).resolve().parents[2];HERE=Path(__file__).resolve().parent
read=lambda p:json.loads(p.read_text())
def sheet(paths,title,out):
    canvas=Image.new('RGB',(1280,((len(paths)+3)//4)*204),'#171717');draw=ImageDraw.Draw(canvas)
    for k,(path,num,pts) in enumerate(paths):
        with Image.open(path) as im:
            im=im.convert('RGB');im.thumbnail((320,180));x=k%4*320;y=k//4*204;canvas.paste(im,(x,y))
        draw.text((x+3,y+182),f'{title} f{num} {pts:.3f}s',fill='white')
    canvas.save(out,quality=95)
def main():
    out=HERE/'training_review';out.mkdir(exist_ok=False)
    manifest=read(ROOT/'artifacts/stage2_goal_20260919/core_v2/evaluation_manifest.json')
    human={r['ID']:r for r in read(ROOT/'artifacts/mac_experiments/baseline_20260916/human_labels.json')}
    cases=[]
    for row in manifest['cases']:
        sid=row['ID']
        if sid not in human or human[sid]['draft']['review']['entry']['status']!='observed':continue
        f=out/sid;f.mkdir();images=row['images'];review=human[sid]['draft']['review'];entry=review['entry'];contact=review['contact'];n=len(images)
        indexes=sorted({round(k*(n-1)/47) for k in range(48)})
        for page in range(3):
            picks=indexes[page*16:(page+1)*16];sheet([(ROOT/images[i]['path'],images[i]['frame'],images[i]['pts_seconds']) for i in picks],sid,f/f'overview_{page}.jpg')
        near=set([0])
        for event in [entry['pts_seconds'],contact['pts_seconds']]:
            for offset in [-.8,-.4,-.2,-.1,0,.1,.2,.4,.8]:near.add(min(range(n),key=lambda i:abs(images[i]['pts_seconds']-(event+offset))))
        picks=sorted(near)
        for page in range((len(picks)+15)//16):sheet([(ROOT/images[i]['path'],images[i]['frame'],images[i]['pts_seconds']) for i in picks[page*16:(page+1)*16]],sid,f/f'event_{page}.jpg')
        record=dict(ID=sid,human_draft_path=human[sid]['path'],human_draft_sha256=human[sid]['sha256'],human_review=review,images=images,review_sheets=[str(p.relative_to(ROOT)) for p in sorted(f.glob('*.jpg'))],point_reference_is_not_adjudicated=True)
        (f/'input.json').write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n');cases.append(dict(ID=sid,input=str((f/'input.json').relative_to(ROOT)),sheets=record['review_sheets']))
    (out/'manifest.json').write_text(json.dumps(dict(cases=cases,model_outputs_provided=False,new_model_calls=0),ensure_ascii=False,indent=2)+'\n');print('Prepared',len(cases),'human-draft review packets')
if __name__=='__main__':main()
