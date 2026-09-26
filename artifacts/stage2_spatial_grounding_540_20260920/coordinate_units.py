"""Exploratory, post-response coordinate-unit audit; never changes raw evaluation."""
import copy,hashlib,importlib.util,json
from pathlib import Path
from PIL import Image,ImageDraw
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
spec=importlib.util.spec_from_file_location('frozen_eval',HERE/'evaluate.py');e=importlib.util.module_from_spec(spec);spec.loader.exec_module(e)
read=lambda p:json.loads(p.read_text())
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
protected=[HERE/'evaluation.json']+[HERE/'run'/f'f{f}_spatial/result.json' for f in [31,39]]
before={str(p):sha(p) for p in protected}
scale=lambda xy:[xy[0]*1152/1000,xy[1]*768/1000]
rows=[]
for f in [31,39]:
    raw=read(HERE/'run'/f'f{f}_spatial/result.json');original=e.strict_json(raw['raw']);o=copy.deepcopy(original)
    b=o['bbox_xyxy'];o['bbox_xyxy']=scale(b[:2])+scale(b[2:]) if b else None
    for key in ['wheels','boundary']:
        for v in o[key]:
            if v['xy'] is not None:v['xy']=scale(v['xy'])
    if o['inside_point_xy'] is not None:o['inside_point_xy']=scale(o['inside_point_xy'])
    job=read(HERE/'inputs'/f'f{f}_spatial.job.json')
    with Image.open(job['image']) as src:im=src.convert('RGB')
    d=ImageDraw.Draw(im)
    if o['bbox_xyxy']:d.rectangle(o['bbox_xyxy'],outline='red',width=3)
    for i,w in enumerate(o['wheels']):
        if w['xy']:
            x,y=w['xy'];d.ellipse((x-5,y-5,x+5,y+5),outline='cyan',width=3);d.text((x+5,y-15),f'W{i}',fill='cyan')
    if o['boundary']:d.line([tuple(p['xy']) for p in o['boundary']],fill='magenta',width=3)
    if o['inside_point_xy']:
        x,y=o['inside_point_xy'];d.ellipse((x-5,y-5,x+5,y+5),outline='lime',width=3)
    row=dict(frame=f,converted=o,scores={},boundary_exactly_bbox_diagonal=bool(original['boundary']) and [p['xy'] for p in original['boundary']]==[original['bbox_xyxy'][:2],original['bbox_xyxy'][2:]])
    meta=next(r for r in read(ROOT/'artifacts/stage2_wheel_detail_540_20260920/input_checks.json')['cases'] if r['frame']==f)['marker_low_box'];marker=[1.5*(meta[0]+meta[2]),1.5*(meta[1]+meta[3])]
    for who in ['a','b']:
        ref=next(r for r in read(HERE/f'review_{who}.json')['cases'] if r['frame']==f)
        row['scores'][who]=e.score(o,ref,marker)
        for i,w in enumerate(ref['wheels']):
            if w.get('xy'):
                x,y=w['xy'];rad=w['tolerance_px'];color='orange' if who=='a' else 'white'
                d.ellipse((x-rad,y-rad,x+rad,y+rad),outline=color,width=2);d.text((x+rad,y+rad),f'{who.upper()}{i}',fill=color)
    d.rectangle((0,0,1152,28),fill='black');d.text((12,8),'POST-HOC /1000: model box red, wheels cyan, boundary magenta; reference wheels A orange / B white',fill='white')
    im.save(HERE/f'f{f}_normalized_hypothesis_overlay.png');rows.append(row)
assert all(sha(Path(p))==digest for p,digest in before.items())
output=dict(status='complete_exploratory_only',new_model_calls=0,transformation='x*1152/1000,y*768/1000, no offset and no fitted parameters; official Qwen3-VL cookbook conversion',source='https://github.com/QwenLM/Qwen3-VL/blob/main/cookbooks/2d_grounding.ipynb',chosen_after_responses=True,raw_evaluation_modified=False,protected_sha256=before,rows=rows,limits='Supports coordinate-unit mismatch for these outputs, not a new preregistered success or proof of old internal reasoning. Wheel and boundary grounding still require independent validation.')
(HERE/'coordinate_unit_audit.json').write_text(json.dumps(output,ensure_ascii=False,indent=2)+'\n')
print(json.dumps([dict(frame=r['frame'],bbox=r['converted']['bbox_xyxy'],wheels=r['converted']['wheels'],iou={w:s['bbox_iou'] for w,s in r['scores'].items()},wheel_error={w:[m['distance_px'] for m in s['wheel_matches']] for w,s in r['scores'].items()},boundary_bbox_diagonal=r['boundary_exactly_bbox_diagonal']) for r in rows],indent=2))
