"""Strict observable-coordinate scoring against each pre-frozen AI reference."""
import itertools,json,math
from pathlib import Path
from PIL import Image,ImageDraw
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
read=lambda p:json.loads(p.read_text())

def strict_json(raw):
    def pairs(items):
        out={}
        for key,value in items:
            if key in out:raise ValueError('duplicate JSON key')
            out[key]=value
        return out
    def invalid_constant(value):raise ValueError('nonstandard JSON constant')
    return json.loads(raw,object_pairs_hook=pairs,parse_constant=invalid_constant)

def point(p):
    return isinstance(p,list) and len(p)==2 and all(type(v) in (int,float) and math.isfinite(v) for v in p) and 0<=p[0]<=1151 and 0<=p[1]<=767

def schema(o):
    if not isinstance(o,dict):return False
    if set(o)!={'bbox_xyxy','wheels','boundary','inside_point_xy','lane_state'}:return False
    b=o['bbox_xyxy']
    if b is not None and not (isinstance(b,list) and len(b)==4 and point(b[:2]) and point(b[2:]) and b[0]<b[2] and b[1]<b[3]):return False
    if o['inside_point_xy'] is not None and not point(o['inside_point_xy']):return False
    states=['INSIDE','OUTSIDE','UNCERTAIN']
    if o['lane_state'] not in states:return False
    if not isinstance(o['wheels'],list) or len(o['wheels'])>4:return False
    for w in o['wheels']:
        if not isinstance(w,dict) or set(w)!={'xy','role','lane_relation'}:return False
        if w['xy'] is not None and not point(w['xy']):return False
        if w['role'] not in ['front','rear','unknown'] or w['lane_relation'] not in states:return False
    if not isinstance(o['boundary'],list) or len(o['boundary']) not in [0,2,3,4,5,6]:return False
    for p in o['boundary']:
        if not isinstance(p,dict) or set(p)!={'xy','kind'} or not point(p['xy']) or p['kind'] not in ['visible','extended']:return False
    return all(a['xy'][1]<b['xy'][1] for a,b in zip(o['boundary'],o['boundary'][1:]))

def iou(a,b):
    if a is None or b is None:return None
    inter=max(0,min(a[2],b[2])-max(a[0],b[0]))*max(0,min(a[3],b[3])-max(a[1],b[1]))
    return inter/((a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-inter)

def line_x(poly,y):
    for a,b in zip(poly,poly[1:]):
        a,b=a['xy'],b['xy']
        if a is not None and b is not None and a[1]<=y<=b[1] and a[1]!=b[1]:return a[0]+(b[0]-a[0])*(y-a[1])/(b[1]-a[1])
    return None

def segment_distance(p,a,b):
    dx,dy=b[0]-a[0],b[1]-a[1]
    t=max(0,min(1,((p[0]-a[0])*dx+(p[1]-a[1])*dy)/(dx*dx+dy*dy)))
    return math.dist(p,[a[0]+t*dx,a[1]+t*dy])

def score(o,r,marker):
    box=iou(o['bbox_xyxy'],r['bbox_xyxy']);b=o['bbox_xyxy']
    contains=b is not None and b[0]<=marker[0]<=b[2] and b[1]<=marker[1]<=b[3]
    refs=[w for w in r['wheels'] if w.get('visible',True) and w.get('xy') is not None]
    pred=[(i,w) for i,w in enumerate(o['wheels']) if w['xy'] is not None]
    # At most four points: maximize accepted matches, then minimize total error.
    matches=[]
    for n in range(min(len(refs),len(pred))+1):
        for ri in itertools.combinations(range(len(refs)),n):
            for pj in itertools.permutations(range(len(pred)),n):
                rows=[dict(reference_index=i,prediction_index=pred[j][0],distance_px=math.dist(refs[i]['xy'],pred[j][1]['xy']),tolerance_px=refs[i]['tolerance_px']) for i,j in zip(ri,pj)]
                matches.append(rows)
    matched=max(matches,key=lambda rows:(sum(v['distance_px']<=v['tolerance_px'] for v in rows),len(rows),-sum(v['distance_px'] for v in rows)))
    for v in matched:v['within_tolerance']=v['distance_px']<=v['tolerance_px']
    boundary=[]
    for p in r.get('boundary',[]) or []:
        if p.get('xy') is None:continue
        x=line_x(o['boundary'],p['xy'][1]);error=None if x is None else min(segment_distance(p['xy'],a['xy'],b['xy']) for a,b in zip(o['boundary'],o['boundary'][1:]))
        boundary.append(dict(reference_xy=p['xy'],kind=p['kind'],predicted_x=x,horizontal_error_px=None if x is None else abs(x-p['xy'][0]),error_px=error,tolerance_px=p['tolerance_px'],within_tolerance=None if error is None else error<=p['tolerance_px']))
    ip=o['inside_point_xy'];rp=r.get('inside_point_xy');rb=[v for v in (r.get('boundary') or []) if v.get('xy') is not None]
    px=line_x(rb,ip[1]) if ip else None;rx=line_x(rb,rp[1]) if rp else None
    inside=None if px is None or rx is None else (ip[0]-px)*(rp[0]-rx)>0
    return dict(bbox_iou=box,bbox_iou_pass=None if box is None else box>=.5,marker_center_contained=contains,
        reference_visible_wheels=len(refs),predicted_located_wheels=len(pred),wheel_matches=matched,
        matched_within_tolerance=sum(v['within_tolerance'] for v in matched),unmatched_reference_count=len(refs)-len(matched),
        boundary_anchors=boundary,boundary_reference_available=bool(boundary),inside_point_reference_side_match=inside)

def main():
    protocol=read(HERE/'protocol.json');rows=[]
    for f in protocol['frames']:
        result=read(HERE/'run'/f'f{f}_spatial/result.json');job=read(HERE/'inputs'/f'f{f}_spatial.job.json')
        raw=result['raw']
        try:o=strict_json(raw)
        except (TypeError,ValueError):o=None
        valid=schema(o);row=dict(frame=f,raw=raw,strict_schema_valid=valid,parsed=o,expert_scores={})
        with Image.open(job['image']) as src:im=src.convert('RGB')
        draw=ImageDraw.Draw(im)
        if valid:
            box=next(r for r in read(ROOT/'artifacts/stage2_wheel_detail_540_20260920/input_checks.json')['cases'] if r['frame']==f)['marker_low_box']
            marker=[1.5*(box[0]+box[2]),1.5*(box[1]+box[3])]
            for who in ['a','b']:
                ref=next(r for r in read(HERE/f'review_{who}.json')['cases'] if r['frame']==f)
                row['expert_scores'][who]=score(o,ref,marker)
            if o['bbox_xyxy']:draw.rectangle(o['bbox_xyxy'],outline='red',width=3)
            for i,w in enumerate(o['wheels']):
                if w['xy']:
                    x,y=w['xy'];draw.ellipse((x-5,y-5,x+5,y+5),outline='cyan',width=3);draw.text((x+7,y),f'W{i}',fill='cyan')
            if o['boundary']:draw.line([tuple(p['xy']) for p in o['boundary']],fill='magenta',width=3)
            for p in o['boundary']:
                x,y=p['xy'];draw.text((x+5,y),p['kind'][0],fill='magenta')
            if o['inside_point_xy']:
                x,y=o['inside_point_xy'];draw.ellipse((x-6,y-6,x+6,y+6),outline='lime',width=3)
            states=[w['lane_relation'] for w in o['wheels']]
            aggregate='INSIDE' if 'INSIDE' in states else 'UNCERTAIN' if not states or 'UNCERTAIN' in states else 'OUTSIDE'
            row['reported_relations_aggregate']=aggregate;row['state_vs_reported_relations_consistent']=o['lane_state']==aggregate
        im.save(HERE/f'f{f}_prediction_overlay.png');rows.append(row)
    output=dict(status='complete',actual_model_calls=2,rows=rows,official_S2=None,entry_output_modified=False,
        limits='Frozen AI references, uncertain boundary unscored; coordinate task differs from old classifier. Numeric thresholds are diagnostic tolerances, not official metrics.')
    (HERE/'evaluation.json').write_text(json.dumps(output,ensure_ascii=False,indent=2)+'\n')

if __name__=='__main__':
    assert iou([0,0,10,10],[5,0,15,10]) == 1/3
    assert segment_distance([3,5],[0,0],[0,10]) == 3
    assert line_x([{'xy':[0,0]},{'xy':[10,10]}],20) is None
    assert not point([True,2]) and not point([float('nan'),2])
    main()
