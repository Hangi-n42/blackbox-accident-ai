"""Two fixed spatial queries, reusing the existing offline Mac state worker."""
import argparse
from datetime import datetime,timezone
import importlib.util
from pathlib import Path

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
PREV=ROOT/'artifacts/stage2_wheel_detail_540_20260920'
spec=importlib.util.spec_from_file_location('wheel_runner',ROOT/'artifacts/stage2_wheel_state_540_20260920/experiment.py')
wheel=importlib.util.module_from_spec(spec);spec.loader.exec_module(wheel)
read,write,sha=wheel.read,wheel.write,wheel.sha
PROMPT='''The vehicle marked in yellow is the collision counterpart. Locate observable geometry in this single image. Use pixel coordinates in the full 1152 by 768 image, including the header: origin (0,0) at top left, x increases rightward, y downward. Do not use normalized coordinates.
Return only one JSON object with these keys:
"bbox_xyxy": [left,top,right,bottom] enclosing the marked vehicle, or null if unidentified;
"wheels": a list of at most four objects {"xy":[x,y] or null,"role":"front" or "rear" or "unknown","lane_relation":"INSIDE" or "OUTSIDE" or "UNCERTAIN"}. Locate the road-contact point of each actually visible wheel of that vehicle. Do not invent hidden wheel points. Use null for an indistinct contact point;
"boundary": a list of 2 to 6 objects {"xy":[x,y],"kind":"visible" or "extended"}, ordered from far to near. Locate the camera car lane boundary on the counterpart's side, using the centerline of the painted lane marking. Include the region near the wheels and the road in front of the camera car. Continue the visible boundary across marking gaps or occlusion, marking those points "extended". Use an empty list if this boundary cannot be determined;
"inside_point_xy": one point on the road clearly inside the camera car's driving lane, or null if undetermined;
"lane_state":"INSIDE" or "OUTSIDE" or "UNCERTAIN". INSIDE means at least one counterpart wheel touches the lane boundary or lies on its inside; OUTSIDE means all wheels remain outside without touching. Use UNCERTAIN if the required wheel-boundary relation cannot be determined. Report only these observable coordinates and classifications, without an explanation.'''


def prepare():
    out=HERE/'inputs';out.mkdir(exist_ok=False)
    for f in [31,39]:
        old=read(PREV/'inputs'/f'f{f}_source_detail.job.json')
        assert sha(Path(old['image']))==old['sha256']
        write(out/f'f{f}_spatial.job.json',dict(ID=old['ID'],image=old['image'],sha256=old['sha256'],prompt=PROMPT,max_new_tokens=768))
    write(HERE/'protocol.json',dict(created_utc=datetime.now(timezone.utc).isoformat(),frames=[31,39],prompt=PROMPT,max_new_tokens=768,actual_calls_planned=2,
        input_policy='Exact existing source_detail image bytes; same model, Mac native/sync/deepstack; one fresh worker per frame, no retries or post-response prompt changes.',
        reference_policy='Two independent prior-exposed AI spatial references frozen before calls. Preserve both; disagreement or unresolvable occlusion stays unconfirmed. Existing state labels are diagnostic context only, never force geometry to fit.',
        evaluation_policy='Strict JSON schema and pixel bounds; vehicle bbox IoU>=0.5 and marker containment; visible wheel points matched one-to-one within each expert tolerance; boundary x at each expert anchor y within expert tolerance with no extrapolation; inside point must lie on reference lane side. Report each expert separately and do not claim human GT.',
        interpretation='New question and 768-token budget differ from previous 40-token state question. Observable spatial outputs do not reveal the old internal causal process. No official S2, entry override, production change, or generalization score.',
        stop='Exactly two model calls. Null/invalid/truncated outputs retained, never retried.'))
    write(HERE/'STATUS.json',dict(status='preparing',planned_model_calls=2,actual_model_calls=0))


def freeze():
    assert not (HERE/'freeze.json').exists() and not (HERE/'run').exists()
    assert read(HERE/'preflight_review.json')['status']=='PASS'
    for who in ['a','b']:assert read(HERE/f'review_{who}.json')['new_predictions_seen'] is False
    files=dict(read(PREV/'freeze.json')['files'])
    for name,digest in files.items():assert sha(ROOT/name)==digest,name
    for path in [PREV/'freeze.json',PREV/'evaluation.json',PREV/'independent_verification.json']:
        files[str(path.relative_to(ROOT))]=sha(path)
    for f in [31,39]:
        for name in ['result.json','calls.json','worker_report.json']:
            path=PREV/'run'/f'f{f}_source_detail'/name
            files[str(path.relative_to(ROOT))]=sha(path)
    for path in HERE.rglob('*'):
        if path.is_file() and '__pycache__' not in path.parts and path.name!='STATUS.json':files[str(path.relative_to(ROOT))]=sha(path)
    write(HERE/'freeze.json',dict(created_utc=datetime.now(timezone.utc).isoformat(),status='locked_before_spatial_predictions',model_calls_planned=2,files=files))


def run():
    wheel.HERE=HERE;wheel.ARMS=['spatial'];wheel.run()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','freeze','run']);globals()[p.parse_args().action]()
