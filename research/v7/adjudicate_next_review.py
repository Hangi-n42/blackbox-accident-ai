"""Preserve uncertainty from two frozen AI reviews; never create exact GT."""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / 'research/v7'

def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    ap = BASE / 'next_review_a/review.json'
    bp = BASE / 'next_review_b/review_summary.json'
    assert sha(ap) == 'e8eec2f1889c97fbbe8ef05f029837d808047c30ffda7d723cc353fd2b789888'
    assert sha(bp) == 'ce30cef736a69d12551ecf276ace4606a08d8b9f5910b861f7609b115c832c2c'
    aa = {r['ID']: r for r in read(ap)['records']}
    bb = {r['ID']: r for r in read(bp)['videos']}
    ids = ['00017', '00018', '00019', '00021', '00022', '00023']
    assert sorted(aa) == sorted(bb) == ids
    rows = []
    for ident in ids:
        a, b = aa[ident], bb[ident]
        assert a['source']['sha256'] == b['source_sha256']
        assert a['contact_exact_frame'] is None and b['contact']['exact_frame'] is None
        assert a['entry_frame'] is None and b['entry']['value'] is None
        ai, bi = a['inferred_contact_interval'], b['contact']['interval']
        # Root read both descriptions before this serialization: these three
        # refer to the same described actor/event; no automatic text matching.
        same_inferred_event = ident in {'00019', '00021', '00023'}
        interval = None
        if same_inferred_event:
            assert ai is not None and bi is not None
            lo = min(ai[0]['frame'], bi['frame_interval'][0])
            hi = max(ai[1]['frame'], bi['frame_interval'][1])
            pts = read(BASE / f'next_review_native_qa/{ident}.frames.json')['frames']
            interval = {'frame_interval': [lo, hi],
                        'pts_seconds_interval': [pts[lo]['pts_seconds'], pts[hi]['pts_seconds']],
                        'combination': 'union_hull_not_intersection',
                        'status': 'conditional_inferred_contact_interval'}
        side = a['side'] if same_inferred_event and a['side'] == b['side']['value'] else None
        space = a['space'] if same_inferred_event and a['space'] == b['space']['value'] else None
        rows.append({
            'ID': ident, 'source_sha256': a['source']['sha256'],
            'same_inferred_actor_event_supported_by_both': same_inferred_event,
            'actor_descriptions': {'A': a['actor'] or a['candidate_actor'], 'B': b['same_actor']['description']},
            'contact_exact_frame': None, 'contact_interval': interval,
            'reviewer_contact_status': {'A': a['ego_contact'], 'B': b['contact']['status']},
            'original_intervals': {'A': ai, 'B': bi},
            'entry': None, 'already_in_lane_at_original_start': None,
            'conditional_side': side, 'conditional_space': space,
            'reviewer_reasons': {'A': a['uncertainty'], 'B': b['contact']['reason']},
            'conditions': 'Contact and actor remain inferred; close approach/braking alternatives are not fully excluded. Side/space do not constitute unconditional GT.',
            'disagreement': 'A supports an inferred interval; B cannot distinguish contact. Consensus timing unknown.' if ident == '00022' else None,
        })
    report = {
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'status': 'frozen_secondary_evidence', 'adjudicator': 'root_AI',
        'human_GT_count': 0, 'official_GT_count': 0, 'exact_contact_count': 0,
        'exact_entry_count': 0, 'conditional_contact_interval_count': 3,
        'conditional_side_count': 2, 'conditional_space_count': 1,
        'source_count': 6, 'source_replacements': 0,
        'predictions_read_before_adjudication': False,
        'gate_override': False,
        'limitations': 'Blind procedures do not establish independent reviewer errors, incident independence, or absence from model pretraining. AI agreement is not human or official truth.',
        'bindings': {str(p.relative_to(ROOT)): sha(p) for p in [
            ap, bp, BASE / 'next_review_adjudication_protocol.json',
            BASE / 'new_validation_sources/selection_plan.json',
            BASE / 'next_review_native_qa/report.json',
            BASE / 'next_review_native_qa/reviewer_png_check/report.json',
            Path(__file__).resolve()]},
        'records': rows,
    }
    dest = BASE / 'next_review_adjudication.json'
    with dest.open('x', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
        f.write('\n')
    print(json.dumps({k: v for k, v in report.items() if k.endswith('_count')}, ensure_ascii=True))

if __name__ == '__main__':
    main()
