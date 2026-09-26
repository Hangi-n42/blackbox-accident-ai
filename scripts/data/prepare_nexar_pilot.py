"""Fetch a small, reproducible unseen-source candidate pool; labels stay unknown."""
import hashlib
import argparse
import json
from pathlib import Path

import av
import requests

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'artifacts/data_pilot_20260916'
REPO = 'nexar-ai/nexar_collision_prediction'
BASE = 'https://huggingface.co'


def write(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--folder', default='nexar')
    parser.add_argument('--count', type=int, default=30)
    parser.add_argument('--plan-only', action='store_true')
    args = parser.parse_args()
    if Path(args.folder).name != args.folder or args.count < 1:
        parser.error('Require one folder name and a positive count')
    folder = OUT / args.folder
    folder.mkdir(exist_ok=True)
    session = requests.Session()
    plan_path = folder / 'selection.json'
    if plan_path.exists():
        plan = json.loads(plan_path.read_text())
    else:
        response = session.get(f'{BASE}/api/datasets/{REPO}/revision/main', timeout=30)
        response.raise_for_status()
        revision = response.json()['sha']
        response = session.get(f'{BASE}/api/datasets/{REPO}/tree/{revision}/train/positive?limit=1000', timeout=30)
        response.raise_for_status()
        inventory = json.loads((OUT / 'inventory.json').read_text())
        seen = {row['source_id'] for row in inventory['nexar']}
        for prior in OUT.glob('nexar*/selection.json'):
            seen.update(Path(x['path']).stem for x in json.loads(prior.read_text())['selected'])
        entries = [x for x in response.json() if x['path'].endswith('.mp4') and Path(x['path']).stem not in seen]
        entries.sort(key=lambda x: hashlib.sha256(('mac-pilot-20260916:' + x['path']).encode()).hexdigest())
        plan = dict(repository=REPO, revision=revision, selected=entries[:args.count],
                    selection='Fixed hash order of unused IDs; candidate pool, not confirmed collisions or independent incidents.',
                    excluded_prior_source_ids=sorted(seen))
        assert len(plan['selected']) == args.count
        write(plan_path, plan)
    print('planned_bytes', sum(x['size'] for x in plan['selected']), flush=True)
    if args.plan_only:
        return
    records = []
    for item in plan['selected']:
        path = folder / Path(item['path']).name
        expected = item['lfs']['oid']
        if not path.exists():
            url = f'{BASE}/datasets/{REPO}/resolve/{plan["revision"]}/{item["path"]}'
            partial = path.with_suffix('.part')
            with session.get(url, stream=True, timeout=(20, 60)) as response:
                response.raise_for_status()
                with partial.open('wb') as stream:
                    for chunk in response.iter_content(1024 * 1024):
                        stream.write(chunk)
            with partial.open('rb') as stream:
                assert hashlib.file_digest(stream, 'sha256').hexdigest() == expected
            assert partial.stat().st_size == item['size']
            partial.rename(path)
        pts = []
        with av.open(str(path)) as container:
            for frame in container.decode(video=0):
                pts.append(float(frame.pts * frame.time_base))
        assert pts and all(b > a for a, b in zip(pts, pts[1:]))
        write(path.with_suffix('.pts.json'), pts)
        records.append(dict(source_id=path.stem, path=str(path.relative_to(ROOT)), sha256=expected,
                            frames=len(pts), first_pts=pts[0], last_pts=pts[-1],
                            actual_ego_contact=None, contact_frame=None, entry_frame=None,
                            entry_side=None, ego_space=None, counterpart=None,
                            annotation_status='unreviewed', split='unassigned'))
        write(folder / 'review_queue.json', records)
        print(path.stem, len(pts), 'frames', flush=True)


if __name__ == '__main__':
    main()
