"""Acquire six public review candidates, never promote event labels to contact GT."""
import hashlib
import json
from pathlib import Path
from datetime import datetime, timezone
import requests
import av

ROOT = Path(__file__).resolve().parent / 'nexar_review_candidates'
REPO = 'nexar-ai/nexar_collision_prediction'
BASE = 'https://huggingface.co'


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def main():
    ROOT.mkdir(exist_ok=True)
    if (ROOT / 'plan.json').exists():
        raise RuntimeError('Existing plan: inspect before resuming; do not replace cohort.')
    session = requests.Session()
    def get(url):
        r = session.get(url, timeout=(15, 60))
        r.raise_for_status()
        return r
    info = get(f'{BASE}/api/datasets/{REPO}/revision/main').json()
    revision = info['sha']
    entries = get(f'{BASE}/api/datasets/{REPO}/tree/{revision}/train/positive?limit=1000').json()
    eligible = sorted((x for x in entries if x['path'].endswith('.mp4')), key=lambda x: x['path'])
    selected = eligible[:6]
    if len(selected) != 6 or sum(x['size'] for x in selected) > 200 * 1024**2:
        raise RuntimeError('Pilot count/size outside bound.')
    plan = {'created_at': datetime.now(timezone.utc).isoformat(), 'repository': REPO,
            'revision': revision, 'selection': 'First six lexicographic train/positive MP4 entries; convenience review pilot, not representative validation.',
            'selected': selected, 'labels_loaded': False, 'model_predictions_seen': False,
            'ground_truth_eligible': False, 'split': 'unassigned',
            'required_review': 'Actual contact versus near miss; counterpart; first contact; lane entry; direction; available space; source overlap audit.'}
    write(ROOT / 'plan.json', plan)
    for name in ['LICENSE', 'README.md']:
        payload = get(f'{BASE}/datasets/{REPO}/resolve/{revision}/{name}').content
        (ROOT / name).write_bytes(payload)
    records = []
    for item in selected:
        path = ROOT / Path(item['path']).name
        if path.exists():
            raise RuntimeError('Existing video; refusing overwrite.')
        url = f'{BASE}/datasets/{REPO}/resolve/{revision}/{item["path"]}'
        with session.get(url, stream=True, timeout=(15, 60)) as response:
            response.raise_for_status()
            h = hashlib.sha256()
            size = 0
            with path.with_suffix('.part').open('xb') as out:
                for chunk in response.iter_content(1024 * 1024):
                    size += len(chunk)
                    if size > item['size']:
                        raise RuntimeError('Downloaded size exceeds declared object size.')
                    out.write(chunk)
                    h.update(chunk)
        expected = item.get('lfs', {}).get('oid')
        if size != item['size'] or not expected or h.hexdigest() != expected:
            raise RuntimeError('Repository object hash/size mismatch.')
        path.with_suffix('.part').rename(path)
        frames = []
        with av.open(str(path)) as container:
            stream = container.streams.video[0]
            stream.thread_count = 2
            for number, frame in enumerate(container.decode(stream)):
                if frame.pts is None:
                    raise RuntimeError('Missing native PTS.')
                pts = float(frame.pts * frame.time_base)
                if pts < 0 or (frames and pts <= frames[-1]['pts_seconds']):
                    raise RuntimeError('Invalid/non-increasing PTS.')
                frames.append({'frame': number, 'pts_seconds': pts})
        if not frames:
            raise RuntimeError('Empty video.')
        mapping_path = path.with_suffix('.mapping.json')
        write(mapping_path, {'frame_pts': frames, 'source_video_sha256': h.hexdigest(),
                            'time_origin': 'native source presentation timestamp seconds, no offset applied',
                            'method': 'PyAV native frame.pts * frame.time_base'})
        records.append({'path': str(path), 'source_uri': url, 'sha256': h.hexdigest(),
                        'bytes': size, 'frames': len(frames), 'first_pts': frames[0]['pts_seconds'],
                        'last_pts': frames[-1]['pts_seconds'],
                        'mapping_sha256': hashlib.sha256(mapping_path.read_bytes()).hexdigest(),
                        'human_review': 'pending', 'contact_ground_truth': None,
                        'source_group_id': None, 'independence_verified': False})
        write(ROOT / 'acquisition.json', {'status': 'in_progress', 'records': records})
        print(json.dumps({'downloaded': path.name, 'bytes': size, 'frames': len(frames)}), flush=True)
    write(ROOT / 'acquisition.json', {'status': 'six_review_candidates_acquired_not_validated', 'records': records})


if __name__ == '__main__':
    main()
