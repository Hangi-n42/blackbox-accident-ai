"""Prepare exact decoded frames for local human review. No GT/predictions read."""
import argparse
import hashlib
import json
from pathlib import Path
import av
from PIL import Image


def digest(path):
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def prepare(video, destination, case_id, source_group, source_uri, exposure, resume=False):
    destination = destination.resolve()
    destination.mkdir(parents=True, exist_ok=True)
    case_dir = destination / 'cases' / case_id
    if case_dir.exists() and (not resume or (case_dir / 'case.json').exists()):
        raise ValueError('Case already exists; existing frames and drafts are not overwritten.')
    if not case_id or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in case_id):
        raise ValueError('Use letters, numbers, underscore or dash for case ID.')
    case_dir.mkdir(parents=True, exist_ok=resume)
    frames = []
    with av.open(str(video)) as container:
        stream = container.streams.video[0]
        for number, frame in enumerate(container.decode(stream)):
            if frame.pts is None:
                raise ValueError('Missing native PTS; do not replace with assumed FPS.')
            seconds = float(frame.pts * frame.time_base)
            if frames and seconds <= frames[-1]['pts_seconds']:
                raise ValueError('PTS not strictly increasing; manual time mapping required.')
            name = f'frame_{number:06d}.png'
            path = case_dir / name
            decoded = frame.to_image()
            if path.exists():
                try:
                    with Image.open(path) as existing:
                        if existing.size != decoded.size or existing.convert('RGB').tobytes() != decoded.convert('RGB').tobytes():
                            raise ValueError('Partial frame pixels differ from decoded source; refusing reuse.')
                except OSError:
                    # Preserve an interrupted encoder's incomplete file for inspection.
                    path.rename(path.with_suffix('.interrupted.png'))
            if not path.exists():
                partial = path.with_suffix('.png.part')
                decoded.save(partial, format='PNG', compress_level=1)
                partial.rename(path)
            frames.append({'frame': number, 'pts_seconds': seconds,
                           'image': f'cases/{case_id}/{name}', 'sha256': digest(path)})
    if not frames:
        raise ValueError('No decoded frames')
    case = {'ID': case_id, 'source_group_id': source_group, 'video_path': str(video.resolve()),
            'video_sha256': digest(video), 'source_uri': source_uri,
            'split': 'development' if exposure == 'seen' else 'unassigned',
            'exposure': exposure, 'frame_number_rule': 'zero-based full decoded source stream',
            'time_origin': 'native source presentation timestamp seconds, no offset applied',
            'mapping_method': 'PyAV native frame.pts * frame.time_base',
            'frames': frames, 'labels': None,
            'scope': 'Review input only; not a ground-truth record or independent validation claim.'}
    (case_dir / 'case.json').write_text(json.dumps(case, ensure_ascii=False, indent=2), encoding='utf-8')
    mapping = {'frame_pts': [{'frame': item['frame'], 'pts_seconds': item['pts_seconds']}
                             for item in frames],
               'source_video_sha256': case['video_sha256'],
               'time_origin': case['time_origin'], 'method': case['mapping_method']}
    (case_dir / 'mapping.json').write_text(json.dumps(mapping, ensure_ascii=False, indent=2), encoding='utf-8')
    all_cases = [json.loads(p.read_text(encoding='utf-8')) for p in sorted((destination / 'cases').glob('*/case.json'))]
    (destination / 'cases.js').write_text('window.REVIEW_CASES = ' + json.dumps(all_cases, ensure_ascii=False) + ';\n', encoding='utf-8')
    return {'ID': case_id, 'frames': len(frames), 'case_sha256': digest(case_dir / 'case.json')}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--video', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=Path(__file__).parent / 'dist')
    parser.add_argument('--id', required=True)
    parser.add_argument('--source-group', required=True)
    parser.add_argument('--source-uri', required=True)
    parser.add_argument('--exposure', choices=['seen', 'unseen'], required=True)
    parser.add_argument('--resume', action='store_true', help='Resume incomplete case only; verify existing frame pixels against source.')
    args = parser.parse_args()
    print(json.dumps(prepare(args.video, args.output, args.id, args.source_group, args.source_uri, args.exposure, args.resume)))
