"""Lossless full-frame Mac inputs from the six already acquired source clips."""
import hashlib
import json
from pathlib import Path
import time

import av

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent / 'acquisition'


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    started = time.perf_counter()
    out = HERE / 'full_inputs'
    out.mkdir(exist_ok=False)
    cases = []
    for number in range(24, 30):
        case_id = f'{number:05d}'
        source, pts_path = SOURCE / (case_id + '.mp4'), SOURCE / (case_id + '.pts.json')
        pts = json.loads(pts_path.read_text())
        assert sha(source) == pts['source_sha256']
        folder = out / case_id
        folder.mkdir()
        images = []
        with av.open(str(source)) as container:
            stream = container.streams.video[0]
            stream.thread_count = 2
            for index, frame in enumerate(container.decode(stream)):
                row = pts['mapping'][index]
                assert index == row['frame_id'] and frame.pts == row['native_pts']
                assert str(frame.time_base) == row['time_base'] and float(frame.pts * frame.time_base) == row['time_s']
                image = frame.to_image().convert('RGB')
                target = folder / f'frame_{index:06d}.png'
                image.save(target, compress_level=1)
                images.append(dict(path=str(target.relative_to(ROOT)), frame=index, pts_seconds=row['time_s'],
                                   sha256=sha(target), rgb_sha256=hashlib.sha256(image.tobytes()).hexdigest(),
                                   width=image.width, height=image.height))
        assert len(images) == len(pts['mapping'])
        cases.append(dict(ID=case_id, group='new_source_ai_review', source_sha256=pts['source_sha256'],
                          source_video=str(source.relative_to(ROOT)), pts_source=str(pts_path.relative_to(ROOT)),
                          pts_sha256=sha(pts_path), split='new_source_diagnostic',
                          source_incident_independence='not_certified', images=images))
        print(json.dumps(dict(ID=case_id, frames=len(images))), flush=True)
    with (HERE / 'inputs.json').open('x') as stream:
        json.dump(cases, stream, ensure_ascii=False, indent=2)
    with (HERE / 'input_checks.json').open('x') as stream:
        json.dump(dict(status='PASS', cases=len(cases), frames=sum(len(c['images']) for c in cases),
                       bytes=sum((ROOT / i['path']).stat().st_size for c in cases for i in c['images']),
                       lossless_rgb=True, native_pts_verified=True, model_predictions_read=False,
                       source_hashes_verified=True, inputs_sha256=sha(HERE / 'inputs.json'),
                       extraction_script_sha256=sha(Path(__file__)), wall_seconds=time.perf_counter() - started), stream, indent=2)


if __name__ == '__main__':
    main()
