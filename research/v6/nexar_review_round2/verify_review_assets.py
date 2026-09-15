"""Verify prepared review files, timestamps and sample pixels without annotation."""
import sys, json, hashlib, datetime
from pathlib import Path
sys.dont_write_bytecode = True
import av
from PIL import Image
HERE = Path(__file__).resolve().parent
DIST = HERE.parents[1]/'v6_review_tool/dist'

def sha(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle,'sha256').hexdigest()

def main():
    destination = HERE/'review_readiness.json'
    assert not destination.exists()
    acquired = json.loads((HERE/'acquisition.json').read_text(encoding='utf8'))
    rows = []
    for source in acquired['records']:
        video = Path(source['path'])
        case_id = 'NEXAR_REVIEW_'+video.stem
        case_path = DIST/'cases'/case_id/'case.json'
        case = json.loads(case_path.read_text(encoding='utf8'))
        mapping = json.loads(video.with_suffix('.mapping.json').read_text(encoding='utf8'))
        assert case['video_sha256'] == source['sha256'] == sha(video)
        assert case['exposure'] == 'unseen' and case['split'] == 'unassigned' and case['labels'] is None
        assert [dict(frame=f['frame'],pts_seconds=f['pts_seconds']) for f in case['frames']] == mapping['frame_pts']
        assert len(case['frames']) == source['frames']
        for f in case['frames']:
            assert sha(DIST/f['image']) == f['sha256']
        wanted = {0, len(case['frames'])//2, len(case['frames'])-1}
        checked = []
        with av.open(str(video)) as container:
            stream = container.streams.video[0]
            stream.thread_count = 2
            total = 0
            for number, frame in enumerate(container.decode(stream)):
                total += 1
                assert float(frame.pts*frame.time_base) == case['frames'][number]['pts_seconds']
                if number in wanted:
                    with Image.open(DIST/case['frames'][number]['image']) as image:
                        assert image.convert('RGB').tobytes() == frame.to_image().convert('RGB').tobytes()
                    checked.append(number)
            assert total == len(case['frames'])
        rows.append(dict(ID=case_id,frames=total,source_sha256=source['sha256'],case_sha256=sha(case_path),
                         full_pts_equal=True,all_png_hashes_equal=True,exact_rgb_sample_frames=checked,
                         contact_ground_truth=None,review_pending=True))
    report = dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),status='READY_FOR_HUMAN_REVIEW',
                  sources=rows,new_frame_count=sum(r['frames'] for r in rows),model_calls=0,
                  note='Native PTS and image integrity verified; contact, incident independence and accuracy are not certified.',
                  contact_review_scope='Counterpart and first contact, or explicit uncertainty/no contact reason. Other fields may remain unknown.')
    with destination.open('x',encoding='utf8') as handle:
        json.dump(report,handle,ensure_ascii=False,indent=2)
    print(json.dumps(report,ensure_ascii=False),flush=True)

if __name__ == '__main__':
    main()
