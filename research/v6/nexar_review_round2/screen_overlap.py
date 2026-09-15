"""Source-only sparse overlap screen for new cohort; no model or labels."""
import os, sys, json, hashlib, itertools, datetime, importlib.util
from pathlib import Path
sys.dont_write_bytecode = True
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '2'
import av
import cv2
import numpy as np
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
OLD = ROOT / 'research/v6_stage2/nexar_source_overlap_audit'
spec = importlib.util.spec_from_file_location('previous_source_audit', OLD/'audit.py')
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)
cv2.setNumThreads(2)

def save(name, value):
    with (HERE/name).open('x', encoding='utf8') as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)

def main():
    manifest = json.loads((HERE/'manifest.json').read_text(encoding='utf8'))
    prior = json.loads((OLD/'plan_frozen.json').read_text(encoding='utf8'))
    sources = [dict(id='NEXAR_'+Path(s['path']).stem, group='round2',
                    path=str(HERE/Path(s['path']).name), sha256=s['sha256'])
               for s in manifest['selected']]
    sources += [dict(s, group='previously_exposed') for s in prior['sources']]
    assert all(base.sha(s['path']) == s['sha256'] for s in sources)
    save('overlap_plan.json', dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
         sources=sources, script_sha256=base.sha(__file__), helper_sha256=base.sha(OLD/'audit.py'),
         sampling='12 rounded linspace native decoded frame indices per video',
         near_hamming_max=8, near_gray_mae_max=.08,
         candidate_rule='Any exact sampled frame or >=3 distinct near-matching frames on both sides in one view',
         scope='New-new and new-old source matches only. No model predictions or labels.',
         limitation='Sparse screen cannot certify different incidents or pretraining independence.'))
    data = {}
    for s in sources:
        # Metadata frame count is verified against sequential decoding.
        with av.open(s['path']) as container:
            stream = container.streams.video[0]
            stream.thread_count = 2
            count = stream.frames
            assert count > 0
            chosen = set(np.rint(np.linspace(0, count-1, 12)).astype(int).tolist())
            sampled = []
            decoded = 0
            for number, frame in enumerate(container.decode(stream)):
                decoded += 1
                if number not in chosen:
                    continue
                im = frame.to_ndarray(format='bgr24')
                h,w = im.shape[:2]
                sampled.append(dict(index=number, shape=list(im.shape),
                    sha256=hashlib.sha256(im.tobytes()).hexdigest(),
                    full=base.signature(im), center80percent=base.signature(im[int(.1*h):int(.9*h),int(.1*w):int(.9*w)])))
            assert decoded == count and len(sampled) == len(chosen)
        data[s['id']] = sampled
        print(s['id'], decoded, flush=True)
    pairs = []
    for a,b in itertools.combinations(sources,2):
        if a['group'] != 'round2' and b['group'] != 'round2':
            continue
        exact, near = [], []
        for f,g in itertools.product(data[a['id']],data[b['id']]):
            if f['shape'] == g['shape'] and f['sha256'] == g['sha256']:
                exact.append([f['index'],g['index']])
            for view in ('full','center80percent'):
                fs,gs = f[view],g[view]
                distance = int(np.count_nonzero(fs[0] != gs[0]))
                mae = float(np.mean(np.abs(fs[1]-gs[1])))
                if distance <= 8 and mae <= .08:
                    near.append(dict(a=f['index'],b=g['index'],view=view,hamming=distance,gray_mae=mae))
        flagged = bool(exact) or any(len({x['a'] for x in near if x['view']==v})>=3 and len({x['b'] for x in near if x['view']==v})>=3 for v in ('full','center80percent'))
        pairs.append(dict(a=a['id'],b=b['id'],exact=exact,near=near,candidate=flagged,same_file_sha=a['sha256']==b['sha256']))
    assert len(pairs) == 36 and all(base.sha(s['path']) == s['sha256'] for s in sources)
    result = dict(status='sparse_source_screen_complete',pair_count=len(pairs),sample_count=sum(map(len,data.values())),
                  candidates=[p for p in pairs if p['candidate']],pairs=pairs,
                  model_calls=0,labels_read=0,independence_certified=False)
    save('overlap_report.json',result)
    print(json.dumps(dict(pairs=len(pairs),flagged=len(result['candidates']))),flush=True)

if __name__ == '__main__':
    main()
