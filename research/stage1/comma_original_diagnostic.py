"""Frozen Stage1 false-positive audit on existing comma2k19 original road videos."""
from pathlib import Path
import sys
import json
import hashlib
import shutil
import time
from collections import Counter
import cv2
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from solution.stage1 import extract_features,feature_probability
from solution.stage1_tpo_merged import TPODetector

OUT=Path(__file__).resolve().parent/'comma_original_diagnostic'
OUT.mkdir(exist_ok=True)
DATA=ROOT/'external_data/comma2k19'
MODEL=ROOT/'model/stage1'


def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()


def save(name,x):
    (OUT/name).write_text(json.dumps(x,indent=2),encoding='utf-8')


def sequential_frames(path,initial_n):
    def run(n):
        targets=np.unique(np.linspace(0,n-1,min(12,n)).round().astype(int)).tolist()
        desired=set(targets); frames={}; i=0
        cap=cv2.VideoCapture(str(path))
        if not cap.isOpened():raise RuntimeError(f'Cannot open {path}')
        reported={'frame_count':cap.get(cv2.CAP_PROP_FRAME_COUNT),'fps':cap.get(cv2.CAP_PROP_FPS)}
        while cap.grab():
            if i in desired:
                ok,bgr=cap.retrieve()
                if not ok:raise RuntimeError(f'Retrieve failed at {i}')
                frames[i]=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)
            i+=1
        cap.release()
        return i,targets,frames,reported
    actual,targets,frames,reported=run(initial_n)
    passes=1
    if actual!=initial_n:
        actual2,targets,frames,_=run(actual);passes=2
        if actual2!=actual:raise RuntimeError('Unstable sequential decode count')
    if len(frames)!=len(targets):raise RuntimeError('Missing target frames')
    return [frames[i] for i in targets], {'method':'sequential HEVC decode; linspace index rule identical to Stage1 with actual decoded total',
            'passes':passes,'initial_count_from_frame_times':initial_n,'decoded_total':actual,
            'reported_opencv':reported,'selected_zero_based_indices':targets,
            'decoded_rgb_sha256':[hashlib.sha256(frames[i].tobytes()).hexdigest() for i in targets]}


def main():
    cv2.setNumThreads(2)
    excluded=json.loads((ROOT/'research/stage3_external_overlap.json').read_text())
    exclude={x['segment'] for x in excluded['excluded']}
    manifest_files=sorted(DATA.glob('*_manifest.json'))
    selected=[r for p in manifest_files for r in json.loads(p.read_text()) if r['segment'] not in exclude]
    assert len(selected)==23 and len({r['route'] for r in selected})==23
    inventory=json.loads((DATA/'inventory.json').read_text())
    inv={x['path']:x for x in inventory['files']}
    cfg=json.loads((MODEL/'config.json').read_text());artifact=MODEL/cfg['forensic_artifact']
    forensic=json.loads(artifact.read_text())
    assert cfg['threshold']==.5 and cfg['forensic_weight']==.5
    assets=[MODEL/'config.json',artifact,MODEL/'tpo/merged_visual.pt',ROOT/'solution/stage1.py',ROOT/'solution/stage1_tpo_merged.py']
    frozen={str(p.relative_to(ROOT)):sha(p) for p in assets}
    sources=[]
    for r in selected:
        path=ROOT/r['local']/'video.hevc';times=ROOT/r['local']/'global_pose/frame_times'
        video_sha=sha(path); key=str(path.relative_to(DATA))
        assert inv[key]['sha256']==video_sha
        sources.append(r|{'video_path':str(path.relative_to(ROOT)),'video_sha256':video_sha,
                         'frame_times_sha256':sha(times),'original_label_basis':'official road-facing EON camera video in raw dataset; no recapture transformation performed',
                         'inventory_video_hash_verified':True})
    save('sources.json',{'official_repository':'https://github.com/commaai/comma2k19',
                        'official_dataset_card':'https://huggingface.co/datasets/commaai/comma2k19',
                        'license':'MIT, official dataset-card designation plus preserved license text',
                        'source_metadata_sha256':{str(p.relative_to(ROOT)):sha(p) for p in manifest_files+[DATA/'inventory.json',DATA/'LICENSE',ROOT/'research/stage3_external_overlap.json']},
                        'excluded':excluded,'videos':sources})
    shutil.copyfile(DATA/'LICENSE',OUT/'comma2k19_LICENSE')
    save('frozen_manifest.json',{'configuration':cfg,'artifact_sha256':frozen,'thresholds':{'tpo':.5,'forensic':.5,'blend':.5},
                               'scope':'original-only false positives; no Macro-F1, no cross-domain pooled metric',
                               'model_or_threshold_changes':False,'new_training':False})
    results=[];detector=TPODetector(MODEL/'tpo');start=time.monotonic()
    try:
        for i,r in enumerate(sources):
            before=time.monotonic();path=ROOT/r['video_path']
            frame_times=np.load(ROOT/r['local']/'global_pose/frame_times').ravel()
            frames,sampling=sequential_frames(path,len(frame_times))
            x,_=extract_features(frames);fp=feature_probability(x,forensic)
            tpo_per_frame=detector.score(frames);tp=float(tpo_per_frame.mean())
            row={'segment':r['segment'],'route':r['route'],'vehicle':'RAV4' if 'Chunk_1/' in r['segment'] else 'Civic',
                 'y_true':'ORIGINAL','frame_sampling':sampling,'image_shapes':[list(f.shape) for f in frames],
                 'tpo_per_frame':tpo_per_frame.tolist(),'tpo':tp,'forensic':fp,'blend':.5*tp+.5*fp,
                 'seconds':time.monotonic()-before}
            row['false_positives']={k:bool(row[k]>=.5) for k in ['tpo','forensic','blend']}
            results.append(row);save('predictions.json',results)
            print(i+1,'/23',r['segment'], 'TPO',round(tp,4),'forensic',round(fp,4),'blend',round(row['blend'],4),flush=True)
    finally:
        detector.close()
    def summary(rows):
        return {k:{'n_original_videos':len(rows),'false_positive_count':sum(r['false_positives'][k] for r in rows),
                   'false_positive_rate':sum(r['false_positives'][k] for r in rows)/len(rows)} for k in ['tpo','forensic','blend']}
    unchanged={str(p.relative_to(ROOT)):sha(p)==frozen[str(p.relative_to(ROOT))] for p in assets}
    assert all(unchanged.values())
    save('metrics.json',{'overall':summary(results),'by_vehicle':{v:summary([r for r in results if r['vehicle']==v]) for v in ['RAV4','Civic']},
                         'seconds':time.monotonic()-start,'n_images':sum(len(r['image_shapes']) for r in results),
                         'decode_methods':dict(Counter(r['frame_sampling']['method'] for r in results)),
                         'model_code_hashes_unchanged':unchanged,'limitations':['Original-only: no recapture recall or overall Macro-F1','Two vehicles and geographically restricted commute dataset','Aligned public duplicate excluded; full route/source overlap cannot be ruled out','No metric pooled with DLC']})
    print(json.dumps(summary(results),indent=2),flush=True)


if __name__=='__main__':main()
