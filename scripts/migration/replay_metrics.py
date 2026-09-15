"""Recompute last experiment's contact metrics with portable evidence paths, no model."""
import argparse
import hashlib
import json
from pathlib import Path
from paths import ROOT,resolve_recorded

def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--run',default='research/v7/native_video_run')
    args=ap.parse_args();folder=resolve_recorded(args.run)
    report=read(folder/'report.json');evaluation=read(folder/'evaluation.json')
    if evaluation['report_sha256']!=sha(folder/'report.json'):raise ValueError('Frozen report SHA differs')
    by={r['ID']:r for r in report['videos']};errors={k:[] for k in ['baseline','candidate']}
    for row in evaluation['videos']:
        path=resolve_recorded(row['review_path'])
        if sha(path)!=row['review_sha256']:raise ValueError('Review SHA differs')
        draft=read(path);result=by[row['ID']]
        if draft['source_video_sha256']!=result['source_sha256']:raise ValueError('Source binding differs')
        label=draft['review']['contact'];times={p['frame']:p['pts_seconds'] for p in result['frame_pts']}
        if times[label['frame']]!=label['pts_seconds']:raise ValueError('Native time mapping differs')
        for kind in errors:errors[kind].append(abs(times[result[kind]['collision_frame']]-label['pts_seconds']))
    summary={k:{'n':len(v),'correct':sum(x<=.3+1e-12 for x in v),'mae':sum(v)/len(v)} for k,v in errors.items()}
    for k,v in summary.items():
        expected=evaluation['scores'][k]['contact']
        if v['correct']!=expected['correct'] or abs(v['mae']-expected['mae'])>1e-10:raise ValueError('Metric reproduction failed')
    print(json.dumps({'status':'passed','contact':summary,'scope':'Archived exposed human drafts; no new accuracy evaluation; source video bytes not reread'},indent=2))

if __name__=='__main__':main()
