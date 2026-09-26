"""Align existing CAN signals to 10 Hz without extrapolation; keep raw data intact."""
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'artifacts/data_pilot_20260916/comma'


def align(t, values, query):
    t, values = np.asarray(t).ravel(), np.asarray(values).ravel()
    assert len(t) == len(values) and np.isfinite(t).all() and np.isfinite(values).all()
    assert np.all(np.diff(t) >= 0)
    # Keep the last CAN observation when several share a timestamp.
    keep = np.r_[np.diff(t) != 0, True]
    t, values = t[keep], values[keep]
    valid = (query >= t[0]) & (query <= t[-1])
    right = np.searchsorted(t, query, side='left').clip(1, len(t)-1)
    valid &= (t[right]-t[right-1] <= .1)
    result = np.interp(query, t, values)
    result[~valid] = np.nan
    return result, valid, int((~keep).sum())


def main():
    OUT.mkdir(exist_ok=True)
    exclusions = {x['segment'] for x in json.loads((ROOT/'research/stage3_external_overlap.json').read_text())['excluded']}
    records = []
    for manifest in sorted((ROOT/'external_data/comma2k19').glob('*_manifest.json')):
        for row in json.loads(manifest.read_text()):
            if row['segment'] in exclusions:
                continue
            # Legacy manifests contain Windows paths; resolve from the segment instead.
            chunk, route, segment = row['segment'].split('/')
            base = ROOT/'external_data/comma2k19'/chunk/route.replace('|','_')/segment
            ft = np.load(base/'global_pose/frame_times').ravel()
            query = ft[0]+np.arange(int(np.floor((ft[-1]-ft[0])*10))+1)/10
            right = np.searchsorted(ft, query).clip(1, len(ft)-1)
            frame = np.where(query-ft[right-1] <= ft[right]-query, right-1, right)
            payload = {'time':query, 'frame_index':frame, 'frame_time':ft[frame]}
            record = {'source_group':row['route'],'segment':row['segment'],'samples':len(query),'signals':{}}
            for name in ('speed','steering_angle'):
                signal = base/'processed_log/CAN'/name
                value, valid, removed = align(np.load(signal/'t'), np.load(signal/'value'), query)
                payload[name], payload[name+'_valid'] = value, valid
                record['signals'][name] = {'valid':int(valid.sum()),'duplicate_timestamps_removed':removed}
            name = row['segment'].replace('/','_').replace('|','_')+'.npz'
            np.savez_compressed(OUT/name, **payload)
            record['path'] = str((OUT/name).relative_to(ROOT))
            records.append(record)
    assert len(records) == 23
    (OUT/'manifest.json').write_text(json.dumps({'policy':'10 Hz native-time grid; last duplicate observation; no endpoint extrapolation; mask gaps > 0.1s (observed maximum < 0.04s). Signals only, not official class labels.', 'records':records},indent=2))
    print('aligned_segments',len(records),'samples',sum(x['samples'] for x in records))


if __name__ == '__main__':
    value, valid, removed = align([0,0,.05,.10],[1,2,3,4],np.array([-.1,0,.025,.1,.2]))
    assert removed == 1 and valid.tolist() == [False,True,True,True,False]
    assert np.allclose(value[valid],[2,2.5,4])
    main()
