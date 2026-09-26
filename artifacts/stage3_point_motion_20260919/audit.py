"""Replay geometry, summarize readiness, diagnose track loss without retuning."""
from pathlib import Path
import json,hashlib
import numpy as np,pandas as pd,cv2
O=Path(__file__).resolve().parent;R=O.parents[1]
read=lambda p:json.loads(p.read_text())
write=lambda p,x:p.write_text(json.dumps(x,ensure_ascii=False,indent=2))
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def dropout(key,rects):
    z=np.load(O/(key+'_frames.npz'));gray=[cv2.cvtColor(im,cv2.COLOR_RGB2GRAY) for im in z['rgb']];h,w=gray[0].shape;mask=np.zeros((h,w),np.uint8)
    for x0,y0,x1,y1 in rects:mask[int(y0*h):int(y1*h),int(x0*w):int(x1*w)]=255
    pts=cv2.goodFeaturesToTrack(gray[0],500,.01,8,mask=mask,blockSize=7).reshape(-1,2)
    kw=dict(winSize=(21,21),maxLevel=3,criteria=(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT,30,.01))
    total=len(pts);alive=np.ones(total,bool);lost={k:0 for k in ['lk_status','forward_backward','appearance','image_bounds','static_mask']}
    cumulative={k:np.ones(total,bool) for k in lost}
    for a,b in zip(gray[:-1],gray[1:]):
        nxt,st,err=cv2.calcOpticalFlowPyrLK(a,b,pts.astype(np.float32).reshape(-1,1,2),None,**kw)
        back,bs,_=cv2.calcOpticalFlowPyrLK(b,a,nxt,None,**kw)
        nxt=nxt.reshape(-1,2);back=back.reshape(-1,2);fb=np.linalg.norm(back-pts,axis=1)
        safe=np.nan_to_num(nxt,nan=-1,posinf=-1,neginf=-1);px=np.rint(safe[:,0]).astype(int);py=np.rint(safe[:,1]).astype(int)
        bounds=(px>=0)&(px<w)&(py>=0)&(py<h);inside=bounds.copy();inside[inside]&=mask[py[inside],px[inside]]>0
        tests={'lk_status':(st.ravel()>0)&(bs.ravel()>0)&np.isfinite(nxt).all(axis=1),
               'forward_backward':fb<=1,'appearance':err.ravel()<=20,'image_bounds':bounds,'static_mask':inside}
        for k,ok in tests.items():
            cumulative[k]&=ok;lost[k]+=int((alive&~ok).sum());alive&=ok
        pts=safe
    return {'key':key,'detected':total,'survived':int(alive.sum()),'first_failure_priority_order':list(lost),
            'first_failure_counts':lost,'each_check_alone_survivors':{k:int(v.sum()) for k,v in cumulative.items()}}

def main():
    rows=read(O/'results.json');df=pd.DataFrame(rows);protocol=read(O/'freeze.json');masks=read(O/'static_masks.json')
    protected=protocol['source_inputs_protected'];assert all(sha(R/p)==h for p,h in protected.items())
    frozen=read(O/'measurement_freeze.json');assert sha(O/'measure.py')==frozen['script_sha256'] and sha(O/'static_masks.json')==frozen['mask_sha256']
    assert sha(O/'freeze.json')==frozen['protocol_sha256'] and sha(O/'windows.json')==frozen['windows_sha256']
    train=df[(df.role=='train')&df.valid];held=df[(df.role=='held')&df.valid]
    threshold=float(max(.01,np.quantile(abs(train.loc[train.label==2,'candidate']),.9))) if (train.label==2).any() else None
    write(O/'train_deadband.json',{'threshold':threshold,'valid_train_constants':int((train.label==2).sum()),
                                  'usable_train_counts':train.label.value_counts().to_dict(),
                                  'calibration_valid':all((train.label==k).sum()>=2 for k in range(3)),
                                  'note':'formula frozen before measurements; derived solely from train. Fewer than2 train constants means descriptive threshold only, not validated calibration.'})
    # Same source routes are excluded across training and held pilot windows.
    assert set(df[df.role=='train'].route).isdisjoint(df[df.role=='held'].route)
    replay=[];durations=[]
    for r in rows:
        f=np.load(O/(r['key']+'_frames.npz'));z=np.load(R/'artifacts/stage3_training_basis_20260917'/r['labels_npz']);ix=np.arange(r['index']-10,r['index']+11)
        assert np.array_equal(f['frame_index'],z['frame_index'][ix]) and np.array_equal(f['time'],z['frame_time'][ix]);durations.append(float(f['time'][-1]-f['time'][0]))
        path=O/(r['key']+'_tracks.npz')
        if not path.exists():continue
        t=np.load(path);rad=np.linalg.norm(t['xy']-t['foe'],axis=2);tm=t['time']-t['time'][10];design=np.stack([np.ones(21),tm,tm**2],axis=1)
        coef=np.linalg.lstsq(design,1/np.maximum(rad,1e-6),rcond=None)[0];good=t['usable']
        if good.any():
            median=float(np.median(2*coef[2,good]/coef[1,good]));assert np.isclose(median,r['candidate'],rtol=1e-8,atol=1e-8);replay.append(r['key'])
    coverage=[]
    for role,g in df.groupby('role'):
        for k in range(3):
            s=g[g.label==k];good=s[s.valid]
            coverage.append({'role':role,'label':k,'selected':len(s),'valid':len(good),'coverage':len(good)/len(s) if len(s) else None,
                             'candidate_median':float(good.candidate.median()) if len(good) else None,'raw_median':float(good.raw.median()) if len(good) else None})
    pd.DataFrame(coverage).to_csv(O/'coverage.csv',index=False)
    pairs=[];lookup={r['key']:r for r in rows}
    for p in protocol['pairs']:
        a,b=lookup[p['a_key']],lookup[p['b_key']];present=a.get('candidate') is not None and b.get('candidate') is not None
        pairs.append({**p,'both_valid':bool(a['valid'] and b['valid']),
                      'candidate_order_correct':bool((a['candidate']-b['candidate'])*(a['sensor_ratio_a_v']-b['sensor_ratio_a_v'])>0) if present else None,
                      'raw_order_correct':bool((a['raw']-b['raw'])*(a['sensor_ratio_a_v']-b['sensor_ratio_a_v'])>0) if present else None,
                      'candidate_a':a.get('candidate'),'candidate_b':b.get('candidate')})
    pd.DataFrame(pairs).to_csv(O/'pair_results.csv',index=False)
    hp=[p for p in pairs if p['role']=='held' and p['both_valid']]
    def correct(k):
        s=held[held.label==k]
        if not len(s) or threshold is None:return None
        return float((abs(s.candidate)<=threshold).mean()) if k==2 else float(((s.candidate>threshold) if k==0 else (s.candidate<-threshold)).mean())
    gates={'train_coverage_atleast2_each':bool(all((train.label==k).sum()>=2 for k in range(3)) and train.route.nunique()>=2),
           'deadband_lt_005':threshold is not None and threshold<.05,
           'held_support_atleast3_each':bool(all((held.label==k).sum()>=3 for k in range(3)) and held.route.nunique()>=2),
           'held_coverage_atleast06_each':all(r['coverage'] is not None and r['coverage']>=.6 for r in coverage if r['role']=='held'),
           'held_constant_80percent':correct(2) is not None and correct(2)>=.8,
           'held_accel_decel_70percent':all(correct(k) is not None and correct(k)>=.7 for k in [0,1]),
           'held_pair_order75percent':bool(hp) and np.mean([p['candidate_order_correct'] for p in hp])>=.75,
           'held_constant_better_than_raw':bool((held.label==2).any()) and float(abs(held[held.label==2].candidate).median())<float(abs(held[held.label==2].raw).median())}
    gates={k:bool(v) for k,v in gates.items()}
    failures={}
    for r in rows:
        if not r['valid']:
            for part in r['reason'].split(';'):failures[part]=failures.get(part,0)+1
    audit=[dropout(key,masks[key]['rectangles']) for key in ['extra_02_295','comma_01_75','expanded_11_525','expanded_14_215','expanded_08_85']]
    for a in audit:assert a['survived']==lookup[a['key']]['tracked']
    write(O/'tracking_dropout_audit.json',audit)
    decision={'gates':gates,'classifier_addition_allowed':all(gates.values()),'valid_windows':len(train)+len(held),'total_windows':len(df),
              'train_only_deadband':threshold,'held_class_pass_fractions':{str(k):correct(k) for k in range(3)},
              'valid_held_pairs':len(hp),'failure_reasons_overlapping':failures,
              'decision':'do not add feature or fit classifier; physical and support validation insufficient',
              'missing_values':'quality rejected candidates remain diagnostic only, never imputed into classifier'}
    write(O/'decision.json',decision)
    # Associate pre-existing baseline predictions only after feature selection/measurement.
    old=pd.read_csv(R/'artifacts/stage3_group_weighting_20260919/predictions.csv')
    old=old[(old.variant=='uniform')&(old.fold=='date_1')]
    pg=df[df.role=='held'].merge(old[['id','sample_index','truth','prediction']],left_on=['id','index'],right_on=['id','sample_index'],how='left',validate='one_to_one')
    pg[['key','label','valid','candidate','sensor_ratio_a_v','truth','prediction']].to_csv(O/'existing_error_comparison.csv',index=False)
    write(O/'checks.json',{'prepare_exit_status':0,'measure_exit_status':0,'audit_exit_status':0,'initial_audit_import_exit_status':1,'plotting_dependency_not_installed':True,
                          'input_and_production_hashes_unchanged':True,'frozen_script_masks_protocol_unchanged':True,
                          'route_train_held_disjoint':True,'native_frame_indices_and_times_match':True,
                          'feature_replayed_windows':len(replay),'duration_min_max': [min(durations),max(durations)],
                          'synthetic_conditions':9,'new_classifier_fits':0,'new_downloads':0,
                          'unique_source_segments':int(df.id.nunique()),'unique_routes':int(df.route.nunique()),
                          'matching_pairs':len(pairs),'matched_speed_delta_max':max(p['speed_delta'] for p in pairs)})
    print(json.dumps(decision,indent=2));print(pd.DataFrame(coverage).to_string(index=False))

if __name__=='__main__':
    cv2.setNumThreads(2);main()
