"""Separate tracker-only contrast from center sensitivity on fixed old tracks."""
from pathlib import Path
import importlib.util,json,time
import numpy as np,pandas as pd,cv2
from PIL import Image,ImageDraw
O=Path(__file__).resolve().parent;R=O.parents[1];P=R/'artifacts/stage3_point_motion_20260919'
spec=importlib.util.spec_from_file_location('previous_measure',P/'measure.py');old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
read,write,sha=old.read,old.write,old.sha
KEYS=['extra_02_295','comma_01_75','expanded_11_525','expanded_14_215','expanded_14_165',
      'expanded_11_225','expanded_19_275','extra_00_75','comma_01_315','expanded_08_85']

def sample(flow,pts):
    return cv2.remap(flow,pts[:,0].astype(np.float32).reshape(-1,1),pts[:,1].astype(np.float32).reshape(-1,1),cv2.INTER_LINEAR,borderMode=cv2.BORDER_REPLICATE).reshape(-1,2)

def patches_error(a,b,p,q):
    return np.array([np.abs(cv2.getRectSubPix(a,(21,21),tuple(map(float,x))).astype(float)-cv2.getRectSubPix(b,(21,21),tuple(map(float,y))).astype(float)).mean() for x,y in zip(p,q)])

def track(rgb,rects):
    gray=[cv2.cvtColor(im,cv2.COLOR_RGB2GRAY) for im in rgb];h,w=gray[0].shape;mask=np.zeros((h,w),np.uint8)
    for x0,y0,x1,y1 in rects:mask[int(y0*h):int(y1*h),int(x0*w):int(x1*w)]=255
    first=cv2.goodFeaturesToTrack(gray[0],maxCorners=500,qualityLevel=.01,minDistance=8,mask=mask,blockSize=7).reshape(-1,2)
    histories={};counts={};alive_sets={};elapsed={}
    for method in ['lk','dis']:
        started=time.perf_counter();pts=first.copy();history=[pts.copy()];alive=np.ones(len(pts),bool);legacy=alive.copy()
        drop={k:0 for k in ['status','fb','photometric','mask']}
        flow1=cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_FAST);flow2=cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_FAST)
        for a,b in zip(gray[:-1],gray[1:]):
            if method=='lk':
                kw=dict(winSize=(21,21),maxLevel=3,criteria=(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT,30,.01))
                nxt,st,err=cv2.calcOpticalFlowPyrLK(a,b,pts.reshape(-1,1,2),None,**kw)
                back,bs,_=cv2.calcOpticalFlowPyrLK(b,a,nxt,None,**kw);nxt=nxt.reshape(-1,2);back=back.reshape(-1,2);status=(st.ravel()>0)&(bs.ravel()>0)
            else:
                f=flow1.calc(a,b,None);rev=flow2.calc(b,a,None);nxt=pts+sample(f,pts);back=nxt+sample(rev,nxt);status=np.ones(len(pts),bool)
            status&=np.isfinite(nxt).all(axis=1)&np.isfinite(back).all(axis=1)
            fb=np.linalg.norm(back-pts,axis=1);safe=np.nan_to_num(nxt,nan=-1,posinf=-1,neginf=-1)
            photo=patches_error(a,b,pts,safe)
            px=np.rint(safe[:,0]).astype(int);py=np.rint(safe[:,1]).astype(int)
            inside=(px>=0)&(px<w)&(py>=0)&(py<h);inside[inside]&=mask[py[inside],px[inside]]>0
            tests={'status':status,'fb':fb<=1,'photometric':photo<=20,'mask':inside}
            for k,ok in tests.items():drop[k]+=int((alive&~ok).sum());alive&=ok
            if method=='lk':legacy&=status&(fb<=1)&(err.ravel()<=20)&inside
            pts=safe.astype(np.float32);history.append(pts.copy())
        histories[method]=np.stack(history);alive_sets[method]=alive
        counts[method]={'detected':len(first),'survived':int(alive.sum()),'first_failure_counts':drop}
        if method=='lk':alive_sets['legacy']=legacy;histories['legacy']=histories['lk']
        elapsed[method]=time.perf_counter()-started
    return histories,alive_sets,counts,elapsed

def geometry(x,t,h,w):
    r={'tracked':x.shape[1],'valid':False,'candidate':None,'raw':None,'reason':'full_tracks_lt15'}
    if x.shape[1]<15:return r,None,np.zeros(x.shape[1],bool)
    center,inside=old.foe_fit(x[0],x[-1]);a,_=old.foe_fit(x[0],x[10]);b,_=old.foe_fit(x[10],x[-1])
    if center is None:r['reason']='no_foe';return r,None,np.zeros(x.shape[1],bool)
    ratio,raw,good,_=old.temporal_feature(x,t,center);good&=inside
    left=int(((x[10,:,0]<center[0])&good).sum());right=int(good.sum())-left
    drift=float(np.linalg.norm(a-b)) if a is not None and b is not None else None
    checks={'usable15':int(good.sum())>=15,'bounds':bool(.15*w<=center[0]<=.85*w and .15*h<=center[1]<=.70*h),
            'both_sides':left>=3 and right>=3,'drift20':drift is not None and drift<=20}
    r.update(foe=center.tolist(),usable=int(good.sum()),left=left,right=right,half_drift=drift,
             candidate=float(np.median(ratio[good])) if good.any() else None,
             raw=float(np.median(raw[good])) if good.any() else None,
             valid=all(checks.values()),reason=';'.join(k for k,v in checks.items() if not v) or 'passed')
    return r,center,good

def frozen_point_feature(x,t,center,good):
    if not good.any():return None
    tt=t-t[10];r=np.linalg.norm(x[:,good]-center,axis=2)
    if np.min(r)<1e-6:return None
    coef=np.linalg.lstsq(np.stack([np.ones(21),tt,tt**2],axis=1),1/r,rcond=None)[0]
    if (abs(coef[1])<1e-12).any():return None
    return float(np.median(2*coef[2]/coef[1]))

def stability(key,x,t,center,good):
    r={'key':key,'n_original_tracks':x.shape[1],'foe_available':center is not None}
    if center is None:return r,[]
    # All coordinates/track identities held fixed. Dispersion is not statistical CI.
    rng=np.random.default_rng(20260920);centers=[]
    for _ in range(50):
        ix=rng.choice(x.shape[1],x.shape[1],replace=True);c,_=old.foe_fit(x[0,ix],x[-1,ix])
        if c is not None:centers.append(c)
    samples=np.array(centers);dist=np.linalg.norm(samples-center,axis=1) if len(samples) else np.array([])
    delta=x[-1]-x[0];norm=np.linalg.norm(delta,axis=1);keep=norm>3
    n=np.stack([-delta[keep,1],delta[keep,0]],axis=1)/norm[keep,None]
    residual=abs(np.sum(n*x[0,keep],axis=1)-n@center);inliers=residual<=3
    condition=float(np.linalg.cond(n[inliers])) if inliers.sum()>=2 else None
    halves={}
    for name,ix in [('left',x[10,:,0]<center[0]),('right',x[10,:,0]>=center[0]),
                    ('top',x[10,:,1]<np.median(x[10,:,1])),('bottom',x[10,:,1]>=np.median(x[10,:,1]))]:
        c,_=old.foe_fit(x[0,ix],x[-1,ix]);halves[name]={'n':int(ix.sum()),'foe':c.tolist() if c is not None else None,
                                                           'distance':float(np.linalg.norm(c-center)) if c is not None else None}
    base=frozen_point_feature(x,t,center,good);perturb=[]
    for pixels in [2,5,10]:
        for axis in [0,1]:
            for sign in [-1,1]:
                offset=np.zeros(2);offset[axis]=sign*pixels;v=frozen_point_feature(x,t,center+offset,good)
                perturb.append({'key':key,'pixels':pixels,'axis':axis,'sign':sign,'value':v,'base':base,
                                'absolute_delta':abs(v-base) if v is not None and base is not None else None,
                                'sign_flipped':bool(v*base<0) if v is not None and base is not None else None})
    r.update(base=base,original_center=center.tolist(),bootstrap_success=len(samples),
             bootstrap_p50_distance=float(np.median(dist)) if len(dist) else None,
             bootstrap_p90_distance=float(np.quantile(dist,.9)) if len(dist) else None,
             line_condition_number=condition,spatial_subsets=halves,
             residual_median=float(np.median(residual[inliers])) if inliers.any() else None)
    np.savez_compressed(O/(key+'_center_bootstrap.npz'),centers=samples,original=center)
    return r,perturb

def main():
    assert not (O/'freeze.json').exists();start=time.perf_counter()
    rows={r['key']:r for r in read(P/'results.json')};masks=read(P/'static_masks.json')
    protected=read(P/'freeze.json')['source_inputs_protected']
    inputs={str((P/n).relative_to(R)):sha(P/n) for n in ['measure.py','results.json','static_masks.json','windows.json']}
    for k in KEYS:inputs[str((P/(k+'_frames.npz')).relative_to(R))]=sha(P/(k+'_frames.npz'))
    assert all(sha(R/p)==h for p,h in protected.items())
    write(O/'freeze.json',{'script_sha256':sha(Path(__file__)),'inputs':inputs,'protected':protected,'keys':KEYS,
        'selection':'5 previously named failure types: night/track loss, daytime track loss, bad center, half-center drift, one-sided support;5 old accepted controls A/D/C incl held D/C; known development cases, no independent claim',
        'tracker_control':'LK legacy replay + LK common gate vs DIS_FAST dense-flow point advection common gate; identical640gray21frames, initial corners, static masks,21x21 bilinear mean absolute patch error<=20,forward-backward<=1px,geometric thresholds, feature formula. LK status retained; DIS finite flow. LK internal error used ONLY legacy check. DIS default FAST parameters no sweeps.',
        'foe_control':'Original legacy LK tracks only; original robust FOE fixed method;50 point bootstrap samples seed20260920, left/right/top/bottom fits, normal-matrix condition, center offsets +-2/5/10px in x/y with same original accepted point identities. No retracking, no sensor fit, no modified pose or rotation compensation.',
        'diagnostic_gates':'Prefer candidate only if common-gate geometry coverage improves, no lost valid controls, no new misleading A/D direction or C outside prior train deadband .030443026891414933. This10window sample cannot promote classifier; report common-initial-survivor intersection control.',
        'limitations':'Fixed manual masks, no independent point/pose truth, bootstrap dispersion not confidence interval, candidate sign != calibrated acceleration; old all28 readiness not established; no classifier fit/download/production change',
        'dis_reference':'https://docs.opencv.org/4.x/da/d06/classcv_1_1DISOpticalFlow.html'})
    results=[];stabs=[];perturbs=[];pictures=[]
    for key in KEYS:
        r=rows[key];z=np.load(P/(key+'_frames.npz'));rgb=z['rgb'];t=z['time'];h,w=rgb.shape[1:3]
        hist,alive,counts,times=track(rgb,masks[key]['rectangles']);cache={}
        common=alive['lk']&alive['dis']
        for variant,method,mask in [('legacy','legacy',alive['legacy']),('lk_common','lk',alive['lk']),('dis_common','dis',alive['dis']),
                                    ('lk_intersection','lk',common),('dis_intersection','dis',common)]:
            x=hist[method][:,mask];g,c,good=geometry(x,t,h,w);cache[variant]=(x,c,good)
            results.append({'key':key,'variant':variant,'role':r['role'],'label':r['label'],'truth_ratio':r['sensor_ratio_a_v'],
                            **g,'detected':counts['lk']['detected'],'matched_initial_points':int(common.sum())})
            np.savez_compressed(O/f'{key}_{variant}.npz',xy=x,time=t,initial_point_indices=np.flatnonzero(mask),
                                center=np.array(c) if c is not None else np.full(2,np.nan),usable=good)
            if variant=='legacy':
                assert g['tracked']==r['tracked'] and g['valid']==r['valid']
                if r.get('candidate') is not None:assert np.isclose(g['candidate'],r['candidate'],atol=1e-10)
                source=P/(key+'_tracks.npz')
                if source.exists():assert np.array_equal(x,np.load(source)['xy'])
            if variant in ['lk_common','dis_common']:
                im=Image.fromarray(rgb[10]);draw=ImageDraw.Draw(im)
                for j in range(x.shape[1]):draw.line([tuple(map(float,v)) for v in x[:,j]],fill='lime' if good[j] else 'orange',width=2)
                if c is not None:
                    a,b=c;draw.ellipse((a-4,b-4,a+4,b+4),fill='red')
                canvas=Image.new('RGB',(640,h+32),'white');canvas.paste(im,(0,32));d=ImageDraw.Draw(canvas)
                d.text((3,3),f"{key} {variant}: n={g['tracked']} valid={g['valid']}",fill='black');pictures.append(canvas)
        x,c,good=cache['legacy'];s,pp=stability(key,x,t,c,good);stabs.append(s);perturbs.extend(pp)
        write(O/(key+'_tracking_counts.json'),{'counts':counts,'seconds':times})
        print(key,[(v,results[-5+i]['tracked'],results[-5+i]['valid']) for i,v in enumerate(['legacy','lk','dis','lk_intersection','dis_intersection'])],flush=True)
    write(O/'results.json',results);pd.DataFrame(results).to_csv(O/'results.csv',index=False)
    write(O/'center_stability.json',stabs);pd.DataFrame(perturbs).to_csv(O/'center_sensitivity.csv',index=False)
    for i in range(0,len(pictures),4):
        canvas=Image.new('RGB',(1280,2*pictures[0].height),'white')
        for j,im in enumerate(pictures[i:i+4]):canvas.paste(im,((j%2)*640,(j//2)*im.height))
        canvas.save(O/f'comparison_{i//4}.jpg')
    assert all(sha(R/p)==h for p,h in {**inputs,**protected}.items())
    assert sha(Path(__file__))==read(O/'freeze.json')['script_sha256']
    write(O/'checks.json',{'completed':True,'legacy_replay_exact':True,'frozen_inputs_production_unchanged':True,
                          'same_initial_points':True,'tracker_common_photometric_gate':True,'foe_uses_only_legacy_tracks':True,
                          'classifier_fits':0,'downloads':0,'windows':len(KEYS),'elapsed_seconds':time.perf_counter()-start})

if __name__=='__main__':
    cv2.setNumThreads(2);main()
