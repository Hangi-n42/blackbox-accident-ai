"""Video-only same-point projective temporal diagnostic. No classifier fit."""
from pathlib import Path
import json,hashlib,time
import numpy as np
import pandas as pd
import cv2
from PIL import Image,ImageDraw
O=Path(__file__).resolve().parent
read=lambda p:json.loads(p.read_text())
write=lambda p,x:p.write_text(json.dumps(x,ensure_ascii=False,indent=2))
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def foe_fit(start,end):
    delta=end-start;dist=np.linalg.norm(delta,axis=1);keep=dist>3
    start=start[keep];delta=delta[keep];dist=dist[keep]
    if len(start)<8:return None,np.zeros(len(keep),bool)
    a=np.stack([-delta[:,1],delta[:,0]],axis=1)/dist[:,None]
    b=np.sum(a*start,axis=1);rng=np.random.default_rng(42);best=None
    for _ in range(300):
        ix=rng.choice(len(start),2,replace=False)
        if abs(np.linalg.det(a[ix]))<.05:continue
        center=np.linalg.solve(a[ix],b[ix]);res=abs(a@center-b)
        score=(int((res<=3).sum()),-float(np.minimum(res,3).sum()))
        if best is None or score>best[0]:best=(score,center)
    if best is None:return None,np.zeros(len(keep),bool)
    center=best[1]
    for _ in range(6):
        res=abs(a@center-b);ok=res<=3
        if ok.sum()<8 or np.linalg.cond(a[ok])>100:return None,np.zeros(len(keep),bool)
        center=np.linalg.lstsq(a[ok],b[ok],rcond=None)[0]
    mask=np.zeros(len(keep),bool);mask[np.flatnonzero(keep)]=abs(a@center-b)<=3
    return center,mask

def temporal_feature(tracks,t,center):
    r=np.linalg.norm(tracks-center,axis=2)
    u=1/np.maximum(r,1e-6);tt=t-t[len(t)//2]
    design=np.stack([np.ones(len(t)),tt,tt**2],axis=1)
    cu=np.linalg.lstsq(design,u,rcond=None)[0]
    cr=np.linalg.lstsq(design,r,rcond=None)[0]
    residual=np.sqrt(np.mean((design@cu-u)**2,axis=0))/(np.ptp(u,axis=0)+1e-12)
    good=(r.min(axis=0)>=25)&(cu[1]<-1e-5)&(cr[1]>0)&(residual<=.1)
    ratio=np.full(r.shape[1],np.nan);raw=ratio.copy()
    ratio[good]=2*cu[2,good]/cu[1,good];raw[good]=2*cr[2,good]/cr[1,good]
    return ratio,raw,good,residual

def synthetic_check():
    t=np.arange(-10,11)/10;center=np.array([320.,150.]);depth=np.linspace(45,120,40)
    angle=np.linspace(.1,3.04,40);xy=15*np.stack([np.cos(angle),np.sin(angle)],axis=1)
    checks=[]
    for speed in [8.,16.,28.]:
        for acc in [-1.,0.,1.]:
            zz=depth[None,:]-speed*t[:,None]-.5*acc*t[:,None]**2
            tracks=center+400*xy[None,:,:]/zz[:,:,None]
            fc,inlier=foe_fit(tracks[0],tracks[-1]);assert fc is not None and np.max(abs(fc-center))<1e-8
            candidate,raw,good,_=temporal_feature(tracks,t,fc)
            assert good.sum()>=15 and np.max(abs(candidate[good]-acc/speed))<1e-9
            if acc==0:assert np.median(raw[good])>.1
            checks.append({'speed':speed,'a':acc,'truth_ratio':acc/speed,'candidate':float(np.median(candidate[good])),'raw_radius_change':float(np.median(raw[good])),'n':int(good.sum())})
    return checks

def measure(row,rectangles):
    z=np.load(O/(row['key']+'_frames.npz'));rgb=z['rgb'];t=z['time'];gray=[cv2.cvtColor(im,cv2.COLOR_RGB2GRAY) for im in rgb];h,w=gray[0].shape
    mask=np.zeros((h,w),np.uint8)
    for x0,y0,x1,y1 in rectangles:
        mask[int(y0*h):int(y1*h),int(x0*w):int(x1*w)]=255
    pts=cv2.goodFeaturesToTrack(gray[0],maxCorners=500,qualityLevel=.01,minDistance=8,mask=mask,blockSize=7)
    if pts is None:return {'reason':'no corners','valid':False,'tracked':0},None
    points=pts.reshape(-1,2);history=[points.copy()];fbmax=np.zeros(len(points));alive=np.ones(len(points),bool)
    kw=dict(winSize=(21,21),maxLevel=3,criteria=(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT,30,.01))
    for j in range(1,len(gray)):
        nxt,status,err=cv2.calcOpticalFlowPyrLK(gray[j-1],gray[j],points.astype(np.float32).reshape(-1,1,2),None,**kw)
        back,bs,_=cv2.calcOpticalFlowPyrLK(gray[j],gray[j-1],nxt,None,**kw)
        nxt=nxt.reshape(-1,2);back=back.reshape(-1,2);fb=np.linalg.norm(back-points,axis=1)
        safe=np.nan_to_num(nxt,nan=-1,posinf=-1,neginf=-1);px=np.rint(safe[:,0]).astype(int);py=np.rint(safe[:,1]).astype(int)
        inside=(px>=0)&(px<w)&(py>=0)&(py<h);inside[inside]&=mask[py[inside],px[inside]]>0
        alive&=(status.ravel()>0)&(bs.ravel()>0)&np.isfinite(nxt).all(axis=1)&(fb<=1)&(err.ravel()<=20)&inside
        fbmax=np.maximum(fbmax,np.nan_to_num(fb,nan=1e6));points=safe;history.append(points.copy())
    tracks=np.stack(history)[:,alive];n=int(alive.sum());result={'detected':len(pts),'tracked':n,'valid':False,'reason':'fewer than15 full tracks','candidate':None,'raw':None}
    if n<15:return result,{'rgb':rgb,'tracks':tracks,'center':None,'accepted':np.zeros(n,bool)}
    center,inlier=foe_fit(tracks[0],tracks[-1]);a,_=foe_fit(tracks[0],tracks[10]);b,_=foe_fit(tracks[10],tracks[-1])
    result['reason']='unstable or unavailable expansion center'
    if center is None:return result,{'rgb':rgb,'tracks':tracks,'center':None,'accepted':np.zeros(n,bool)}
    ratio,raw,good,res=temporal_feature(tracks,t,center);good&=inlier
    nvalid=int(good.sum());left=int(((tracks[10,:,0]<center[0])&good).sum());right=int(((tracks[10,:,0]>=center[0])&good).sum())
    drift=float(np.linalg.norm(a-b)) if a is not None and b is not None else None
    checks={'n15':nvalid>=15,'center_bounds':bool(.15*w<=center[0]<=.85*w and .15*h<=center[1]<=.70*h),
            'both_sides':left>=3 and right>=3,'half_center_drift20':drift is not None and drift<=20}
    result.update(usable_tracks=nvalid,foe=center.tolist(),half_center_drift=drift,left_tracks=left,right_tracks=right,
                  geometry_checks=checks,valid=all(checks.values()),reason=';'.join(k for k,v in checks.items() if not v) or 'passed',
                  candidate=float(np.median(ratio[good])) if nvalid else None,raw=float(np.median(raw[good])) if nvalid else None)
    np.savez_compressed(O/(row['key']+'_tracks.npz'),xy=tracks,time=t,foe=center,inlier=inlier,usable=good,
                        inverse_radius_ratio=ratio,raw_radius_ratio=raw,normalized_fit_residual=res)
    return result,{'rgb':rgb,'tracks':tracks,'center':center,'accepted':good}

def main():
    assert not (O/'results.json').exists()
    masks=read(O/'static_masks.json');rows=read(O/'windows.json');start=time.perf_counter()
    write(O/'measurement_freeze.json',{'script_sha256':sha(Path(__file__)),'mask_sha256':sha(O/'static_masks.json'),
                                     'protocol_sha256':sha(O/'freeze.json'),'windows_sha256':sha(O/'windows.json')})
    write(O/'synthetic_checks.json',synthetic_check());out=[];panels=[]
    for row in sorted(rows,key=lambda r:(r['role']!='train',r['key'])):
        m=masks[row['key']]
        if not m['rectangles']:result={'valid':False,'reason':'visual static support unavailable','candidate':None,'raw':None};pic=None
        else:result,pic=measure(row,m['rectangles'])
        out.append({**row,**result,'mask_review':m['reason']})
        if pic is not None:
            im=Image.fromarray(pic['rgb'][10]);draw=ImageDraw.Draw(im)
            for k in range(pic['tracks'].shape[1]):
                path=[tuple(map(float,v)) for v in pic['tracks'][:,k]]
                draw.line(path,fill=(0,255,0) if pic['accepted'][k] else (255,120,0),width=2)
            if pic['center'] is not None:
                x,y=pic['center'];draw.ellipse((x-4,y-4,x+4,y+4),fill='red')
            for x0,y0,x1,y1 in m['rectangles']:draw.rectangle((x0*im.width,y0*im.height,x1*im.width,y1*im.height),outline='cyan',width=2)
            canvas=Image.new('RGB',(640,im.height+35),'white');canvas.paste(im,(0,35));d=ImageDraw.Draw(canvas)
            d.text((4,4),row['key']+' valid='+str(result['valid'])+' '+result['reason'],fill='black');panels.append(canvas)
        print(row['key'],result,flush=True)
    for j in range(0,len(panels),4):
        pp=panels[j:j+4];img=Image.new('RGB',(1280,2*pp[0].height),'white')
        for k,im in enumerate(pp):img.paste(im,((k%2)*640,(k//2)*im.height))
        img.save(O/f'tracks_review_{j//4}.jpg')
    write(O/'results.json',out);pd.DataFrame(out).to_csv(O/'results.csv',index=False)
    write(O/'runtime.json',{'seconds':time.perf_counter()-start,'windows':len(rows),'opencv':cv2.__version__,'classifier_fits':0})

if __name__=='__main__':
    cv2.setNumThreads(2);main()
