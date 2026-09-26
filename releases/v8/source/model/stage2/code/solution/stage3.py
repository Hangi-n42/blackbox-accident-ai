"""Independent per-video motion features and frozen Stage 3 classifiers."""
from pathlib import Path
import cv2
import joblib
import numpy as np
import pandas as pd
from scipy.ndimage import uniform_filter1d

ACCEL = np.array(['ACCELERATING', 'DECELERATING', 'CONSTANT', 'STOPPED'])
STEER = np.array(['LEFT', 'STRAIGHT', 'RIGHT'])
VIDEO_EXT = {'.mp4', '.avi', '.mov', '.mkv', '.hevc', '.m4v', '.webm', '.3gp', '.3gpp', '.wmv'}

def transfer_features(features, regressor):
    summary=features.reshape(-1,6,3,4,4,3).mean(axis=(2,3)).reshape(len(features),-1)
    return np.concatenate([regressor.predict(features),summary],axis=1)

def _frame_feature(flow):
    h,w=flow.shape[:2]
    yy,xx=np.mgrid[:h,:w].astype(np.float32)
    xx=(xx-w*.5)/w; yy=(yy-h*.34)/h
    fx=flow[...,0]/w; fy=flow[...,1]/h
    mag=np.hypot(fx,fy)
    radial=(fx*xx+fy*yy)/(xx*xx+yy*yy+.02)
    output=[]
    # Fixed regions avoid dependence on other files or metadata.
    for ya,yb in ((.15,.50),(.50,.72),(.72,.93)):
        for xa,xb in ((.03,.27),(.27,.5),(.5,.73),(.73,.97)):
            s=np.s_[int(ya*h):int(yb*h),int(xa*w):int(xb*w)]
            for v in (fx[s],fy[s],mag[s],radial[s]):
                output.extend((np.median(v),np.percentile(v,20),np.percentile(v,80)))
    return np.nan_to_num(output).astype(np.float32)

def extract_motion(path, source_fps=10., width=256):
    """Return per-decoded-frame features; source FPS is explicit, never MP4 timebase."""
    cap=cv2.VideoCapture(str(path))
    dis=cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_FAST)
    stride=max(1,int(round(source_fps/10)))
    prev=None; features=[]; centers=[]; count=0
    while True:
        ok,frame=cap.read()
        if not ok: break
        idx=count; count+=1
        if idx%stride: continue
        target_h=max(96,int(round(frame.shape[0]*width/frame.shape[1])))
        gray=cv2.cvtColor(cv2.resize(frame,(width,target_h)),cv2.COLOR_BGR2GRAY)
        if prev is not None:
            flow=dis.calc(prev,gray,None)*(source_fps/stride)
            features.append(_frame_feature(flow)); centers.append(idx-stride*.5)
        prev=gray
    cap.release()
    if not count: raise ValueError(f'Cannot decode {Path(path).name}')
    if not features: return np.zeros((count,144*6),np.float32)
    raw=np.array(features)
    # Temporal summaries on a fixed 10Hz grid, then interpolate to all decoded frames.
    parts=[raw]
    for window in (5,15,31):
        parts.append(uniform_filter1d(raw,size=window,axis=0,mode='nearest'))
    smooth=parts[2]
    for lag in (5,15):
        ix=np.arange(len(smooth))
        parts.append((smooth[np.minimum(ix+lag,len(ix)-1)]-smooth[np.maximum(ix-lag,0)])/(2*lag/10))
    combined=np.concatenate(parts,axis=1)
    result=np.empty((count,combined.shape[1]),np.float32)
    for j in range(combined.shape[1]): result[:,j]=np.interp(np.arange(count),centers,combined[:,j])
    return np.nan_to_num(result)

def predict_stage3(data_dir, model_dir):
    model=joblib.load(Path(model_dir)/'motion_model.joblib')
    rows=[]
    old_threads=cv2.getNumThreads(); cv2.setNumThreads(2)
    try:
        for path in sorted((Path(data_dir)/'videos').iterdir()):
            if not path.is_file() or path.suffix.lower() not in VIDEO_EXT: continue
            features=extract_motion(path,source_fps=10.)
            if 'motion_regressor' in model:features=transfer_features(features,model['motion_regressor'])
            accel_features=transfer_features(features,model['accel_motion_regressor']) if 'accel_motion_regressor' in model else features
            steer_features=transfer_features(features,model['steer_motion_regressor']) if 'steer_motion_regressor' in model else features
            accel=model['accel'].predict(accel_features).astype(int)
            steer=model['steer'].predict(steer_features).astype(int)
            rows.extend({'ID':path.stem,'sample_index':i,'accel_label':str(ACCEL[a]),'steer_label':str(STEER[s])} for i,(a,s) in enumerate(zip(accel,steer)))
    finally: cv2.setNumThreads(old_threads)
    return pd.DataFrame(rows,columns=['ID','sample_index','accel_label','steer_label'])
