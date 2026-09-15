"""Equivalent Stage 3 candidate: grouped ROI quantiles and video-local geometry."""
from pathlib import Path
import cv2
import joblib
import numpy as np
import pandas as pd
from scipy.ndimage import uniform_filter1d
from .stage3 import ACCEL, STEER, VIDEO_EXT, transfer_features


class FrameFeatureComputer:
    """Cache only geometry, owned by one extract_motion/video invocation."""
    def __init__(self):
        self.shape=None

    def __call__(self,flow):
        h,w=flow.shape[:2]
        if self.shape!=(h,w):
            yy,xx=np.mgrid[:h,:w].astype(np.float32)
            self.xx=(xx-w*.5)/w
            self.yy=(yy-h*.34)/h
            self.rois=[(int(ya*h),int(yb*h),int(xa*w),int(xb*w))
                       for ya,yb in ((.15,.50),(.50,.72),(.72,.93))
                       for xa,xb in ((.03,.27),(.27,.5),(.5,.73),(.73,.97))]
            self.shape=(h,w)
        xx,yy=self.xx,self.yy
        fx=flow[...,0]/w;fy=flow[...,1]/h
        mag=np.hypot(fx,fy)
        radial=(fx*xx+fy*yy)/(xx*xx+yy*yy+.02)
        channels=np.stack((fx,fy,mag,radial))
        output=np.empty((12,4,3),np.float64)
        for index,(ya,yb,xa,xb) in enumerate(self.rois):
            values=channels[:,ya:yb,xa:xb].reshape(4,-1)
            # Keep float32 median arithmetic separate from float64 quantiles.
            output[index,:,0]=np.median(values,axis=1)
            quantiles=np.percentile(values,(20,80),axis=1)
            output[index,:,1]=quantiles[0]
            output[index,:,2]=quantiles[1]
        return np.nan_to_num(output.reshape(-1)).astype(np.float32)


def _frame_feature(flow):
    return FrameFeatureComputer()(flow)


def extract_motion(path,source_fps=10.,width=256):
    cap=cv2.VideoCapture(str(path))
    dis=cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_FAST)
    feature_computer=FrameFeatureComputer()
    stride=max(1,int(round(source_fps/10)))
    prev=None;features=[];centers=[];count=0
    while True:
        ok,frame=cap.read()
        if not ok:break
        idx=count;count+=1
        if idx%stride:continue
        target_h=max(96,int(round(frame.shape[0]*width/frame.shape[1])))
        gray=cv2.cvtColor(cv2.resize(frame,(width,target_h)),cv2.COLOR_BGR2GRAY)
        if prev is not None:
            flow=dis.calc(prev,gray,None)*(source_fps/stride)
            features.append(feature_computer(flow));centers.append(idx-stride*.5)
        prev=gray
    cap.release()
    if not count:raise ValueError(f'Cannot decode {Path(path).name}')
    if not features:return np.zeros((count,144*6),np.float32)
    raw=np.array(features)
    parts=[raw]
    for window in (5,15,31):
        parts.append(uniform_filter1d(raw,size=window,axis=0,mode='nearest'))
    smooth=parts[2]
    for lag in (5,15):
        ix=np.arange(len(smooth))
        parts.append((smooth[np.minimum(ix+lag,len(ix)-1)]-smooth[np.maximum(ix-lag,0)])/(2*lag/10))
    combined=np.concatenate(parts,axis=1)
    result=np.empty((count,combined.shape[1]),np.float32)
    for j in range(combined.shape[1]):result[:,j]=np.interp(np.arange(count),centers,combined[:,j])
    return np.nan_to_num(result)


def predict_stage3(data_dir,model_dir):
    model=joblib.load(Path(model_dir)/'motion_model.joblib')
    rows=[]
    old_threads=cv2.getNumThreads();cv2.setNumThreads(2)
    try:
        for path in sorted((Path(data_dir)/'videos').iterdir()):
            if not path.is_file() or path.suffix.lower() not in VIDEO_EXT:continue
            features=extract_motion(path,source_fps=10.)
            if 'motion_regressor' in model:features=transfer_features(features,model['motion_regressor'])
            accel_features=transfer_features(features,model['accel_motion_regressor']) if 'accel_motion_regressor' in model else features
            steer_features=transfer_features(features,model['steer_motion_regressor']) if 'steer_motion_regressor' in model else features
            accel=model['accel'].predict(accel_features).astype(int)
            steer=model['steer'].predict(steer_features).astype(int)
            rows.extend({'ID':path.stem,'sample_index':i,'accel_label':str(ACCEL[a]),'steer_label':str(STEER[s])} for i,(a,s) in enumerate(zip(accel,steer)))
    finally:cv2.setNumThreads(old_threads)
    return pd.DataFrame(rows,columns=['ID','sample_index','accel_label','steer_label'])
