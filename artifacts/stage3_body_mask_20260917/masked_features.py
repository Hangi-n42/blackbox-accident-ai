"""Conservative same-camera-agnostic body interior exclusion from ROI statistics."""
import numpy as np
from solution.stage3_v5_compatible import FrameFeatureComputer as Base
POLYGON_X=[0,.15,.5,.85,1]
POLYGON_Y=[.90,.86,.84,.86,.90]
class BodyFeatureComputer(Base):
 def __call__(self,flow):
  original=super().__call__(flow).reshape(12,4,3).copy();h,w=flow.shape[:2]
  fx=flow[...,0]/w;fy=flow[...,1]/h;mag=np.hypot(fx,fy);radial=(fx*self.xx+fy*self.yy)/(self.xx*self.xx+self.yy*self.yy+.02);channels=np.stack((fx,fy,mag,radial))
  boundary=np.interp(np.arange(w)/w,POLYGON_X,POLYGON_Y)*h;keep=np.arange(h)[:,None]<boundary[None,:]
  for i,(ya,yb,xa,xb) in enumerate(self.rois):
   mask=keep[ya:yb,xa:xb]
   if mask.all():continue
   assert mask.any(),'Empty ROI forbidden; no zero filling'
   values=channels[:,ya:yb,xa:xb][:,mask];original[i,:,0]=np.median(values,axis=1);original[i,:,1:]=np.percentile(values,(20,80),axis=1).T
  return np.nan_to_num(original.reshape(-1)).astype(np.float32)
