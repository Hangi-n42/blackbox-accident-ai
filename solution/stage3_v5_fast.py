"""One equivalent V5 implementation: cached quantile indices and shared partitions."""
from pathlib import Path
import cv2
import joblib
import numpy as np
import pandas as pd
from scipy.ndimage import uniform_filter1d
from .stage3_fast import FrameFeatureComputer as OriginalFrameFeatureComputer
from .stage3 import ACCEL, STEER, VIDEO_EXT
from .stage3_v5 import GlobalResidualComputer as OriginalGlobalResidualComputer


class QuantilePlan:
    """Same NumPy1.26 linear interpolation; indices depend only on ROI size."""
    def __init__(self,n,quantiles):
        self.mid_lo=(n-1)//2
        self.mid_hi=n//2
        virtual=(n-1)*(np.asarray(quantiles,dtype=np.float64)/100.)
        self.lo=np.floor(virtual).astype(np.intp)
        self.hi=np.minimum(self.lo+1,n-1)
        self.gamma=virtual-self.lo
        self.kth=np.unique(np.r_[0,n-1,self.mid_lo,self.mid_hi,self.lo,self.hi])

    def partition(self,values):
        return np.partition(values,self.kth,axis=-1)

    def median(self,partitioned):
        # Preserve float32 means for original channels and float64 for residuals.
        return np.mean(partitioned[...,self.mid_lo:self.mid_hi+1],axis=-1)

    def percentile(self,partitioned):
        a=partitioned[...,self.lo];b=partitioned[...,self.hi]
        diff=b-a
        # NumPy _lerp computes float32 differences before float64 interpolation.
        result=a+diff*self.gamma
        np.subtract(b,diff*(1-self.gamma),out=result,where=self.gamma>=.5,casting='unsafe')
        return result


class FrameFeatureComputer:
    def __init__(self):
        self.shape=None
        self.fallback=OriginalFrameFeatureComputer()

    def __call__(self,flow):
        if not np.isfinite(flow).all():
            return self.fallback(flow)
        h,w=flow.shape[:2]
        if self.shape!=(h,w):
            yy,xx=np.mgrid[:h,:w].astype(np.float32)
            self.xx=(xx-w*.5)/w
            self.yy=(yy-h*.34)/h
            self.rois=[(int(ya*h),int(yb*h),int(xa*w),int(xb*w))
                       for ya,yb in ((.15,.50),(.50,.72),(.72,.93))
                       for xa,xb in ((.03,.27),(.27,.5),(.5,.73),(.73,.97))]
            self.plans=[QuantilePlan((yb-ya)*(xb-xa),(20,80)) for ya,yb,xa,xb in self.rois]
            self.shape=(h,w)
        fx=flow[...,0]/w;fy=flow[...,1]/h
        mag=np.hypot(fx,fy)
        radial=(fx*self.xx+fy*self.yy)/(self.xx*self.xx+self.yy*self.yy+.02)
        channels=np.stack((fx,fy,mag,radial))
        output=np.empty((12,4,3),np.float64)
        for i,((ya,yb,xa,xb),plan) in enumerate(zip(self.rois,self.plans)):
            values=channels[:,ya:yb,xa:xb].reshape(4,-1)
            partitioned=plan.partition(values)
            output[i,:,0]=plan.median(partitioned)
            output[i,:,1:]=plan.percentile(partitioned)
            # Adding median kth can reorder equal signed-zero ties. For a zero
            # quantile, preserve the original partition and interpolation path.
            if np.any(output[i,:,1:]==0):
                output[i,:,1:]=np.percentile(values,(20,80),axis=1).T
        return np.nan_to_num(output.reshape(-1)).astype(np.float32)


class GlobalResidualComputer(OriginalGlobalResidualComputer):
    def _geometry(self,h,w):
        super()._geometry(h,w)
        self.plans=[QuantilePlan((yb-ya)*(xb-xa),(80,)) for ya,yb,xa,xb in self.rois]

    def __call__(self, flow):
        self.calls += 1
        h, w = flow.shape[:2]
        if self.shape != (h, w):
            self._geometry(h, w)
        uv = flow[self.sy, self.sx].astype(np.float64)/w
        keep = np.isfinite(uv).all(axis=1)
        coef = np.zeros(4, np.float64)
        globals_ = np.zeros(8, np.float64)
        try:
            if keep.sum() < 16:
                raise ValueError('insufficient finite grid samples')
            matrix = self.matrix[keep].reshape(-1, 4)
            target = uv[keep].reshape(-1)
            coef, _, rank, _ = np.linalg.lstsq(matrix, target, rcond=None)
            if rank < 4:
                raise ValueError('rank deficient global motion fit')
            for _ in range(3):
                error = np.linalg.norm((target-matrix@coef).reshape(-1, 2), axis=1)
                weight = self._weights(error)
                root = np.repeat(np.sqrt(weight), 2)
                coef, _, rank, _ = np.linalg.lstsq(matrix*root[:, None], target*root, rcond=None)
                if rank < 4 or not np.isfinite(coef).all():
                    raise ValueError('nonfinite or rank deficient robust fit')
            error = np.linalg.norm((target-matrix@coef).reshape(-1, 2), axis=1)
            globals_[:4] = coef
            globals_[4:6] = np.percentile(error, (50, 80), method='linear')
            globals_[6] = self._weights(error).mean()
            globals_[7] = 1.
        except (ValueError, FloatingPointError, np.linalg.LinAlgError):
            self.failures += 1
            coef[:] = 0
            globals_[:] = 0
        tx, ty, a, b = coef
        u = np.nan_to_num(flow[..., 0].astype(np.float64)/w, nan=0., posinf=0., neginf=0.)
        v = np.nan_to_num(flow[..., 1].astype(np.float64)/w, nan=0., posinf=0., neginf=0.)
        ru = u-(tx+a*self.x-b*self.y)
        rv = v-(ty+b*self.x+a*self.y)
        magnitude = np.hypot(ru, rv)
        local = np.empty((12, 3), np.float64)
        channels = np.stack((ru, rv, magnitude))
        for i, ((ya, yb, xa, xb), plan) in enumerate(zip(self.rois, self.plans)):
            values = channels[:, ya:yb, xa:xb].reshape(3, -1)
            partitioned = plan.partition(values)
            local[i, :2] = plan.median(partitioned[:2])
            local[i, 2] = plan.percentile(partitioned[2])[0]
        return np.nan_to_num(np.concatenate((globals_, local.ravel()))).astype(np.float32)


def summarize(raw, centers, count):
    """Same arithmetic/order as the production extractor, separately per block."""
    raw = np.asarray(raw, dtype=np.float32)
    parts = [raw]
    for window in (5, 15, 31):
        parts.append(uniform_filter1d(raw, size=window, axis=0, mode='nearest'))
    smooth = parts[2]
    for lag in (5, 15):
        ix = np.arange(len(smooth))
        parts.append((smooth[np.minimum(ix+lag, len(ix)-1)]-smooth[np.maximum(ix-lag, 0)])/(2*lag/10))
    combined = np.concatenate(parts, axis=1)
    result = np.empty((count, combined.shape[1]), np.float32)
    for j in range(combined.shape[1]):
        result[:, j] = np.interp(np.arange(count), centers, combined[:, j])
    return np.nan_to_num(result)


def extract_motion(path, source_fps=10., width=256, return_diagnostics=False):
    cap = cv2.VideoCapture(str(path))
    dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_FAST)
    base_computer, extra_computer = FrameFeatureComputer(), GlobalResidualComputer()
    stride = max(1, int(round(source_fps/10)))
    prev = None
    base, extra, centers = [], [], []
    count = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            idx = count
            count += 1
            if idx % stride:
                continue
            target_h = max(96, int(round(frame.shape[0]*width/frame.shape[1])))
            gray = cv2.cvtColor(cv2.resize(frame, (width, target_h)), cv2.COLOR_BGR2GRAY)
            if prev is not None:
                flow = dis.calc(prev, gray, None)*(source_fps/stride)
                base.append(base_computer(flow))
                extra.append(extra_computer(flow))
                centers.append(idx-stride*.5)
            prev = gray
    finally:
        cap.release()
    if not count:
        raise ValueError(f'Cannot decode {Path(path).name}')
    result = np.concatenate((summarize(base, centers, count), summarize(extra, centers, count)), axis=1) if base else np.zeros((count, 1128), np.float32)
    detail = {'rows': count, 'flow_pairs': extra_computer.calls, 'fit_failures': extra_computer.failures}
    return (result, detail) if return_diagnostics else result


def predict_stage3(data_dir, model_dir):
    """Load only preexisting candidate weights; never train or update on inputs."""
    model = joblib.load(Path(model_dir)/'motion_model.joblib')
    rows = []
    old_threads = cv2.getNumThreads()
    cv2.setNumThreads(2)
    try:
        for path in sorted((Path(data_dir)/'videos').iterdir()):
            if not path.is_file() or path.suffix.lower() not in VIDEO_EXT:
                continue
            features = extract_motion(path, source_fps=10.)
            accel = model['accel'].predict(features).astype(int)
            steer = model['steer'].predict(features).astype(int)
            rows.extend({'ID': path.stem, 'sample_index': i, 'accel_label': str(ACCEL[a]), 'steer_label': str(STEER[s])}
                        for i, (a, s) in enumerate(zip(accel, steer)))
    finally:
        cv2.setNumThreads(old_threads)
    return pd.DataFrame(rows, columns=['ID', 'sample_index', 'accel_label', 'steer_label'])
