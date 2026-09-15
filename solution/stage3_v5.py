"""Research-only V5: preserve 864 motion features, append 264 global/residual features."""
from pathlib import Path
import cv2
import joblib
import numpy as np
import pandas as pd
from scipy.ndimage import uniform_filter1d
from .stage3_fast import FrameFeatureComputer
from .stage3 import ACCEL, STEER, VIDEO_EXT


class GlobalResidualComputer:
    """Deterministic fixed-iteration image-motion descriptor, local to one video."""
    def __init__(self):
        self.shape = None
        self.calls = 0
        self.failures = 0

    def _geometry(self, h, w):
        yy, xx = np.mgrid[:h, :w].astype(np.float64)
        self.x = (xx-w*.5)/w
        self.y = (yy-h*.5)/w
        xs = np.rint(np.linspace(.03*(w-1), .97*(w-1), 16)).astype(int)
        ys = np.rint(np.linspace(.15*(h-1), .72*(h-1), 12)).astype(int)
        sy, sx = np.meshgrid(ys, xs, indexing='ij')
        self.sy, self.sx = sy.ravel(), sx.ravel()
        x, y = self.x[self.sy, self.sx], self.y[self.sy, self.sx]
        self.matrix = np.zeros((len(x), 2, 4), np.float64)
        self.matrix[:, 0] = np.stack((np.ones_like(x), np.zeros_like(x), x, -y), axis=1)
        self.matrix[:, 1] = np.stack((np.zeros_like(x), np.ones_like(x), y, x), axis=1)
        self.rois = [(int(ya*h), int(yb*h), int(xa*w), int(xb*w))
                     for ya, yb in ((.15, .50), (.50, .72), (.72, .93))
                     for xa, xb in ((.03, .27), (.27, .5), (.5, .73), (.73, .97))]
        self.shape = (h, w)

    @staticmethod
    def _weights(error):
        scale = max(float(np.median(error)), 1e-6)
        return np.minimum(1., 1.345*scale/np.maximum(error, 1e-12))

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
        for i, (ya, yb, xa, xb) in enumerate(self.rois):
            local[i, 0] = np.median(ru[ya:yb, xa:xb])
            local[i, 1] = np.median(rv[ya:yb, xa:xb])
            local[i, 2] = np.percentile(magnitude[ya:yb, xa:xb], 80, method='linear')
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
