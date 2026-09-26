"""Per-video screen recapture features. No filename/metadata class predictors."""
from pathlib import Path
import json
import re
import cv2
import numpy as np
import pandas as pd
from PIL import Image

VIDEO_EXTS = {'.mp4', '.avi', '.mov', '.mkv', '.webm', '.m4v', '.3gp', '.3gpp', '.wmv'}


def sample_video(path, count=12):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f'Cannot open video: {path}')
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []
    if total > 0:
        for index in np.unique(np.linspace(0, total - 1, min(count, total)).round().astype(int)):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(index))
            ok, frame = cap.read()
            if ok:
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    if not frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        while len(frames) < count:
            ok, frame = cap.read()
            if not ok:
                break
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    if not frames:
        raise RuntimeError(f'No decodable frames: {path}')
    return frames


def _patch_features(patch):
    p = patch.astype(np.float32) / 255.0
    gray = cv2.cvtColor(p, cv2.COLOR_RGB2GRAY)
    residual = gray - cv2.GaussianBlur(gray, (0, 0), 1.0)
    lap = cv2.Laplacian(gray, cv2.CV_32F)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1)
    grad = np.sqrt(gx * gx + gy * gy)
    quality = [np.std(gray), np.mean(np.abs(residual)), np.std(lap),
               np.mean(grad), np.quantile(grad, .9), np.mean(gray < .025),
               np.mean(gray > .975), np.mean(np.max(p, 2) - np.min(p, 2))]
    n, m = gray.shape
    win = np.hanning(n)[:, None] * np.hanning(m)[None, :]
    spectrum = np.abs(np.fft.fftshift(np.fft.fft2(residual * win))) ** 2
    yy, xx = np.meshgrid(np.fft.fftshift(np.fft.fftfreq(n)),
                         np.fft.fftshift(np.fft.fftfreq(m)), indexing='ij')
    radius = np.sqrt(xx * xx + yy * yy)
    energy = np.sum(spectrum) + 1e-12
    frequency = [float(np.sum(spectrum[(radius >= a) & (radius < b)]) / energy)
                 for a, b in [(0, .1), (.1, .2), (.2, .3), (.3, .4), (.4, .71)]]
    valid = spectrum[(radius > .07) & (radius < .48)]
    frequency += [float(np.quantile(valid, .999) / (np.mean(valid) + 1e-12)),
                  float(np.sum(spectrum[(np.abs(xx) < .015) & (radius > .07)]) / energy),
                  float(np.sum(spectrum[(np.abs(yy) < .015) & (radius > .07)]) / energy)]
    color_res = p - cv2.GaussianBlur(p, (0, 0), 1.0)
    chroma = color_res[:, :, 0] - color_res[:, :, 2]
    frequency += [float(np.std(chroma) / (np.std(residual) + 1e-5)),
                  float(np.std(residual.mean(1)) / (np.std(residual) + 1e-5)),
                  float(np.std(residual.mean(0)) / (np.std(residual) + 1e-5))]
    return np.asarray(quality + frequency, np.float64)


def extract_features(frames):
    patch_rows, border_rows = [], []
    for frame in frames:
        h, w = frame.shape[:2]
        side = min(192, h, w)
        for fy, fx in [(0.2, .2), (.2, .8), (.5, .5), (.8, .2), (.8, .8)]:
            y = int(np.clip(fy * h - side / 2, 0, h - side))
            x = int(np.clip(fx * w - side / 2, 0, w - side))
            patch_rows.append(_patch_features(frame[y:y + side, x:x + side]))
        small = cv2.resize(frame, (320, 180), interpolation=cv2.INTER_AREA).astype(np.float32) / 255
        gray = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY)
        zones = [gray[:12], gray[-12:], gray[:, :16], gray[:, -16:]]
        border_rows.append([v for z in zones for v in (np.mean(z), np.std(z), np.mean(z < .025))])
    patches = np.asarray(patch_rows)
    borders = np.asarray(border_rows)
    # Log compression prevents one narrow FFT spike dominating the linear model.
    patches[:, 13] = np.log1p(patches[:, 13])
    features = np.concatenate([np.median(patches, 0), np.quantile(patches, .9, axis=0),
                               np.median(borders, 0)])
    names = ([f'patch_median_{i}' for i in range(patches.shape[1])] +
             [f'patch_q90_{i}' for i in range(patches.shape[1])] +
             [f'border_{i}' for i in range(borders.shape[1])])
    return features, names


def feature_probability(features, artifact):
    indices = np.asarray(artifact['indices'], dtype=int)
    x = (features[indices] - np.asarray(artifact['mean'])) / np.asarray(artifact['scale'])
    z = float(x @ np.asarray(artifact['coef']) + artifact['intercept'])
    return float(1 / (1 + np.exp(-np.clip(z, -40, 40))))


VISUAL_PROMPT = '''These are temporally ordered frames from ONE dashcam video. Determine whether the footage was filmed directly by the dashcam (ORIGINAL), or replayed on a physical screen and filmed again by another camera (RERECORDED). Ordinary re-encoding, black letterboxing, rain, windshield reflections, low resolution, subtitles and poor quality alone do NOT establish screen recapture. Look for a physical display boundary and surrounding room/device, screen-surface reflections, spatial moire/pixel grid, scan bands or independent camera-to-screen perspective/motion. Return JSON only: {"answer":"ORIGINAL or RERECORDED","confidence":"high or medium or low","evidence":"brief directly visible evidence"}. If no specific screen-recapture evidence is visible, choose ORIGINAL with low confidence. Do not infer from filename.'''


def visual_judgment(frames, vlm):
    chosen = [Image.fromarray(frames[i]) for i in np.unique(np.linspace(0, len(frames) - 1, min(4, len(frames))).round().astype(int))]
    reply = vlm.ask(chosen, VISUAL_PROMPT, max_new_tokens=180)
    text = reply if isinstance(reply, str) else json.dumps(reply)
    match = re.search(r'\{.*\}', text, flags=re.S)
    try:
        result = json.loads(match.group() if match else text)
    except (ValueError, TypeError):
        return {'answer': None, 'confidence': 'low', 'evidence': text}
    return result


def predict_stage1(data_dir, model_dir):
    data, model = Path(data_dir), Path(model_dir)
    video_dir = data / 'stage1' / 'videos'
    if not video_dir.is_dir():
        video_dir = data / 'videos' if (data / 'videos').is_dir() else data
    stage_model = model / 'stage1' if (model / 'stage1').is_dir() else model
    config_path = stage_model / 'config.json'
    config = json.loads(config_path.read_text(encoding='utf-8')) if config_path.is_file() else {}
    artifact_path = stage_model / config.get('forensic_artifact', 'forensic.json')
    if not artifact_path.is_file():
        artifact_path = model / 'forensic.json'
    artifact = json.loads(artifact_path.read_text(encoding='utf-8'))
    detector = None
    if config.get('mode') == 'tpo_forensic_ensemble':
        from .stage1_tpo_merged import TPODetector
        detector = TPODetector(stage_model / 'tpo')
    results = []
    try:
        for path in sorted(p for p in video_dir.iterdir() if p.suffix.lower() in VIDEO_EXTS):
            frames = sample_video(path, artifact.get('sample_frames', 12))
            features, _ = extract_features(frames)
            probability = feature_probability(features, artifact)
            if detector is not None:
                weight = float(config.get('forensic_weight', .5))
                probability = weight * probability + (1 - weight) * float(detector.score(frames).mean())
            answer = 'RERECORDED' if probability >= config.get('threshold', artifact['threshold']) else 'ORIGINAL'
            results.append({'ID': path.stem, 'answer': answer})
    finally:
        if detector is not None:
            detector.close()
    return pd.DataFrame(results, columns=['ID', 'answer'])
