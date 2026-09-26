"""Merged TPO weights with only the original MIT OpenAI CLIP visual implementation.

No TPO/FoundPAD/loralib/CLIP-LoRA source is imported by this runtime.
"""
from pathlib import Path
import gc
import importlib.util
import cv2
import numpy as np


class TPODetector:
    def __init__(self, model_dir, device=None):
        import torch
        self.torch = torch
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        root = Path(model_dir).resolve()
        source = root / 'clip_source/clip/model.py'
        weights = root / 'merged_visual.pt'
        if not source.is_file() or not weights.is_file():
            raise FileNotFoundError(f'Merged CLIP runtime assets missing under {root}')
        spec = importlib.util.spec_from_file_location('_competition_openai_clip_visual', source)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        artifact = torch.load(weights, map_location='cpu', weights_only=True)
        if artifact['format_version'] != 1 or artifact['class_order'] != ['RERECORDED', 'ORIGINAL']:
            raise ValueError('Unsupported merged TPO artifact format or class order')
        self.visual = module.VisionTransformer(**artifact['architecture']).float()
        self.visual.load_state_dict(artifact['visual'], strict=True)
        self.head = torch.nn.Linear(artifact['architecture']['output_dim'], 2)
        self.head.load_state_dict(artifact['head'], strict=True)
        self.visual.to(self.device).eval()
        self.head.to(self.device).eval()
        self.mean = np.asarray(artifact['preprocessing']['mean'], np.float32)
        self.std = np.asarray(artifact['preprocessing']['std'], np.float32)
        self.size = int(artifact['preprocessing']['size'])
        del artifact

    def score(self, frames, batch_size=8):
        scores = []
        for start in range(0, len(frames), batch_size):
            batch = []
            for frame in frames[start:start + batch_size]:
                image = cv2.resize(frame, (self.size, self.size), interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255
                batch.append(np.transpose((image - self.mean) / self.std, (2, 0, 1)))
            x = self.torch.from_numpy(np.stack(batch)).to(self.device)
            with self.torch.inference_mode():
                features = self.torch.nn.functional.normalize(self.visual(x).float(), dim=-1)
                scores.extend(self.head(features).softmax(1)[:, 0].cpu().tolist())
        return np.asarray(scores, dtype=float)

    def close(self):
        self.visual = None
        self.head = None
        gc.collect()
        if self.torch.cuda.is_available():
            self.torch.cuda.empty_cache()
