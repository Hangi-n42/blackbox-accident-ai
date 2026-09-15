"""Local-only adapter for the separately licensed, unmodified TPO detector."""
from pathlib import Path
import importlib.util
import sys
import cv2
import numpy as np


class TPODetector:
    def __init__(self, model_dir):
        import torch
        self.torch = torch
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        root = Path(model_dir).resolve()
        required = [root / 'ViT-B-16.pt', root / 'foundpad_tpo.pth',
                    root / 'source/src/models.py', root / 'clip_source/clip/__init__.py']
        for path in required:
            if not path.is_file():
                raise FileNotFoundError(f'TPO local asset missing: {path}')
        sys.path.insert(0, str(root / 'clip_source'))
        sys.path.insert(0, str(root / 'source/src'))
        spec = importlib.util.spec_from_file_location('_competition_tpo_models', root / 'source/src/models.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        # An existing local path bypasses clip.load's network download branch.
        module.BACKBONE = str(root / 'ViT-B-16.pt')
        ckpt = torch.load(root / 'foundpad_tpo.pth', map_location='cpu', weights_only=True)
        cfg = ckpt.get('lora', {})
        self.model = module.build_model(self.device, lora_rank=cfg.get('rank', 8),
                                       lora_alpha=cfg.get('alpha', 8), lora_dropout=cfg.get('dropout', .4),
                                       clip_download_root=str(root), verbose=False)
        missing, unexpected = self.model.load_state_dict(ckpt.get('model', ckpt), strict=False)
        if unexpected or any('lora_' in k or k.startswith('header.') for k in missing):
            raise RuntimeError('TPO checkpoint does not match the expected adapter architecture')
        self.model.eval()
        self.mean = np.asarray([.48145466, .4578275, .40821073], np.float32)
        self.std = np.asarray([.26862954, .26130258, .27577711], np.float32)

    def score(self, frames, batch_size=8):
        scores = []
        for start in range(0, len(frames), batch_size):
            batch = []
            for frame in frames[start:start + batch_size]:
                rgb = cv2.resize(frame, (224, 224), interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255
                batch.append(np.transpose((rgb - self.mean) / self.std, (2, 0, 1)))
            x = self.torch.from_numpy(np.stack(batch)).to(self.device)
            with self.torch.inference_mode():
                scores.extend(self.model(x).softmax(1)[:, 0].cpu().tolist())
        return np.asarray(scores, dtype=float)

    def close(self):
        self.model = None
        import gc
        gc.collect()
        if self.torch.cuda.is_available():
            self.torch.cuda.empty_cache()

