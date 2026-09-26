"""Experimental Apple GPU Stage2. MLX weights are NOT the frozen CUDA NF4 model."""
import argparse
import dataclasses
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('HF_HUB_DISABLE_IMPLICIT_TOKEN', '1')
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')

class MLXVLM:
    def __init__(self, model_path, trace_dir, pixel_budget=1_200_000):
        import mlx.core as mx
        from mlx_vlm import load
        from mlx_vlm.utils import load_config
        if not mx.metal.is_available():
            raise RuntimeError('An accessible Apple Metal GPU is required')
        mx.set_default_device(mx.gpu)
        mx.random.seed(0)
        self.mx = mx
        self.trace_dir = trace_dir
        self.pixel_budget = pixel_budget
        self.calls = []
        self.config = load_config(str(model_path))
        started = time.perf_counter()
        self.model, self.processor = load(str(model_path), trust_remote_code=False)
        self.deepstack_fix = os.environ.get('BLACKBOX_DEEPSTACK_FIX', '0') == '1'
        if self.deepstack_fix:
            from deepstack_fix import install
            install()
        self.compute_dtype = os.environ.get('BLACKBOX_COMPUTE_DTYPE', 'native')
        if self.compute_dtype == 'float32':
            self.model.set_dtype(mx.float32)
            mx.eval(self.model.parameters())
        self.load_seconds = time.perf_counter() - started

    def ask(self, images, prompt, max_new_tokens=128):
        from PIL import Image
        from mlx_vlm import generate
        from mlx_vlm.utils import prepare_inputs
        import numpy as np
        if not images:
            raise ValueError('At least one image is required')
        budget = max(1024, self.pixel_budget // len(images))
        bounded = []
        for image in images:
            image = image.convert('RGB')
            scale = min(1.0, math.sqrt(budget / (image.width * image.height)))
            size = (max(32, int(image.width*scale)//32*32), max(32, int(image.height*scale)//32*32))
            bounded.append(image.resize(size, Image.Resampling.BICUBIC))
        # Preserve the original user prompt and contact-sheet sizing. Only backend changes.
        content = [{'type': 'image'} for _ in bounded] + [{'type': 'text', 'text': prompt}]
        text = self.processor.apply_chat_template([{'role': 'user', 'content': content}], tokenize=False, add_generation_prompt=True)
        # Materialize the exact processor output before asynchronous GPU generation.
        inputs = prepare_inputs(self.processor, images=bounded, prompts=text,
                                image_token_index=getattr(self.model.config, 'image_token_index', None),
                                add_special_tokens=True)
        self.mx.eval(*[v for v in inputs.values() if isinstance(v, self.mx.array)])
        input_hashes = {k: hashlib.sha256(np.asarray(v).tobytes()).hexdigest()
                        for k, v in inputs.items() if isinstance(v, self.mx.array)}
        kwargs = {k: v for k, v in inputs.items()
                  if k not in ('input_ids', 'pixel_values', 'attention_mask')}
        self.mx.synchronize()
        started = time.perf_counter()
        mode = os.environ.get('BLACKBOX_DECODE_MODE', 'legacy')
        token_trace = []
        if mode == 'sync':
            from sync_decode import generate_sync
            result, token_trace = generate_sync(self.model, self.processor, inputs, int(max_new_tokens))
        elif mode == 'legacy':
            result = generate(self.model, self.processor, text,
                              input_ids=inputs['input_ids'], pixel_values=inputs.get('pixel_values'),
                              mask=inputs.get('attention_mask'), **kwargs,
                              max_tokens=int(max_new_tokens), temperature=0.0, verbose=False)
        else:
            raise ValueError(f'Unknown decode mode: {mode}')
        self.mx.synchronize()
        record = dataclasses.asdict(result)
        record.pop('logprobs', None)
        record.update(deepstack_fix=self.deepstack_fix, compute_dtype=self.compute_dtype, decode_mode=mode, token_trace=token_trace, prompt=prompt, max_new_tokens=max_new_tokens, processor_input_sha256=input_hashes,
                      image_sizes=[list(x.size) for x in bounded],
                      image_sha256=[hashlib.sha256(x.tobytes()).hexdigest() for x in bounded],
                      prompt_sha256=hashlib.sha256(text.encode()).hexdigest(),
                      seconds=time.perf_counter()-started)
        self.calls.append(record)
        (self.trace_dir/'calls.json').write_text(json.dumps(self.calls, ensure_ascii=False, indent=2))
        print(json.dumps({'call': len(self.calls), 'answer': result.text, 'seconds': record['seconds'], 'peak_memory_GB': result.peak_memory}), flush=True)
        return result.text.strip()

