"""Frozen, local-only vision-language inference with bounded visual tokens."""
from __future__ import annotations
import gc
import math
from pathlib import Path
from PIL import Image


class LocalVLM:
    def __init__(self, model_path, *, pixel_budget=1_200_000):
        import torch
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
        self.torch = torch
        self.pixel_budget = int(pixel_budget)
        self.model_path = Path(model_path)
        if not (self.model_path / 'config.json').is_file():
            raise FileNotFoundError(f'Local VLM assets missing: {self.model_path}')
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.processor = AutoProcessor.from_pretrained(
            str(self.model_path), local_files_only=True, trust_remote_code=False)
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            str(self.model_path), local_files_only=True, trust_remote_code=False,
            dtype=torch.float16 if self.device == 'cuda' else torch.float32,
            attn_implementation='sdpa').to(self.device).eval()

    def ask(self, images: list[Image.Image], prompt: str, max_new_tokens=128) -> str:
        if not images:
            raise ValueError('At least one image is required')
        budget = max(32*32, self.pixel_budget // len(images))
        bounded=[]
        for image in images:
            image=image.convert('RGB')
            scale=min(1.0, math.sqrt(budget / (image.width * image.height)))
            w=max(32, int(image.width*scale)//32*32)
            h=max(32, int(image.height*scale)//32*32)
            bounded.append(image.resize((w,h),Image.Resampling.BICUBIC))
        content=[{'type':'image'} for _ in bounded]
        content.append({'type':'text','text':prompt})
        text=self.processor.apply_chat_template(
            [{'role':'user','content':content}],tokenize=False,add_generation_prompt=True)
        inputs=self.processor(text=[text],images=bounded,return_tensors='pt',
                              images_kwargs={'min_pixels':32*32,'max_pixels':budget})
        inputs=inputs.to(self.device)
        with self.torch.inference_mode():
            generated=self.model.generate(**inputs,do_sample=False,max_new_tokens=int(max_new_tokens),
                                          use_cache=True)
        answer=self.processor.batch_decode(
            generated[:,inputs['input_ids'].shape[1]:],skip_special_tokens=True,
            clean_up_tokenization_spaces=False)[0]
        del generated,inputs
        return answer.strip()

    def close(self):
        self.model=None
        self.processor=None
        gc.collect()
        if self.torch.cuda.is_available():
            self.torch.cuda.empty_cache()

    def __enter__(self):return self
    def __exit__(self,*args):self.close()
