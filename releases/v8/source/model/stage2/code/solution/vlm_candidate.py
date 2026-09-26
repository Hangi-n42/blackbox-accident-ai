"""Experimental 4B loading only; inherits the frozen submission's exact ask()."""
from __future__ import annotations

import importlib.metadata
import json
from pathlib import Path
import time

from .vlm import LocalVLM


class CandidateVLM(LocalVLM):
    def __init__(self, model_path, *, pixel_budget=1_200_000,
                 precision="nf4", gpu_memory="5GiB", cpu_memory="12GiB"):
        import torch
        from transformers import AutoProcessor, BitsAndBytesConfig, Qwen3VLForConditionalGeneration

        if precision not in {"nf4", "fp16_offload"}:
            raise ValueError(f"Unsupported candidate precision: {precision}")
        if not torch.cuda.is_available():
            raise RuntimeError("Candidate evaluation requires an available CUDA GPU")
        self.torch = torch
        self.device = "cuda"
        self.pixel_budget = int(pixel_budget)
        self.model_path = Path(model_path)
        if not (self.model_path / "config.json").is_file():
            raise FileNotFoundError(f"Local candidate assets missing: {self.model_path}")
        self.processor = AutoProcessor.from_pretrained(
            str(self.model_path), local_files_only=True, trust_remote_code=False)
        kwargs = dict(local_files_only=True, trust_remote_code=False,
                      dtype=torch.float16, attn_implementation="sdpa", low_cpu_mem_usage=True)
        saved_config = json.loads((self.model_path/"config.json").read_text(encoding="utf-8"))
        stored_quantization = saved_config.get("quantization_config")
        if stored_quantization and precision != "nf4":
            raise ValueError("Use nf4 mode to reload an already quantized checkpoint")
        if precision == "nf4":
            kwargs.update(device_map={"": 0})
            if stored_quantization:
                if (stored_quantization.get("quant_method") != "bitsandbytes"
                        or stored_quantization.get("bnb_4bit_quant_type") != "nf4"
                        or stored_quantization.get("bnb_4bit_compute_dtype") != "float16"
                        or not stored_quantization.get("load_in_4bit")
                        or stored_quantization.get("bnb_4bit_use_double_quant")):
                    raise ValueError("Stored quantization differs from the evaluated NF4/FP16 configuration")
                # Transformers reconstructs the saved packed weights and quantization state.
                # Supplying a fresh config here could override serialized settings.
            else:
                kwargs.update(quantization_config=BitsAndBytesConfig(
                    load_in_4bit=True, bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=False))
        else:
            kwargs.update(device_map="auto", max_memory={0: gpu_memory, "cpu": cpu_memory})
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(str(self.model_path), **kwargs).eval()
        # Accelerate owns device placement. A subsequent .to('cuda') would defeat offload.
        torch.cuda.synchronize()
        self.candidate_metadata = {
            "precision": precision, "compute_dtype": "float16", "attention": "sdpa",
            "prequantized_checkpoint": bool(stored_quantization),
            "quantization_config_source": "saved_config" if stored_quantization else "runtime_configuration",
            "double_quant": False if precision == "nf4" else None,
            "model_load_seconds": time.perf_counter() - started,
            "load_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "load_peak_reserved_bytes": torch.cuda.max_memory_reserved(),
            "loaded_allocated_bytes": torch.cuda.memory_allocated(),
            "model_memory_footprint_bytes": self.model.get_memory_footprint(),
            "device_map": {k: str(v) for k, v in getattr(self.model, "hf_device_map", {}).items()},
            "max_memory": {"cuda:0": gpu_memory, "cpu": cpu_memory} if precision == "fp16_offload" else None,
            "versions": {name: importlib.metadata.version(name)
                         for name in ("torch", "transformers", "accelerate", "bitsandbytes")},
            "ask_implementation": "inherited unchanged from solution.vlm.LocalVLM.ask",
        }
