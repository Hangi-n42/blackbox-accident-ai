---
license: apache-2.0
base_model: Qwen/Qwen3-VL-4B-Instruct
---

# Local NF4 derivative of Qwen3-VL-4B-Instruct

Original publisher: Qwen. Source: https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct
Original revision: ebb281ec70b05090aa6165b016eac8ec08e71b17

License: Apache-2.0; see LICENSE-APACHE-2.0.txt and README.original.md.

Modification: pretrained weights were converted to bitsandbytes NF4 with FP16 compute, without double quantization, then saved with Transformers save_pretrained. No fine-tuning or public/evaluation-label training was performed. Processor assets were reserialized with save_pretrained. This derivative is locally generated and is not an official Qwen-distributed checkpoint.

Reload equivalence and offline inference are NOT yet verified by export. Those checks are required before adoption.
