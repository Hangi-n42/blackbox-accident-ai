"""Diagnostic synchronous greedy decoder for the local Qwen3-VL model."""
import hashlib
import numpy as np
import mlx.core as mx
from mlx_vlm.models import cache
from mlx_vlm.generate import GenerationResult


def generate_sync(model, processor, inputs, max_tokens):
    tokenizer = processor.tokenizer
    tokenizer.stopping_criteria.reset(model.config.eos_token_id)
    detokenizer = processor.detokenizer
    detokenizer.reset()
    kv = cache.make_prompt_cache(model.language_model)
    extra = {k: v for k, v in inputs.items()
             if k not in ('input_ids', 'pixel_values', 'attention_mask')}
    traces, text = [], ''
    with mx.stream(mx.default_stream(mx.gpu)):
        output = model(inputs['input_ids'], inputs.get('pixel_values'),
                       cache=kv, mask=inputs.get('attention_mask'), **extra)
        if getattr(output, 'cross_attention_states', None) is not None or getattr(output, 'encoder_outputs', None) is not None:
            raise ValueError('Decoder supports Qwen3-VL only')
        for index in range(max_tokens):
            logits = output.logits[:, -1, :]
            mx.eval(logits)
            values = np.asarray(logits.astype(mx.float32))
            if not np.isfinite(values).all():
                raise ValueError('Nonfinite logits')
            token = int(mx.argmax(logits, axis=-1).item())
            top = np.argsort(values[0])[-2:][::-1]
            traces.append(dict(token=token, logits_sha256=hashlib.sha256(values.tobytes()).hexdigest(),
                               top2=top.tolist(), margin=float(values[0, top[0]]-values[0, top[1]])))
            if tokenizer.stopping_criteria(token):
                break
            detokenizer.add_token(token, skip_special_token_ids=[])
            text += detokenizer.last_segment
            if index + 1 < max_tokens:
                output = model.language_model(mx.array([[token]], dtype=mx.int32), cache=kv)
        detokenizer.finalize()
        text += detokenizer.last_segment
        mx.synchronize()
    result = GenerationResult(text=text, prompt_tokens=inputs['input_ids'].size,
                              generation_tokens=len(traces), peak_memory=mx.get_peak_memory()/1e9)
    del kv, output
    mx.clear_cache()
    return result, traces
