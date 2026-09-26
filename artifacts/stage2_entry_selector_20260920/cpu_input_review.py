"""CPU-only processor and tile-order proof; never loads a model or calls an encoder."""
import hashlib
import inspect
import json
import os
from pathlib import Path
import sys
sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
read = lambda p: json.loads(p.read_text())


def sha(p):
    with p.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def audit():
    import numpy as np
    from PIL import Image
    import socket
    attempts = []
    def deny(*args, **kwargs):
        attempts.append(True)
        raise RuntimeError('CPU input audit attempted network')
    socket.socket.connect = socket.socket.connect_ex = socket.create_connection = deny
    os.environ.update(HF_HUB_OFFLINE='1', HF_HUB_DISABLE_IMPLICIT_TOKEN='1', TOKENIZERS_PARALLELISM='false')
    import mlx.core as mx
    mx.set_default_device(mx.cpu)
    import torch
    torch.set_num_threads(2)
    from mlx_vlm.utils import load_processor, load_image_processor, prepare_inputs
    model = ROOT / 'artifacts/mac_experiments/stage2_mlx/model'
    cfg = read(model / 'config.json')
    processor = load_processor(model, True, eos_token_ids=cfg.get('eos_token_id'), trust_remote_code=False, local_files_only=True)
    optional = load_image_processor(model, trust_remote_code=False, local_files_only=True)
    if optional is not None:
        processor.image_processor = optional
    def process(image, prompt):
        text = processor.apply_chat_template([{'role':'user','content':[{'type':'image'},{'type':'text','text':prompt}]}], tokenize=False, add_generation_prompt=True)
        inputs = prepare_inputs(processor, images=[image], prompts=text, image_token_index=cfg.get('image_token_index') or cfg['image_token_id'], add_special_tokens=True)
        mx.eval(*[v for v in inputs.values() if isinstance(v, mx.array)])
        arrays = {k: np.asarray(v) for k, v in inputs.items() if isinstance(v, mx.array)}
        return text, arrays
    jobs = read(HERE / 'inputs.json')['jobs']
    assert len(jobs) == 16 and len({j['ID'] for j in jobs}) == 16
    records = []
    for job in jobs:
        image_path = ROOT / job['image']; history = read(ROOT / job['history_call'])[2]
        assert sha(image_path) == job['image_sha256']
        with Image.open(image_path) as im:
            image = im.convert('RGB')
        assert image.size == (1536, 768)
        assert [hashlib.sha256(image.tobytes()).hexdigest()] == history['image_sha256']
        text, arrays = process(image, history['prompt'])
        hashes = {k: hashlib.sha256(v.tobytes()).hexdigest() for k, v in arrays.items()}
        assert hashes == history['processor_input_sha256']
        assert hashlib.sha256(text.encode()).hexdigest() == history['prompt_sha256']
        assert arrays['image_grid_thw'].tolist() == [[1,48,96]]
        visual = int(np.sum(arrays['input_ids'] == cfg['image_token_id']))
        assert visual == 1152 and arrays['input_ids'].shape[-1] == history['prompt_tokens']
        assert job['frames'] == sorted(set(job['frames'])) and len(job['frames']) == 12
        records.append(dict(ID=job['ID'], role=job['role'], image_sha256=job['image_sha256'], processor_hashes=hashes, image_grid_thw=arrays['image_grid_thw'].tolist(), image_tokens=visual, input_tokens=history['prompt_tokens']))
    # Unique colors for every 32x32 merged cell. Decoding processor patches proves order
    # without using or simulating learned model weights.
    cells = np.empty((24,48,3), dtype=np.uint8)
    for y in range(24):
        for x in range(48):
            cells[y,x] = [8 + y*9, 4 + x*5, 17 + ((y//8)*4+x//12)*19]
    pixels = np.repeat(np.repeat(cells,32,axis=0),32,axis=1)
    _, arrays = process(Image.fromarray(pixels), 'Synthetic processor ordering test.')
    ip = processor.image_processor
    assert ip.patch_size == 16 and ip.temporal_patch_size == 2 and ip.merge_size == 2
    patches = arrays['pixel_values'].reshape(-1,3,2,16,16)
    decoded = patches.mean(axis=(2,3,4)) * np.asarray(ip.image_std) + np.asarray(ip.image_mean)
    decoded = decoded / float(ip.rescale_factor)
    assert decoded.shape == (4608,3)
    expected_patches = np.repeat(cells.reshape(1152,3),4,axis=0)
    np.testing.assert_allclose(decoded, expected_patches, rtol=0, atol=1e-3)
    merged = decoded.reshape(1152,4,3).mean(axis=1)
    np.testing.assert_allclose(merged.reshape(24,48,3), cells, rtol=0, atol=1e-3)
    import selector
    pooled = selector.pool_tiles(merged)
    expected = np.stack([pixels[(i//4)*256+32:(i//4+1)*256,(i%4)*384:(i%4+1)*384].mean((0,1)) for i in range(12)])
    np.testing.assert_allclose(pooled, expected, rtol=0, atol=1e-3)
    processor_path = Path(inspect.getfile(type(ip)))
    from mlx_vlm.models.qwen3_vl.vision import VisionModel, PatchMerger
    vision_path = Path(inspect.getfile(VisionModel))
    # Source text is inspected, not executed: Qwen3's blocks preserve patch sequence,
    # and final PatchMerger groups four contiguous patches. No window-reorder stage.
    vision_source, merger_source = inspect.getsource(VisionModel.__call__), inspect.getsource(PatchMerger.__call__)
    assert 'hidden_states = self.merger(hidden_states)' in vision_source and '.reshape(-1, self.hidden_size)' in merger_source
    assert 'window_index' not in vision_source
    assert not attempts and str(mx.default_device()) == 'Device(cpu, 0)'
    return dict(status='PASS', model_loads=0, encoder_forwards=0, text_generations=0, real_training_runs=0, device=str(mx.default_device()), network_attempts=0,
                inputs=records, source_hashes={str(processor_path):sha(processor_path),str(vision_path):sha(vision_path),str(HERE/'selector.py'):sha(HERE/'selector.py')},
                synthetic_test=dict(patch_shape=list(patches.shape),merged_grid=[24,48],tiles=12,pooled_cells_per_tile=84,dropped_top_pixel_rows_per_tile=32,maximum_patch_color_error=float(np.max(np.abs(decoded-expected_patches))),maximum_pool_color_error=float(np.max(np.abs(pooled-expected))),ordering='four contiguous16x16 patches per32x32 merged cell;24x48 cells row-major;12tiles row-major'),
                limits=['No model/encoder run. Color reconstruction proves processor ordering and spatial pooling indices, not semantic localization.','Final vision blocks attend across the full sheet; excluding header token rows from pooling does not erase header information or make tile features independent.','32-row exclusion also removes4scene pixels below the28-pixel header; fixed before fit.','Averaged tile embedding represents a scene area, not a verified collision counterpart.'])


if __name__ == '__main__':
    result = audit()
    out = HERE / 'cpu_input_review.json'
    assert not out.exists()
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('inputs','source_hashes')},ensure_ascii=False))
