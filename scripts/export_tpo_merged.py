"""Merge fixed TPO query/value adapter tensors into an OpenAI CLIP visual encoder.

This transformation implements matrix addition directly; it imports no TPO,
FoundPAD, CLIP-LoRA or loralib source. No optimization or data fitting occurs.
"""
from pathlib import Path
import argparse
import hashlib
import json
import math
import torch


ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-dir', type=Path, default=ROOT / 'model/stage1/tpo')
    args = parser.parse_args()
    root = args.model_dir.resolve()
    backbone, adapter = root / 'ViT-B-16.pt', root / 'foundpad_tpo.pth'
    expected = '5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f'
    if sha256(backbone) != expected:
        raise ValueError('Official CLIP ViT-B/16 backbone hash mismatch')
    torch.set_num_threads(2)
    with backbone.open('rb') as stream:
        clip = torch.jit.load(stream, map_location='cpu')
    visual = {key.removeprefix('visual.'): value.detach().float().clone()
              for key, value in clip.state_dict().items() if key.startswith('visual.')}
    del clip
    checkpoint = torch.load(adapter, map_location='cpu', weights_only=True)
    state = checkpoint['model']
    rank = int(checkpoint['lora']['rank'])
    alpha = float(checkpoint['lora']['alpha'])
    scale = alpha / math.sqrt(rank)
    width = int(visual['conv1.weight'].shape[0])
    layers = len({key.split('.')[2] for key in visual if key.startswith('transformer.resblocks.')})
    consumed, operations = set(), []
    for layer in range(layers):
        projection_name = f'transformer.resblocks.{layer}.attn.in_proj_weight'
        combined = visual[projection_name]
        if tuple(combined.shape) != (3 * width, width):
            raise ValueError(f'Unexpected QKV matrix shape: {projection_name}')
        for projection, block in [('q', 0), ('v', 2)]:
            prefix = f'visual.transformer.resblocks.{layer}.attn.{projection}_proj.w_lora_'
            key_a, key_b = prefix + 'A', prefix + 'B'
            a, b = state[key_a].float(), state[key_b].float()
            if tuple(a.shape) != (rank, width) or tuple(b.shape) != (width, rank):
                raise ValueError(f'Unexpected adapter shape for {prefix}')
            delta = (b @ a) * scale
            combined[block * width:(block + 1) * width].add_(delta)
            consumed.update((key_a, key_b))
            operations.append({'layer': layer, 'projection': projection, 'scale': scale,
                               'offset': block * width, 'delta_l2': float(torch.linalg.vector_norm(delta))})
    head = {name: state['header.' + name].detach().float().clone() for name in ['weight', 'bias']}
    consumed.update(('header.weight', 'header.bias'))
    if set(state) != consumed:
        raise ValueError(f'Unconsumed adapter tensors: {sorted(set(state) - consumed)}')
    patch_size = int(visual['conv1.weight'].shape[-1])
    grid_size = math.isqrt(int(visual['positional_embedding'].shape[0]) - 1)
    configuration = {'input_resolution': patch_size * grid_size, 'patch_size': patch_size,
                     'width': width, 'layers': layers, 'heads': width // 64,
                     'output_dim': int(visual['proj'].shape[1])}
    result = {'format_version': 1, 'visual': visual, 'head': head, 'architecture': configuration,
              'preprocessing': {'size': 224, 'color': 'RGB', 'interpolation': 'opencv_INTER_LINEAR',
                                'mean': [.48145466, .4578275, .40821073],
                                'std': [.26862954, .26130258, .27577711]},
              'class_order': ['RERECORDED', 'ORIGINAL'],
              'source_backbone_sha256': sha256(backbone), 'source_adapter_sha256': sha256(adapter)}
    target = root / 'merged_visual.pt'
    torch.save(result, target)
    manifest = {'transformation': 'Frozen rsLoRA merge: W_q/v += (alpha/sqrt(rank)) * B @ A',
                'training_performed': False, 'source_backbone': str(backbone.name),
                'source_backbone_sha256': result['source_backbone_sha256'],
                'source_adapter': str(adapter.name), 'source_adapter_sha256': result['source_adapter_sha256'],
                'output': target.name, 'output_sha256': sha256(target), 'output_bytes': target.stat().st_size,
                'architecture': configuration, 'merged_dtype': 'float32', 'operations': operations,
                'implementation': 'Participant-authored matrix addition, no TPO/LoRA source imported',
                'runtime_source': 'Unmodified OpenAI CLIP model.py (MIT), custom inference wrapper',
                'weights_license': 'TPO adapter-derived merged weights: CC BY-NC-SA 4.0; underlying CLIP rights preserved'}
    (root / 'merged_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(json.dumps({key: value for key, value in manifest.items() if key != 'operations'}, indent=2))


if __name__ == '__main__':
    main()
