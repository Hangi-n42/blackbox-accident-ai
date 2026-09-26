"""Local fix for mlx-vlm 0.3.4 Qwen3-VL's two-dimensional mask indexing."""
import numpy as np
import mlx.core as mx


def deepstack_process(self, hidden_states, visual_pos_masks, visual_embeds):
    mask = np.asarray(visual_pos_masks)
    if mask.ndim != 2 or tuple(mask.shape) != tuple(hidden_states.shape[:2]):
        raise ValueError('Expected visual mask [batch, sequence]')
    batch, sequence = np.nonzero(mask)
    if visual_embeds.shape != (len(batch), hidden_states.shape[-1]):
        raise ValueError('Visual embedding count does not match image tokens')
    # The old np.where(mask)[0] returns repeated batch indices, not token indices.
    # Each (batch, sequence) below is unique, avoiding overlapping scatter writes.
    batch, sequence = mx.array(batch), mx.array(sequence)
    updates = hidden_states[batch, sequence, :] + visual_embeds.astype(hidden_states.dtype)
    hidden_states[batch, sequence, :] = updates
    return hidden_states


def install():
    from importlib.metadata import version
    from mlx_vlm.models.qwen3_vl.language import Qwen3VLModel
    from mlx_vlm.models.qwen3_vl.qwen3_vl import Model
    if version('mlx-vlm') != '0.3.4':
        raise RuntimeError('Re-audit local deepstack fix before changing mlx-vlm version')
    Qwen3VLModel._deepstack_process = deepstack_process
    Model.get_input_embeddings = get_input_embeddings


def get_input_embeddings(self, input_ids=None, pixel_values=None, image_grid_thw=None):
    if pixel_values is None:
        return self.language_model.model.embed_tokens(input_ids), None, None
    pixels = pixel_values.astype(self.vision_tower.patch_embed.proj.weight.dtype)
    embeddings = self.language_model.model.embed_tokens(input_ids)
    hidden, deepstack = self.vision_tower(pixels, image_grid_thw)
    # The installed implementation splits by per-image lengths as though they
    # were cumulative indices, then concatenates. Descending lengths duplicate
    # features. Features are already in image order: no split is needed.
    embeddings, mask = self.merge_input_ids_with_image_features(
        hidden, embeddings, input_ids, self.config.image_token_index, self.config.video_token_index)
    return embeddings, mask[..., 0], deepstack
