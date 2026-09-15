"""V6A verifier with individual exact mosaic tiles as its only presentation change.

The shared VLM pixel budget is unchanged. More input images can change effective
resolution and resizing, so this is a bundled presentation experiment.
"""
import hashlib
import json
import logging
from pathlib import Path

from . import stage2_contact_verify_v6a as parent

baseline = parent.baseline
COLUMNS = parent.COLUMNS


class _TileAdapter:
    def __init__(self, vlm, count):
        if not 1 <= count <= 18:
            raise ValueError('Expected one to eighteen offered tiles')
        self.vlm, self.count = vlm, count
        self.presentation = None

    def ask(self, images, prompt, max_new_tokens):
        if len(images) != 1 or self.presentation is not None:
            raise ValueError('Expected exactly one parent mosaic call')
        mosaic = images[0]
        # Parent _sheet clamps requested columns=5 when fewer than five tiles.
        columns = min(5, self.count)
        expected = (columns * 384, ((self.count + columns - 1) // columns) * 256)
        if mosaic.size != expected:
            raise ValueError('Unexpected parent mosaic dimensions')
        tiles = [mosaic.crop(((i % columns) * 384, (i // columns) * 256,
                              (i % columns + 1) * 384, (i // columns + 1) * 256))
                 for i in range(self.count)]
        self.presentation = dict(
            version='contact_tiles_v6b', tile_count=self.count, tile_size=[384, 256],
            columns=columns, mosaic_size=list(mosaic.size),
            tile_rgb_sha256=[hashlib.sha256(tile.convert('RGB').tobytes()).hexdigest() for tile in tiles],
            order='Original mosaic row-major offered-frame order; blank padded cells omitted',
            pixel_budget_changed_by_adapter=False,
            caveat='Unchanged shared pixel cap does not imply equal effective per-frame resolution.',
        )
        return self.vlm.ask(tiles, prompt, max_new_tokens=max_new_tokens)


def refine_collision(paths, prediction, diagnostics, vlm):
    numbers = [baseline.base._frame_number(path) for path in paths]
    internal = diagnostics['collision_replacement']['base_collision_frame']
    motion = prediction['collision_frame']
    if not numbers or numbers != sorted(set(numbers)):
        raise ValueError('Expected nonempty, unique, increasing original frame numbers')
    if type(internal) is not int or type(motion) is not int or internal not in numbers or motion not in numbers:
        raise ValueError('Both parent centers must be in valid paths')
    windows = [parent._window_indices(numbers.index(center), len(paths)) for center in (internal, motion)]
    count = len(set(windows[0]) | set(windows[1]))
    adapter = _TileAdapter(vlm, count)
    result, detail = parent.refine_collision(paths, prediction, diagnostics, adapter)
    if adapter.presentation is None:
        raise RuntimeError('Parent did not call verifier')
    if len(detail['contact_verification']['offered_original_frames']) != count:
        raise RuntimeError('Parent offered count differs from presentation')
    return result, {**detail, 'presentation': adapter.presentation}


def _predict_file(paths, scores, vlm):
    prediction, diagnostics = baseline._predict_file(paths, scores, vlm)
    return refine_collision(paths, prediction, diagnostics, vlm)


def predict_stage2(data_dir, model_dir):
    image_root = Path(data_dir) / 'images'
    if not image_root.is_dir():
        raise FileNotFoundError(f'Missing Stage2 image directory: {image_root}')
    rows = []
    with baseline.CandidateVLM(Path(model_dir) / 'vlm', precision='nf4') as vlm:
        for folder in sorted(path for path in image_root.iterdir() if path.is_dir()):
            paths = sorted((path for path in folder.iterdir()
                            if path.suffix.lower() in baseline.base.IMAGE_EXTENSIONS), key=baseline.base._frame_number)
            numbers = [baseline.base._frame_number(path) for path in paths]
            if len(numbers) != len(set(numbers)):
                raise ValueError(f'Duplicate Stage2 frame number in {folder.name}')
            paths, scores, _ = baseline.base._motion_scan(paths)
            prediction, diagnostics = _predict_file(paths, scores, vlm)
            logging.getLogger(__name__).info('Stage2 contact tiles %s: %s', folder.name,
                                             json.dumps(diagnostics, ensure_ascii=False))
            rows.append(dict(ID=folder.name, **prediction))
    return baseline.pd.DataFrame(rows, columns=COLUMNS)
