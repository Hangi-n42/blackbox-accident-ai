"""Research-only: change V6 call-four images to the final motion contact context.

Import with the verified V6 solution package on sys.path. No package entry point,
model loading, label access, prompt rewriting, or changes to the other fields.
"""
from __future__ import annotations

import numpy as np

from solution import stage2_uncapped_jerk_v6c as frozen

base = frozen.baseline.base


def _context(paths, new_scores):
    values = np.asarray(new_scores)
    if not paths or values.shape != (len(paths),) or not np.isfinite(values).all():
        raise ValueError('One finite new score per valid path is required')
    numbers = [base._frame_number(path) for path in paths]
    if numbers != sorted(set(numbers)):
        raise ValueError('Paths must have unique ascending original frame numbers')
    motion = int(np.argmax(values))
    indices = sorted(set([max(0, motion - 2), motion, min(len(paths) - 1, motion + 2)]))
    return indices, dict(version='v7_space_final_motion_anchor',
                         final_collision_frame=numbers[motion],
                         context_indices=indices,
                         context_frames=[numbers[index] for index in indices],
                         context_radius_input_positions=2,
                         renderer='frozen stage2_v2._sheet columns=3',
                         prompt_policy='unchanged original V6 fourth prompt',
                         max_new_tokens=40)


def _space_value(raw):
    parsed = base._json_object(raw)
    value = base._integer(parsed.get('evasion_space'))
    fallback = value not in {0, 1}
    return (0 if fallback else value), parsed, fallback


def predict_space_only(paths, new_scores, vlm, prompt, max_new_tokens=40):
    """One cached-context diagnostic call; caller supplies the actual V6 prompt."""
    if type(max_new_tokens) is not int or max_new_tokens != 40:
        raise ValueError('V6 space token cap must remain exactly 40')
    if not isinstance(prompt, str) or not prompt:
        raise ValueError('Cached original V6 fourth prompt must be supplied')
    indices, details = _context(paths, new_scores)
    raw = vlm.ask([base._sheet(paths, indices, columns=3)], prompt, max_new_tokens=40)
    value, parsed, fallback = _space_value(raw)
    return value, {**details, 'calls': 1, 'raw': raw, 'parsed': parsed,
                   'fallback': fallback, 'evasion_space': value}


class _FourthCallProxy:
    """Per-file proxy. The first three calls are forwarded without modification."""
    def __init__(self, paths, new_scores, vlm):
        self.paths = paths
        self.indices, self.details = _context(paths, new_scores)
        self.vlm = vlm
        self.calls = 0

    def ask(self, images, prompt, max_new_tokens=128):
        if self.calls >= 4:
            raise RuntimeError('V6 four-call budget exceeded')
        self.calls += 1
        if self.calls == 4:
            if type(max_new_tokens) is not int or max_new_tokens != 40:
                raise ValueError('V6 fourth-call token cap changed')
            images = [base._sheet(self.paths, self.indices, columns=3)]
        raw = self.vlm.ask(images, prompt, max_new_tokens=max_new_tokens)
        if self.calls == 4:
            value, parsed, fallback = _space_value(raw)
            self.details.update(raw=raw, parsed=parsed, fallback=fallback,
                                evasion_space=value)
        return raw


def _predict_file(paths, base_scores, new_scores, vlm):
    """Exact frozen V6 four-call execution, except call-four image context."""
    proxy = _FourthCallProxy(paths, new_scores, vlm)
    prediction, diagnostics = frozen._predict_file(paths, base_scores, new_scores, proxy)
    if proxy.calls != 4:
        raise RuntimeError('Expected exactly four frozen V6 calls')
    if prediction['collision_frame'] != proxy.details['final_collision_frame']:
        raise RuntimeError('Final collision differs from the supplied motion anchor')
    if prediction['evasion_space'] != proxy.details['evasion_space']:
        raise RuntimeError('Space parser differs from frozen V6')
    diagnostics = {**diagnostics, 'uncapped_jerk': {
        **diagnostics['uncapped_jerk'],
        'baseline_context_for_other_fields_retained': False,
        'baseline_context_for_entry_side_retained': True,
        'baseline_context_for_evasion_space_retained': False,
    }}
    return prediction, {**diagnostics, 'space_anchor': {**proxy.details, 'calls': 4,
                         'only_fourth_call_images_changed': True}}
