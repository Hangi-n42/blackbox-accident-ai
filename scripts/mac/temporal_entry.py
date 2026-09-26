"""One frozen entry-only candidate; no labels, IDs, or cross-video state."""
from solution import stage2_v2 as v2


def refine_entry(paths, prediction, diagnostics, vlm, *, final_contact=False):
    numbers = [v2._frame_number(p) for p in paths]
    # Use the visual contact estimate for subject identification. Motion maxima
    # can be unrelated to contact; neither estimate truncates the search range.
    proposed = v2._integer(diagnostics.get('collision', {}).get('collision_frame'))
    if final_contact:
        proposed = prediction['collision_frame']
    anchor = numbers.index(proposed) if proposed in numbers else numbers.index(prediction['collision_frame'])
    context = sorted({max(0, anchor-2), anchor, min(len(paths)-1, anchor+2)})
    anchor_sheet = v2._sheet(paths, context, columns=3)
    candidates = v2._uniform_indices(0, len(paths)-1, 12)
    selected = numbers.index(prediction['entry_frame'])
    rounds = []
    for level in range(3):
        response = v2._json_object(vlm.ask(
            [anchor_sheet, v2._sheet(paths, candidates, columns=4)],
            'The first image is contact context: identify the OTHER vehicle that collides with the camera car. '
            'Track that SAME vehicle in the second image, whose frames are chronological left to right then top to bottom. '
            'Select the frame closest to its FIRST wheel touching the camera car driving-lane boundary, not when it first appears in view. '
            'Extend the camera lane through intersections. If that same vehicle is already inside the camera lane at the start of the clip, select frame 0 when offered. '
            'Do not select a later crossing by a different vehicle. '
            f'Allowed entry frames: {[numbers[i] for i in candidates]}. Return JSON with entry_frame only.',
            max_new_tokens=40))
        value = v2._integer(response.get('entry_frame'))
        accepted = value in [numbers[i] for i in candidates]
        rounds.append(dict(level=level, candidates=[numbers[i] for i in candidates], response=response, accepted=accepted))
        if not accepted:
            break
        selected = numbers.index(value)
        slot = candidates.index(selected)
        lo = candidates[max(0, slot-1)]
        hi = candidates[min(len(candidates)-1, slot+1)]
        candidates = v2._uniform_indices(lo, hi, 12)
    return {**prediction, 'entry_frame': numbers[selected]}, dict(
        version='temporal_entry_final_contact' if final_contact else 'temporal_entry_v1', visual_contact_anchor=numbers[anchor], rounds=rounds,
        changed_fields=['entry_frame'], calls=len(rounds))
