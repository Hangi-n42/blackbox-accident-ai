"""Output contracts shared by the future real-ZIP check; no model or GT loading."""
import re
from pathlib import Path

IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}


def frame_numbers(folder):
    paths = [p for p in Path(folder).iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS]
    numbers = []
    for path in paths:
        match = re.search(r'(\d+)$', path.stem)
        if match is None:
            raise ValueError('Image lacks an original frame number: '+path.name)
        numbers.append(int(match.group(1)))
    if not numbers or len(numbers) != len(set(numbers)):
        raise ValueError('Empty or duplicate original frame inventory')
    return set(numbers)


def validate_stage2(result, image_root):
    import pandas as pd
    columns = ['ID', 'collision_frame', 'entry_frame', 'evasion_space', 'entry_side']
    if not isinstance(result, pd.DataFrame) or list(result.columns) != columns or result.isna().any().any():
        raise ValueError('Invalid Stage2 output schema')
    if not result.ID.is_unique or set(result.ID) != {p.name for p in Path(image_root).iterdir() if p.is_dir()}:
        raise ValueError('Stage2 case IDs differ from input')
    for column in ('collision_frame', 'entry_frame', 'evasion_space'):
        if not pd.api.types.is_integer_dtype(result[column]) or pd.api.types.is_bool_dtype(result[column]):
            raise ValueError('Stage2 frame/space outputs must be integers')
    if not set(result.entry_side) <= {'LEFT', 'RIGHT'} or not set(result.evasion_space) <= {0, 1}:
        raise ValueError('Invalid Stage2 category')
    for row in result.itertuples():
        numbers = frame_numbers(Path(image_root)/row.ID)
        if row.collision_frame not in numbers or row.entry_frame not in numbers:
            raise ValueError('Predicted original frame is absent from input')


def compare_expected_stage2(result, report):
    import pandas as pd
    if report.get('status') != 'complete' or report.get('phase') != 'round2_fresh_validation':
        raise ValueError('Actual completed fresh validation report required')
    if report.get('call_count') != 12 or report.get('model_loads') != 1 or report.get('network_attempts') != 0:
        raise ValueError('Fresh validation execution contract failed')
    if [v['ID'] for v in report.get('videos', [])] != ['00008','00010','00013']:
        raise ValueError('Unexpected fresh validation IDs')
    expected = pd.DataFrame([dict(ID=v['ID'], **v['candidate']) for v in report['videos']])[result.columns]
    pd.testing.assert_frame_equal(result.reset_index(drop=True), expected.reset_index(drop=True))
