"""Portable readers for historical evidence; never rewrite frozen manifests."""
from pathlib import Path, PureWindowsPath

ROOT = Path(__file__).resolve().parents[2]
LEGACY_ROOT = PureWindowsPath('C:/Users/dsl/Desktop/Dacon/블랙박스')

def resolve_recorded(value, root=ROOT):
    """Map only this project's recorded Windows prefix. Other paths must be supplied."""
    root=Path(root).resolve()
    value=str(value)
    windows=PureWindowsPath(value)
    if windows.drive:
        try:relative=windows.relative_to(LEGACY_ROOT)
        except ValueError:raise ValueError('Recorded path is outside the old project; supply an explicit local file') from None
        result=root.joinpath(*relative.parts)
    else:
        p=Path(value.replace('\\','/'))
        result=p if p.is_absolute() else root/p
    result=result.resolve()
    if not result.is_relative_to(root):raise ValueError('Path escapes project root')
    return result
