"""Package the fixed Stage1 candidate with preserved V7 Stage2/3; never overwrite."""
import hashlib
import json
import shutil
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'artifacts/submissions/v8_20260920'
EXP = ROOT / 'artifacts/stage1_strong_anchor_20260919'
OLD = ROOT / 'artifacts/stage1_anchored_head_20260919'

def sha(p):
    with p.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()

def write(p, data):
    p.write_text(json.dumps(data, indent=2, ensure_ascii=False) + '\n')

def main():
    baseline = ROOT / 'artifacts/submissions/submit_v7.zip'
    manifest = json.loads((ROOT / 'releases/v7/submission.manifest.json').read_text())
    assert sha(baseline) == manifest['sha256']
    runtime = json.loads((EXP / 'runtime_report.json').read_text())
    assert runtime['status'] == 'PASS'
    weights = EXP / 'candidate_runtime/model/stage1/tpo/merged_visual.pt'
    assert sha(weights) == json.loads((EXP / 'export.json').read_text())['weights_sha256']
    OUT.mkdir(exist_ok=False)
    source = OUT / 'source'
    source.mkdir()
    with zipfile.ZipFile(baseline) as z:
        assert set(z.namelist()) == {r['path'] for r in manifest['files']}
        for r in manifest['files']:
            target = source / r['path']
            assert target.resolve().is_relative_to(source.resolve())
            target.parent.mkdir(parents=True, exist_ok=True)
            with z.open(r['path']) as inp, target.open('wb') as out:
                shutil.copyfileobj(inp, out)
            assert sha(target) == r['sha256']
    print('Original V7 extracted and verified', flush=True)
    tpo = source / 'model/stage1/tpo'
    shutil.copyfile(weights, tpo / 'merged_visual.pt')
    # Preserve the old conversion record, then describe the actual current artifact.
    shutil.copyfile(tpo / 'merged_manifest.json', tpo / 'original_merged_manifest.json')
    data = json.loads((tpo / 'merged_manifest.json').read_text())
    data.update(training_performed=True, output_sha256=sha(weights), output_bytes=weights.stat().st_size,
                transformation='Original frozen TPO/CLIP visual merge, followed by V7-anchored binary classifier fitting; visual tensors unchanged.',
                original_conversion_manifest='original_merged_manifest.json',
                classifier_training_manifest='anchored_head_provenance.json')
    write(tpo / 'merged_manifest.json', data)
    note = tpo / 'MERGED_WEIGHTS_NOTICE.md'
    note.write_text(note.read_text() + '\n## Subsequent classifier modification\n\nOn 2026-09-19 the final 513 binary classifier coefficients were fitted on external VDmoire/REDS, DLC-2021 and comma2k19 data with an L2 anchor to the previous V7 head (lambda=0.01). The visual tensors remain unchanged. The preceding conversion-only description applies to the original checkpoint, not this later classifier. This checkpoint is a derivative distributed under the retained TPO CC BY-NC-SA 4.0 conditions, with underlying CLIP rights preserved. See anchored_head_provenance.json and DATA_SOURCES.md. No evaluation-time training or network access is used.\n')
    provenance = {
        'candidate': 'anchored_0.01', 'lambda': 0.01, 'parameters': 513,
        'head_sha256': sha(EXP / 'anchored_0.01.npz'), 'weights_sha256': sha(weights),
        'training': json.loads((OLD / 'training.json').read_text())['counts'],
        'inference': '12 full frames +12 horizontal flips, mean probability, threshold0.5; forensic disabled',
        'validation': 'Known regression guards and offline entry tests passed. Reserved DLC confirmation skipped by explicit user request. Official improvement unverified.',
        'stage2_stage3': 'Unchanged V7 assets; later experiments did not establish an adopted replacement.',
        'original_v7_zip_sha256': manifest['sha256'],
    }
    write(tpo / 'anchored_head_provenance.json', provenance)
    shutil.copyfile(OLD / 'DATA_SOURCES.md', tpo / 'DATA_SOURCES.md')
    for rec in (7467028, 6466770):
        shutil.copyfile(OLD / f'license_{rec}.txt', tpo / f'LICENSE.DLC-{rec}.txt')
    shutil.copyfile(ROOT / 'research/stage1/comma_original_diagnostic/comma2k19_LICENSE', tpo / 'LICENSE.comma2k19.txt')
    changed = {'model/stage1/tpo/' + n for n in ['merged_visual.pt', 'merged_manifest.json', 'MERGED_WEIGHTS_NOTICE.md']}
    for r in manifest['files']:
        if r['path'] not in changed:
            assert sha(source / r['path']) == r['sha256'], r['path']
    files = [p for p in sorted(source.rglob('*')) if p.is_file()]
    assert not any(p.is_symlink() or '__pycache__' in p.parts for p in files)
    archive = OUT / 'submit_v8.zip'
    entries = []
    with zipfile.ZipFile(archive, 'x', compression=zipfile.ZIP_DEFLATED, compresslevel=1) as z:
        for p in files:
            name = p.relative_to(source).as_posix()
            z.write(p, name)
            entries.append({'path': name, 'bytes': p.stat().st_size, 'sha256': sha(p)})
    print('ZIP built; extracting for verification', flush=True)
    verified = OUT / 'extracted'
    verified.mkdir()
    with zipfile.ZipFile(archive) as z:
        assert z.namelist() == [r['path'] for r in entries]
        for r in entries:
            p = verified / r['path']
            p.parent.mkdir(parents=True, exist_ok=True)
            with z.open(r['path']) as inp, p.open('wb') as out:
                shutil.copyfileobj(inp, out)
            assert sha(p) == r['sha256'], r['path']
    total = sum(r['bytes'] for r in entries)
    assert archive.stat().st_size < 10_000_000_000 and total < 32_000_000_000
    assert sha(baseline) == manifest['sha256']
    result = {'zip': str(archive), 'sha256': sha(archive), 'zip_bytes': archive.stat().st_size,
              'uncompressed_bytes': total, 'files': entries, 'changed': sorted(changed),
              'added': sorted({r['path'] for r in entries} - {r['path'] for r in manifest['files']}),
              'original_v7_preserved': True, 'stage2_stage3_byte_identical': True,
              'entrypoint_requirements_config_unchanged': True, 'extracted_sha_and_crc_verified': True,
              'selection': provenance, 'new_submission': False}
    write(OUT / 'submit_v8.manifest.json', result)
    print(json.dumps({k:v for k,v in result.items() if k not in ('files','selection')}, indent=2), flush=True)

if __name__ == '__main__':
    main()
