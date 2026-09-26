"""Build V7 from exact V6 ZIP; change Stage1 only and verify extracted bytes."""
import ast,hashlib,json,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'artifacts/submissions';SOURCE=ROOT/'releases/v7/source'
def sha(p):
 with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def main():
 old=json.loads((ROOT/'releases/v6/submission.manifest.json').read_text());archive=OUT/'submit_v6.zip';assert sha(archive)==old['sha256']
 SOURCE.mkdir(exist_ok=False)
 with zipfile.ZipFile(archive) as z:
  assert set(z.namelist())=={r['path'] for r in old['files']}
  for r in old['files']:
   name=r['path'];p=SOURCE/name;assert p.resolve().is_relative_to(SOURCE.resolve());p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(z.read(name));assert sha(p)==r['sha256']
 entry=SOURCE/'inference.py';data=entry.read_text();data=data.replace('DACON236753 V6: uncapped jerk collision, guarded decode, equivalent Stage3 arithmetic.','DACON236753 V7: TPO horizontal-flip average; frozen V6 Stage2 and Stage3.')
 assert data.count('from solution.stage1_v4 import predict_stage1 as predict')==1
 data=data.replace('from solution.stage1_v4 import predict_stage1 as predict','from solution.stage1_v7_tpo_hflip import predict_stage1 as predict');entry.write_text(data)
 config={'mode':'tpo_hflip','sample_frames':12,'original_weight':0.5,'hflip_weight':0.5,'forensic_weight':0.0,'threshold':0.5,'validation_scope':'Fixed public road proxy comparison; target dashcam/device generalization unverified.'}
 (SOURCE/'model/stage1/config.json').write_text(json.dumps(config,indent=2)+'\n')
 module=SOURCE/'model/stage2/code/solution/stage1_v7_tpo_hflip.py'
 module.write_text('''"""Frozen TPO full-frame/horizontal-flip average, no forensic inference."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from .stage1_v4 import sample_video, VIDEO_EXTS
from .stage1_tpo_merged import TPODetector


def predict_stage1(data_dir, model_dir):
    data, model = Path(data_dir), Path(model_dir)
    video_dir = data / 'stage1' / 'videos'
    if not video_dir.is_dir():
        video_dir = data / 'videos' if (data / 'videos').is_dir() else data
    stage_model = model / 'stage1' if (model / 'stage1').is_dir() else model
    config = json.loads((stage_model / 'config.json').read_text(encoding='utf-8'))
    if (config['mode'], config['sample_frames'], config['original_weight'],
        config['hflip_weight'], config['forensic_weight'], config['threshold']) != (
            'tpo_hflip', 12, .5, .5, 0., .5):
        raise ValueError('Unexpected V7 Stage1 configuration')
    detector = TPODetector(stage_model / 'tpo')
    rows = []
    try:
        for path in sorted(p for p in video_dir.iterdir() if p.suffix.lower() in VIDEO_EXTS):
            frames = sample_video(path, config['sample_frames'])
            flipped = [np.ascontiguousarray(frame[:, ::-1, :]) for frame in frames]
            scores = detector.score(frames + flipped)
            probability = .5 * float(scores[:len(frames)].mean()) + .5 * float(scores[len(frames):].mean())
            rows.append({'ID': path.stem, 'answer': 'RERECORDED' if probability >= .5 else 'ORIGINAL'})
    finally:
        detector.close()
    return pd.DataFrame(rows, columns=['ID', 'answer'])
''')
 changed={'inference.py','model/stage1/config.json'};added={'model/stage2/code/solution/stage1_v7_tpo_hflip.py'}
 for r in old['files']:
  if r['path'] not in changed:assert sha(SOURCE/r['path'])==r['sha256'],r['path']
 before=ast.parse((OUT/'verify_v6/inference.py').read_text());after=ast.parse(entry.read_text())
 for name in ['_runtime','predict_stage2','predict_stage3']:
  assert ast.dump(next(n for n in before.body if isinstance(n,ast.FunctionDef) and n.name==name))==ast.dump(next(n for n in after.body if isinstance(n,ast.FunctionDef) and n.name==name))
 dest=OUT/'submit_v7.zip';assert not dest.exists();entries=[]
 with zipfile.ZipFile(dest,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=1) as z:
  for p in sorted(SOURCE.rglob('*')):
   if p.is_file():
    name=p.relative_to(SOURCE).as_posix();z.write(p,name);entries.append({'path':name,'bytes':p.stat().st_size,'sha256':sha(p)})
 assert {r['path'] for r in entries}=={r['path'] for r in old['files']}|added
 verify=OUT/'verify_v7';verify.mkdir(exist_ok=False)
 with zipfile.ZipFile(dest) as z:
  assert z.testzip() is None
  for r in entries:
   p=verify/r['path'];p.parent.mkdir(exist_ok=True,parents=True);p.write_bytes(z.read(r['path']));assert sha(p)==r['sha256']
 size=sum(r['bytes'] for r in entries);assert dest.stat().st_size<10_000_000_000 and size<32_000_000_000
 result={'zip':str(dest),'sha256':sha(dest),'zip_bytes':dest.stat().st_size,'uncompressed_bytes':size,'files':entries,'v6_zip_sha256':old['sha256'],'changed':sorted(changed),'added':sorted(added),'unchanged_original_files':len(old['files'])-len(changed),'stage2_stage3_original_assets_byte_identical':True,'stage2_stage3_entry_ast_identical':True}
 dest.with_suffix('.manifest.json').write_text(json.dumps(result,indent=2));(ROOT/'releases/v7/submission.manifest.json').write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k!='files'},indent=2))
if __name__=='__main__':main()
