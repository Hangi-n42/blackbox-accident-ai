import subprocess,sys,json
from pathlib import Path
p=Path('artifacts/stage1_head_pilot_20260919')
for action in ['extract_train','train','development']:
 with (p/f'{action}.log').open('w') as f:r=subprocess.run([sys.executable,'scripts/data/experiment_stage1_frozen_head.py',action],stdout=f,stderr=subprocess.STDOUT)
 print(action,r.returncode,flush=True)
 if r.returncode:raise SystemExit(r.returncode)
s=json.load(open(p/'development_summary.json'));print('DEVELOPMENT GATE',s['gate_pass'],flush=True)
if s['gate_pass']:
 for action in ['new_val','new_tcl']:
  with (p/f'{action}.log').open('w') as f:r=subprocess.run([sys.executable,'scripts/data/experiment_stage1_frozen_head.py',action],stdout=f,stderr=subprocess.STDOUT)
  print(action,r.returncode,flush=True)
  if r.returncode:raise SystemExit(r.returncode)
