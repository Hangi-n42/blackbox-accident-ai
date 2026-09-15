"""Portable V6 runner. CPU Stage1/3; preserve CUDA requirement for Stage2."""
import argparse
import importlib.util
import os
from pathlib import Path
from paths import ROOT

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--stage',type=int,choices=[1,2,3],required=True)
    ap.add_argument('--data',type=Path,required=True,help='Prepared stage input directory, not a label CSV')
    ap.add_argument('--release',type=Path,default=Path(os.getenv('BLACKBOX_V6_DIR','artifacts/submissions/verify_v6')))
    ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args()
    release=(ROOT/args.release).resolve() if not args.release.is_absolute() else args.release.resolve()
    if args.output.exists():raise FileExistsError('Choose a new output file')
    import torch
    if args.stage==2 and not torch.cuda.is_available():
        raise RuntimeError('Exact V6 Stage2 uses CUDA NF4. MPS/CPU replacement is not validated; use a CUDA host.')
    spec=importlib.util.spec_from_file_location('frozen_v6_inference',release/'inference.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    result=getattr(module,f'predict_stage{args.stage}')(args.data,release/'model'/f'stage{args.stage}')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    result.to_csv(args.output,index=False)
    print(f'Wrote {len(result)} rows to {args.output}')

if __name__=='__main__':main()
