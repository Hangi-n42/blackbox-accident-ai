"""Run all real entry points from an extracted ZIP in an isolated interpreter."""
import os
os.environ['HF_HUB_OFFLINE']='1'
os.environ['TRANSFORMERS_OFFLINE']='1'
os.environ['HF_HUB_DISABLE_IMPLICIT_TOKEN']='1'
import argparse,hashlib,importlib.util,json,socket,sys,time
from pathlib import Path
import numpy as np
import pandas as pd
import av
import psutil

def deny_network(*args,**kwargs):
    raise AssertionError('Unexpected network connection during offline inference')

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--package',type=Path,required=True)
    parser.add_argument('--data',type=Path,required=True)
    parser.add_argument('--stage3-data',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--manifest',type=Path)
    parser.add_argument('--require-fresh-output',action='store_true')
    args=parser.parse_args()
    package=args.package.resolve();sys.dont_write_bytecode=True
    socket.socket.connect=deny_network
    socket.socket.connect_ex=deny_network
    socket.create_connection=deny_network
    spec=importlib.util.spec_from_file_location('submission_inference',package/'inference.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    args.output.mkdir(parents=True,exist_ok=not args.require_fresh_output)
    report={'scope':'Real packaged offline execution and output contract, not generalization score',
            'package':str(package),'stages':{},
            'inference_sha256':hashlib.sha256((package/'inference.py').read_bytes()).hexdigest(),
            'package_manifest_sha256':None if args.manifest is None else hashlib.sha256(args.manifest.read_bytes()).hexdigest()}
    for stage in (1,2,3):
        data=(args.stage3_data if stage==3 else args.data/f'stage{stage}').resolve()
        start=time.perf_counter()
        result=getattr(module,f'predict_stage{stage}')(data,package/'model'/f'stage{stage}')
        assert isinstance(result,pd.DataFrame)
        expected={1:['ID','answer'],2:['ID','collision_frame','entry_frame','evasion_space','entry_side'],
                  3:['ID','sample_index','accel_label','steer_label']}[stage]
        assert list(result.columns)==expected
        assert not result.isna().any().any()
        if stage==1:
            assert set(result.ID)=={p.stem for p in (data/'videos').glob('*.mp4')}
            assert result.ID.is_unique
            assert set(result.answer)<= {'ORIGINAL','RERECORDED'}
        elif stage==2:
            assert set(result.ID)=={p.name for p in (data/'images').iterdir() if p.is_dir()}
            assert result.ID.is_unique
            assert set(result.entry_side)<={'LEFT','RIGHT'}
            assert set(result.evasion_space)<={0,1}
            assert pd.api.types.is_integer_dtype(result.evasion_space)
            for row in result.itertuples():
                numbers={int(p.stem.rsplit('_',1)[1]) for p in (data/'images'/row.ID).glob('*.jpg')}
                assert row.collision_frame in numbers and row.entry_frame in numbers
        else:
            assert set(result.accel_label)<={'ACCELERATING','DECELERATING','CONSTANT','STOPPED'}
            assert set(result.steer_label)<={'LEFT','STRAIGHT','RIGHT'}
            assert not result.duplicated(['ID','sample_index']).any()
            assert set(result.ID)=={p.stem for p in (data/'videos').glob('*.mp4')}
            for path in (data/'videos').glob('*.mp4'):
                with av.open(str(path)) as container:count=sum(1 for _ in container.decode(video=0))
                indices=result.loc[result.ID==path.stem,'sample_index'].to_numpy()
                assert np.array_equal(indices,np.arange(count))
        for name,loaded in list(sys.modules.items()):
            if name.startswith('solution') and getattr(loaded,'__file__',None):
                assert Path(loaded.__file__).resolve().is_relative_to(package)
        stage_report={'rows':len(result),'seconds':time.perf_counter()-start,
                      'resident_gib':psutil.Process().memory_info().rss/2**30,'contract':'PASS'}
        report['stages'][str(stage)]=stage_report
        result.to_csv(args.output/f'stage{stage}.csv',index=False,encoding='utf-8')
        (args.output/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
        print(json.dumps({'stage':stage,**stage_report}),flush=True)
    report['status']='PASS'
    (args.output/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')

if __name__=='__main__':main()
