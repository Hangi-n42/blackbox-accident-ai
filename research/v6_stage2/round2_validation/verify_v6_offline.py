"""Actual extracted V6 entry points, only after a real fresh contact gate passes.

Does not package, upload, manufacture reviews, or claim hidden-set accuracy.
The future packager must bind validation_evaluation_sha256 in its manifest.
"""
import os, sys, argparse, hashlib, importlib.util, json, socket, time, zipfile
from pathlib import Path
sys.dont_write_bytecode = True
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key] = '2'
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
MEMBER = 'model/stage2/code/solution/stage2_uncapped_jerk_v6c.py'
C_SHA = '3e86e5117230fb4680c1d4af630cb02dd7e9aca99b7c4184d7baed13f479d64e'

def require(ok, text):
    if not ok:
        raise ValueError(text)

def sha(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle,'sha256').hexdigest()

def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))

def load(name, path):
    spec = importlib.util.spec_from_file_location(name,path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('package','manifest','validation-run','stage1-data','stage3-data','baseline-results','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    args = parser.parse_args()
    require(sys.flags.isolated,'Use Python -I')
    package, output, validation = args.package.resolve(),args.output.resolve(),args.validation_run.resolve()
    require(output.is_relative_to(ROOT.resolve()) and not output.exists(),'New workspace output required')
    evaluation_path = validation/'evaluation.json'
    evaluation = read(evaluation_path)  # Missing real evaluation fails before any writes/model import.
    require(evaluation['phase']=='round2_fresh_validation' and evaluation['gate_passed'] is True
            and all(evaluation['conditions'].values()),'Fresh contact accuracy gate did not pass')
    for name in ('report','freeze','review_binding'):
        require(evaluation[name+'_sha256']==sha(validation/(name+'.json')),'Fresh result binding changed')
    require(evaluation['candidate_sha256']==C_SHA and sha(evaluation['evaluator_path'])==evaluation['evaluator_sha256'],
            'Candidate/evaluator changed')
    for path,digest in evaluation['all_bound_file_sha256s'].items():
        require(sha(path)==digest,'Fresh evidence changed')
    fresh_runner = load('_offline_input_binding',HERE/'run_validation.py')
    frozen = fresh_runner.verify(validation)
    fresh_runner.review_guard(validation,frozen)
    expected_report = read(validation/'report.json')
    manifest = read(args.manifest)
    require(manifest['validation_evaluation_sha256']==sha(evaluation_path),'Manifest is not bound to this fresh evaluation')
    require(sha(manifest['zip'])==manifest['sha256'],'Actual ZIP hash differs')
    entries = manifest['files']
    expected = {item['path']:item for item in entries}
    require(len(expected)==len(entries),'Duplicate manifest member')
    old = read(ROOT/'artifacts/submissions/v5_selection_frozen.json')['expected_archive_sha256']
    require(set(expected)==set(old)|{MEMBER} and expected[MEMBER]['sha256']==C_SHA,'Unapproved package inventory')
    require(all(expected[name]['sha256']==digest for name,digest in old.items() if name!='inference.py'),
            'Stage1/3/model/requirements or original Stage2 assets changed')
    scope_helper = load('_offline_scope_only',HERE.parent/'package_uncapped_jerk_v6c.py')
    expected_entry = scope_helper.modified_inference((ROOT/'artifacts/submissions/verify_v5/inference.py').read_bytes())
    require((package/'inference.py').read_bytes()==expected_entry,'Unexpected entry-point change')
    require(Path(manifest['zip']).stat().st_size<10_000_000_000,'ZIP exceeds compressed size limit')
    with zipfile.ZipFile(manifest['zip']) as archive:
        infos=archive.infolist()
        require(len(infos)==len(expected) and {i.filename for i in infos}==set(expected),'ZIP inventory differs')
        require(sum(i.file_size for i in infos)<32_000_000_000,'ZIP exceeds uncompressed size limit')
        for item in infos:
            require(not item.is_dir() and item.file_size==expected[item.filename]['bytes'],'ZIP member size differs')
            with archive.open(item) as handle:
                require(hashlib.file_digest(handle,'sha256').hexdigest()==expected[item.filename]['sha256'],
                        'Actual ZIP member bytes differ from manifest')
    def verify_package():
        require({p.relative_to(package).as_posix() for p in package.rglob('*') if p.is_file()}==set(expected),
                'Extracted package inventory changed')
        for name,item in expected.items():
            path = (package/name).resolve()
            require(path.is_relative_to(package) and path.stat().st_size==item['bytes'] and sha(path)==item['sha256'],
                    'Extracted package bytes differ')
    verify_package()
    references = {s:args.baseline_results/f'stage{s}.csv' for s in (1,3)}
    known_reference_sha = {1:'1465670b5634c8a17ffd51cc2260fec2d29fe75bc96c3c4283d4148c8cf32bfb',
                           3:'af6187ed01ede5044df771713707a01a3ec0ea4921a703a3557852294066d01e'}
    for s,path in references.items():
        require(sha(path)==known_reference_sha[s],'Frozen V5 output reference differs')
    video_bindings = {str(p.resolve()):sha(p) for directory in (args.stage1_data,args.stage3_data)
                      for p in (directory/'videos').glob('*.mp4')}
    require(video_bindings,'No Stage1/3 input videos')
    output.mkdir()
    image_root = output/'inputs/stage2/images'
    for record in frozen['videos']:
        folder = image_root/record['ID']
        folder.mkdir(parents=True)
        for item in record['input']['input_images']:
            source = Path(record['input_root'])/item['path']
            os.link(source,folder/source.name)  # No pixel conversion or renumbering; read-only inference below.
    report = dict(status='running',scope='Actual packaged offline execution and finite-input output equivalence only',
                  package=str(package),manifest_sha256=sha(args.manifest),zip_sha256=manifest['sha256'],
                  validation_evaluation_sha256=sha(evaluation_path),network_attempts=0,stages={},
                  source_video_sha256s=video_bindings,stage2_input_method='Hard links to frozen full-native PNG files')
    verification_sources = {str(p):sha(p) for p in (Path(__file__).resolve(),HERE/'offline_contracts.py',
        HERE/'run_validation.py',HERE.parent/'package_uncapped_jerk_v6c.py')}
    report['verification_source_sha256s']=verification_sources
    def deny(*unused,**kwargs):
        report['network_attempts']+=1
        raise RuntimeError('Offline inference attempted a network connection')
    socket.socket.connect=deny;socket.socket.connect_ex=deny;socket.create_connection=deny
    try:
        import av
        import numpy as np
        import pandas as pd
        import psutil
        contracts = load('_offline_output_contracts',HERE/'offline_contracts.py')
        require(not any(k=='solution' or k.startswith('solution.') for k in sys.modules),'Foreign solution imported')
        inference = load('_real_v6_inference',package/'inference.py')
        directories = {1:args.stage1_data.resolve(),2:image_root.parent,3:args.stage3_data.resolve()}
        for stage,data in directories.items():
            started=time.perf_counter()
            result=getattr(inference,f'predict_stage{stage}')(data,package/'model'/f'stage{stage}')
            require(isinstance(result,pd.DataFrame) and not result.isna().any().any(),'Invalid output frame')
            if stage==2:
                contracts.validate_stage2(result,image_root)
                contracts.compare_expected_stage2(result,expected_report)
            else:
                wanted=pd.read_csv(references[stage],dtype={'ID':str})
                pd.testing.assert_frame_equal(result.reset_index(drop=True),wanted.reset_index(drop=True))
                require(set(result.ID)=={p.stem for p in (data/'videos').glob('*.mp4')},'Input/output IDs differ')
                if stage==1:
                    require(result.ID.is_unique and set(result.answer)<={'ORIGINAL','RERECORDED'},'Stage1 contract')
                else:
                    require(not result.duplicated(['ID','sample_index']).any(),'Duplicate Stage3 row')
                    for video in (data/'videos').glob('*.mp4'):
                        with av.open(str(video)) as container:
                            count=sum(1 for _ in container.decode(video=0))
                        require(np.array_equal(result.loc[result.ID==video.stem,'sample_index'],np.arange(count)),
                                'Stage3 did not return one row per input frame')
            for name,module in list(sys.modules.items()):
                if (name=='solution' or name.startswith('solution.')) and getattr(module,'__file__',None):
                    require(Path(module.__file__).resolve().is_relative_to(package),'Non-package solution imported')
            result.to_csv(output/f'stage{stage}.csv',index=False,encoding='utf8')
            report['stages'][str(stage)]=dict(rows=len(result),seconds=time.perf_counter()-started,
                 rss_bytes=psutil.Process().memory_info().rss,output_equivalence=True)
            print(json.dumps(dict(stage=stage,status='offline_output_match')),flush=True)
        require(report['network_attempts']==0,'Unexpected network attempt')
        verify_package()
        fresh_runner.verify(validation)
        for path,digest in video_bindings.items():require(sha(path)==digest,'Input video changed')
        for path,digest in verification_sources.items():require(sha(path)==digest,'Offline checker changed during execution')
        for stage,path in references.items():require(sha(path)==known_reference_sha[stage],'V5 CSV reference changed')
        require(sha(manifest['zip'])==report['zip_sha256'],'ZIP changed during inference')
        require(sha(args.manifest)==report['manifest_sha256'] and sha(evaluation_path)==report['validation_evaluation_sha256'],
                'Manifest/evaluation changed during inference')
        report['status']='PASS'
        report['limitations']='This finite local run is not a full hidden-test runtime or accuracy guarantee.'
    except BaseException as error:
        report.update(status='FAILED',error_type=type(error).__name__)
        raise
    finally:
        with (output/'report.json').open('x',encoding='utf8') as handle:
            json.dump(report,handle,ensure_ascii=False,indent=2)

if __name__=='__main__':
    main()
