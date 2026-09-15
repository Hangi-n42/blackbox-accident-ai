"""Build an allowlisted code submission and its content hash manifest."""
from pathlib import Path
import argparse,ast,hashlib,json,zipfile

ROOT=Path(__file__).resolve().parents[1]

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--variant',choices=('v1','nf4','nf4_fast','nf4_scale_fast','nf4_motion_fast','v4','v5'),default='v1')
    args=parser.parse_args()
    sources={}
    use_nf4=args.variant in ('nf4','nf4_fast','nf4_scale_fast','nf4_motion_fast','v4','v5')
    sources['inference.py']=ROOT/({'v1':'inference.py','nf4':'inference_nf4.py',
                                  'nf4_fast':'inference_nf4_fast.py',
                                  'nf4_scale_fast':'inference_nf4_fast.py',
                                  'nf4_motion_fast':'inference_motion_fast.py','v4':'inference_v4.py','v5':'inference_v5.py'}[args.variant])
    sources['requirements.txt']=ROOT/('requirements_nf4.txt' if use_nf4 else 'requirements.txt')
    selected_stage1 = {
        'stage1/config.json', 'stage1/forensic_frequency_augmented.json',
        'stage1/tpo/merged_visual.pt', 'stage1/tpo/MERGED_WEIGHTS_NOTICE.md',
        'stage1/tpo/merged_manifest.json', 'stage1/tpo/source/LICENSE',
        'stage1/tpo/clip_source/LICENSE', 'stage1/tpo/clip_source/clip/model.py',
    }
    for base in (ROOT/'model',ROOT/'artifacts/model'):
        if not base.exists():continue
        for path in base.rglob('*'):
            if not path.is_file() or any(part.startswith('.') or part=='__pycache__' for part in path.relative_to(base).parts):continue
            relative=path.relative_to(base).as_posix()
            if relative.startswith('stage1/') and relative not in selected_stage1:continue
            if use_nf4 and relative.startswith('stage2/'):continue
            name='model/'+relative
            if name in sources:raise ValueError(f'Duplicate model path {name}')
            sources[name]=path
    if use_nf4:
        candidate=ROOT/'artifacts/candidates/qwen3_vl_4b_nf4'
        assert not (candidate/'EXPORT_INCOMPLETE.json').exists()
        assert (candidate/'EXPORT_MANIFEST.json').is_file()
        for path in candidate.rglob('*'):
            if not path.is_file() or any(part.startswith('.') or part=='__pycache__' for part in path.relative_to(candidate).parts):continue
            sources['model/stage2/vlm/'+path.relative_to(candidate).as_posix()]=path
    code_files=['__init__.py','stage1.py','stage1_tpo_merged.py',
                'stage2.py','stage2_v2.py','vlm.py','stage3.py']
    if use_nf4:code_files+=['vlm_candidate.py','stage2_nf4.py']
    if args.variant in ('nf4_fast','nf4_scale_fast','nf4_motion_fast','v4','v5'):code_files+=['stage3_fast.py']
    if args.variant in ('nf4_motion_fast','v4','v5'):code_files+=['stage2_motion_collision.py']
    if args.variant=='v4':code_files+=['stage1_v4.py','stage2_v4_d.py']
    if args.variant=='v5':
        selection=json.loads((ROOT/'artifacts/submissions/v5_selection_frozen.json').read_text(encoding='utf-8'))
        assert selection['status']=='selected_for_packaging'
        assert selection['stage2_module']=='stage2_motion_collision'
        assert selection['stage3_module']=='stage3_v5_compatible'
        required_frozen={'inference_v5.py','requirements_nf4.txt','scripts/build_submission.py',
                         'solution/stage1_v4.py','solution/stage2_motion_collision.py','solution/stage3_v5_compatible.py'}
        assert required_frozen <= selection['source_sha256'].keys()
        code_files+=['stage1_v4.py']
        for module in (selection['stage2_module'],selection['stage3_module']):
            if module+'.py' not in code_files:code_files.append(module+'.py')
        for relative,expected in selection['source_sha256'].items():
            path=(ROOT/relative).resolve()
            assert path.is_relative_to(ROOT.resolve())
            assert hashlib.file_digest(path.open('rb'),'sha256').hexdigest()==expected,relative
    if args.variant=='nf4_scale_fast':
        candidate=ROOT/'artifacts/candidates/stage3_scale'
        for filename in ('motion_model.joblib','training_provenance.json'):
            path=candidate/filename
            if not path.is_file():raise FileNotFoundError(f'Stage3 candidate weights/provenance required: {path}')
            sources['model/stage3/'+filename]=path
    for filename in code_files:
        path=ROOT/'solution'/filename
        sources['model/stage2/code/solution/'+path.name]=path
    if args.variant=='v5':
        expected=selection['expected_archive_sha256']
        assert sources.keys()==expected.keys(),sorted(sources.keys() ^ expected.keys())
        for name,path in sources.items():
            assert hashlib.file_digest(path.open('rb'),'sha256').hexdigest()==expected[name],name
    tree=ast.parse(sources['inference.py'].read_text(encoding='utf-8'))
    names={n.name for n in tree.body if isinstance(n,ast.FunctionDef)}
    assert {'predict_stage1','predict_stage2','predict_stage3'}<=names
    assert {'model/'+name for name in selected_stage1} <= sources.keys()
    if args.variant=='v1':
        assert 'model/stage2/vlm/model.safetensors' in sources
    else:
        index_name='model/stage2/vlm/model.safetensors.index.json'
        if index_name in sources:
            index=json.loads(sources[index_name].read_text(encoding='utf-8'))
            assert {'model/stage2/vlm/'+name for name in index['weight_map'].values()}<=sources.keys()
        else:
            assert 'model/stage2/vlm/model.safetensors' in sources
    assert any(k.startswith('model/stage3/') for k in sources)
    for stage in (1,2,3):
        assert f'model/stage2/code/solution/stage{stage}.py' in sources
    size=sum(p.stat().st_size for p in sources.values())
    assert size<32_000_000_000
    out=ROOT/'artifacts/submissions';out.mkdir(parents=True,exist_ok=True)
    target=out/({'v1':'submit_v1.zip','nf4':'submit_v2_nf4.zip',
                 'nf4_fast':'candidate_nf4_fast.zip',
                 'nf4_scale_fast':'submit_v3_scale_fast.zip',
                 'nf4_motion_fast':'submit_v3_motion_fast.zip','v4':'submit_v4.zip','v5':'submit_v5.zip'}[args.variant])
    if target.exists():raise FileExistsError(f'Preserve existing submission: {target}')
    manifest=[]
    with zipfile.ZipFile(target,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=1) as z:
        for name,path in sorted(sources.items()):
            z.write(path,name)
            manifest.append({'path':name,'bytes':path.stat().st_size,
                             'sha256':hashlib.file_digest(path.open('rb'),'sha256').hexdigest()})
    assert target.stat().st_size<10_000_000_000
    with zipfile.ZipFile(target) as z:
        assert z.testzip() is None
        assert {name.split('/')[0] for name in z.namelist()}=={'model','inference.py','requirements.txt'}
    record={'zip':str(target),'sha256':hashlib.file_digest(target.open('rb'),'sha256').hexdigest(),
            'zip_bytes':target.stat().st_size,'uncompressed_bytes':size,'files':manifest}
    target.with_suffix('.manifest.json').write_text(json.dumps(record,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in record.items() if k!='files'}),flush=True)

if __name__=='__main__':main()
