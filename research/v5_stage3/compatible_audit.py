"""Existing864 model only: bit/prediction equivalence and paired runtime, no accuracy search."""
from pathlib import Path
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='2'
import sys,json,time,hashlib,traceback,ast
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import cv2,numpy as np,pandas as pd,joblib
from threadpoolctl import threadpool_limits
from solution import stage3_fast as baseline
from solution import stage3_v5_compatible as compatible
from research.v4_stage3.audit import protection,differences
OUT=ROOT/'research/v5_stage3/compatible_864'

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(n,v):(OUT/n).write_text(json.dumps(v,indent=2,default=str),encoding='utf8')

def inputs():
    original=json.loads((ROOT/'research/v4_stage3/experiment_plan_frozen.json').read_text())
    routes=set(original['external_stress']['train_external']+original['external_stress']['validation_external'])
    rows=[]
    for p in sorted((ROOT/'external_data/comma2k19').glob('*_manifest.json')):
        rows.extend(r for r in json.loads(p.read_text()) if r['route'] in routes)
    assert len(rows)==23 and {r['route'] for r in rows}==routes
    excluded={r['segment'] for r in json.loads((ROOT/'research/stage3_external_overlap.json').read_text())['excluded']}
    assert not any(r['segment'] in excluded for r in rows)
    return rows

def freeze(rows):
    OUT.mkdir(parents=True,exist_ok=False)
    source=ROOT/'solution/stage3_v5_compatible.py';text=source.read_text();tree=ast.parse(text)
    imports=[]
    for n in ast.walk(tree):
        if isinstance(n,ast.Import):imports.extend(i.name for i in n.names)
        elif isinstance(n,ast.ImportFrom):imports.append('.'*n.level+(n.module or ''))
    assert not any('stage3_v5' in x for x in imports)
    assert not any(token in text for token in ('GlobalResidual','1128','IRLS'))
    assert sha(ROOT/'model/stage3/motion_model.joblib')=='c52ecd15cf68856f2a7b0b37b120c7a961b3bd8889a33af4ec47f8b9719bb3b3'
    frozen={'timestamp_unix':time.time(),'operation':'Existing864 frozen model compatible compute optimization only; no accuracy search',
        'source_sha256':sha(source),'baseline_source_sha256':sha(ROOT/'solution/stage3_fast.py'),'model_sha256':sha(ROOT/'model/stage3/motion_model.joblib'),
        'audit_source_sha256':sha(Path(__file__)),'imports':imports,'external_records':rows,
        'contracts':['original864 cache bit identity on23 external and5 public full decoded sequences','both existinghead predictions identical on every cached and computed row','actual10Hz public features/predictions identical','synthetic signedzero/nonfinite fallback','oneframe/renamedfile/otherfile invariance'],
        'runtime':{'function':'complete predict_stage3 including read-only model loading, extraction, classification and output assembly',
          'sets':['public_actual10hz2998','external_repeat5min3000_runtime_only'],'repeats':3,
          'order':[['baseline','compatible'],['compatible','baseline'],['baseline','compatible']],
          'gate':'median compatible / median baseline <=1.0 in both sets AND identical DataFrame in every paired run',
          'local_only_not_server_guarantee':True},
        'synthetic_seed':49013,'cpu_threads':2,'gpu':False,'training_allowed':False,'new_model_allowed':False,
        'prior_reports_preserved':{str(p.relative_to(ROOT)):sha(p) for p in [ROOT/'research/v5_stage3/audit_report.json',ROOT/'research/v5_stage3/summary.md',ROOT/'research/v5_stage3/equivalent_optimization/audit_report.json',ROOT/'research/v5_stage3/equivalent_optimization/experiment_report.json',ROOT/'research/v5_stage3/equivalent_optimization/summary.md']},
        'protected_before':protection()}
    save('plan_frozen.json',frozen);return frozen

def synthetic_tests():
    rng=np.random.default_rng(49013);cases=[]
    for h,w in ((96,256),(144,256),(160,256),(145,257)):
        cases.extend([(f'{h}x{w}_zero',np.zeros((h,w,2),np.float32)),
            (f'{h}x{w}_negative_zero',np.full((h,w,2),-0.,np.float32)),
            (f'{h}x{w}_normal',rng.normal(0,20,(h,w,2)).astype(np.float32)),
            (f'{h}x{w}_ties',rng.choice(np.array([-2.,-0.,0.,1.,2.],np.float32),size=(h,w,2))),
            (f'{h}x{w}_nan',np.full((h,w,2),np.nan,np.float32)),
            (f'{h}x{w}_largefinite',rng.normal(0,10000,(h,w,2)).astype(np.float32))])
        sparse=rng.normal(0,20,(h,w,2)).astype(np.float32);sparse[::9,::9]=np.nan
        cases.append((f'{h}x{w}_sparse_nan',sparse))
    fixed=np.load(ROOT/'research/v5_stage3/equivalent_optimization/fixed_real_flows.npz')
    cases.extend(('fixed_real_'+k,fixed[k]) for k in sorted(fixed.files,key=int))
    a,b=baseline.FrameFeatureComputer(),compatible.FrameFeatureComputer();reports=[]
    with np.errstate(invalid='ignore'):
        for name,f in cases:
            x,y=a(f),b(f);d=differences(x,y);assert d['bit_equal'],name
            reports.append({'name':name,**d})
    save('same_flow_contract.json',reports)
    return {'synthetic_count':28,'fixed_real_count':len(fixed.files),'all_bit_exact':True,'max_absolute_error':0.}

def feature_cache_tests(rows,model):
    cases=[]
    for p in sorted((ROOT/'Baseline/data/stage3/videos').glob('*.mp4')):
        cases.append((p.stem,p,ROOT/'research/stage3_cache'/f'{p.stem}.npy'))
    for r in rows:
        tag='ext_'+r['segment'].replace('/','_').replace('|','_')
        cases.append((r['segment'],ROOT/r['local']/'video.hevc',ROOT/'research/stage3_cache'/(tag+'.npy')))
    reports=[]
    for i,(name,path,cache) in enumerate(cases):
        started=time.perf_counter();x=np.load(cache);y=compatible.extract_motion(path,20.)
        d=differences(x,y);assert d['bit_equal'] and y.shape[1]==864 and np.isfinite(y).all(),name
        changed={}
        for t in ('accel','steer'):
            a=model[t].predict(x);b=model[t].predict(y);changed[t]=int(np.count_nonzero(a!=b));assert changed[t]==0
        reports.append({'source':name,'rows':len(y),'cache_sha256':sha(cache),'features':d,'prediction_changed':changed,'seconds':time.perf_counter()-started})
        save('cache_contract_partial.json',reports)
        print('compatible cache',i+1,'/28',len(y),'bit exact; head changes',changed,flush=True)
    return reports

def actual_public_tests(model):
    reports=[]
    for p in sorted((ROOT/'artifacts/public_eval_10hz/stage3/videos').glob('*.mp4')):
        x=baseline.extract_motion(p,10.);y=compatible.extract_motion(p,10.)
        d=differences(x,y);assert d['bit_equal'] and np.isfinite(y).all()
        changed={t:int(np.count_nonzero(model[t].predict(x)!=model[t].predict(y))) for t in ('accel','steer')}
        assert not any(changed.values())
        reports.append({'file':p.name,'rows':len(y),'features':d,'prediction_changed':changed})
    assert sum(r['rows'] for r in reports)==2998
    save('actual_public_contract.json',reports);return reports

def runtime_tests():
    datasets={'public':ROOT/'artifacts/public_eval_10hz/stage3',
              'external_runtime':ROOT/'research/stage3_fast_benchmark/external_stress'}
    result={}
    for name,data in datasets.items():
        runs={'baseline':[],'compatible':[]};equality=[]
        for rep,order in enumerate((('baseline','compatible'),('compatible','baseline'),('baseline','compatible'))):
            tables={}
            for arm in order:
                start=time.perf_counter();tables[arm]=(baseline if arm=='baseline' else compatible).predict_stage3(data,ROOT/'model/stage3')
                runs[arm].append(time.perf_counter()-start)
                print('compatible runtime',name,rep,arm,round(runs[arm][-1],3),flush=True)
            assert tables['baseline'].equals(tables['compatible'])
            for _,group in tables['compatible'].groupby('ID'):
                assert group.sample_index.to_list()==list(range(len(group)))
            equality.append(True)
            if rep==0:
                tables['baseline'].to_csv(OUT/f'{name}_baseline_predictions.csv',index=False)
                tables['compatible'].to_csv(OUT/f'{name}_compatible_predictions.csv',index=False)
            save('runtime_partial.json',{'completed':result,'current':name,'seconds':runs})
        ratio=float(np.median(runs['compatible'])/np.median(runs['baseline']))
        result[name]={'rows':len(tables['compatible']),'seconds':runs,'median_ratio':ratio,'passed':ratio<=1.,'all_paired_dataframes_equal':all(equality)}
        save('runtime_partial.json',{'completed':result})
    assert result['public']['rows']==2998 and result['external_runtime']['rows']==3000
    return result

def file_contracts():
    folder=OUT/'contract_inputs'/'videos';folder.mkdir(parents=True)
    os.link(ROOT/'artifacts/public_eval_10hz/stage3/videos/OPEN_001.mp4',folder/'RENAMED.mp4')
    before=compatible.predict_stage3(folder.parent,ROOT/'model/stage3')
    original=pd.read_csv(OUT/'public_baseline_predictions.csv');expected=original[original.ID=='OPEN_001'].reset_index(drop=True).copy();expected.ID='RENAMED'
    assert before.equals(expected)
    os.link(ROOT/'artifacts/public_eval_10hz/stage3/videos/OPEN_002.mp4',folder/'OTHER.mp4')
    together=compatible.predict_stage3(folder.parent,ROOT/'model/stage3')
    assert before.equals(together[together.ID=='RENAMED'].reset_index(drop=True))
    single=OUT/'single_frame'/'videos';single.mkdir(parents=True)
    p=single/'SINGLE.avi';writer=cv2.VideoWriter(str(p),cv2.VideoWriter_fourcc(*'FFV1'),10.,(256,144));assert writer.isOpened()
    writer.write(np.zeros((144,256,3),np.uint8));writer.release()
    x=baseline.extract_motion(p,10.);y=compatible.extract_motion(p,10.)
    assert x.shape==y.shape==(1,864) and np.array_equal(x.view(np.uint32),y.view(np.uint32))
    a=baseline.predict_stage3(single.parent,ROOT/'model/stage3');b=compatible.predict_stage3(single.parent,ROOT/'model/stage3')
    assert len(b)==1 and a.equals(b)
    return {'renamed_file_equal':True,'other_file_addition_equal':True,'single_frame_features_and_real_model_output_equal':True,'all_runtime_rows_indices_equal':True}

def main():
    rows=inputs();frozen=freeze(rows);started=time.perf_counter();cv2.setNumThreads(2)
    try:
        model=joblib.load(ROOT/'model/stage3/motion_model.joblib')
        assert model['accel'].n_features_in_==864 and model['steer'].n_features_in_==864
        syn=synthetic_tests();cached=feature_cache_tests(rows,model);actual=actual_public_tests(model)
        runtime=runtime_tests();files=file_contracts()
        after=protection();assert all(after.get(k)==v for k,v in frozen['protected_before'].items())
        assert all(sha(ROOT/p)==value for p,value in frozen['prior_reports_preserved'].items())
        assert sha(ROOT/'solution/stage3_v5_compatible.py')==frozen['source_sha256']
        passed=all(r['passed'] for r in runtime.values())
        report={'status':'compatible864_contract_and_runtime_passed_root_review' if passed else 'rejected_compatible864_runtime_gate',
          'all_features_bit_exact':True,'all_existing_head_predictions_identical':True,'synthetic':syn,'cached_sources':cached,
          'actual10hz_public':actual,'runtime':runtime,'file_contracts':files,'protected_unchanged':True,'prior_negative_reports_preserved':True,
          'no_training':True,'no_accuracy_search':True,'no_new_model':True,'gpu_used':False,'seconds':time.perf_counter()-started}
    except Exception:
        report={'status':'rejected_compatible864_contract_or_execution','error':traceback.format_exc(),'no_training':True,'no_new_model':True,'seconds':time.perf_counter()-started}
        save('audit_report.json',report);raise
    save('audit_report.json',report)
    print('COMPATIBLE FINAL',json.dumps({'status':report['status'],'ratios':{k:v['median_ratio'] for k,v in runtime.items()},'seconds':report['seconds']}),flush=True)

if __name__=='__main__':
    with threadpool_limits(limits=2):main()
