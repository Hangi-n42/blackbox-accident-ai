"""One equivalent V5 optimization audit; original failed experiment is immutable."""
from pathlib import Path
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='2'
import sys,json,time,hashlib,traceback,cProfile,pstats,io
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import cv2,numpy as np
from unittest.mock import patch
from threadpoolctl import threadpool_limits
from solution import stage3_v5 as original
from solution import stage3_v5_fast as optimized
from solution import stage3_fast as baseline
from research.v4_stage3.audit import protection,differences
OUT=ROOT/'research/v5_stage3/equivalent_optimization'

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(n,v):(OUT/n).write_text(json.dumps(v,indent=2,default=str),encoding='utf8')

def freeze():
    assert not (OUT/'optimization_plan_frozen.json').exists()
    plan=json.loads((ROOT/'research/v5_stage3/plan_frozen.json').read_text())
    frozen={'timestamp_unix':time.time(),'single_equivalent_implementation_attempt':1,
        'original_plan_sha256':sha(ROOT/'research/v5_stage3/plan_frozen.json'),
        'original_failed_audit_sha256':sha(ROOT/'research/v5_stage3/audit_report.json'),
        'original_summary_sha256':sha(ROOT/'research/v5_stage3/summary.md'),
        'original_feature_source_sha256':sha(ROOT/'solution/stage3_v5.py'),
        'optimized_feature_source_sha256':sha(ROOT/'solution/stage3_v5_fast.py'),
        'audit_source_sha256':sha(Path(__file__)),
        'profile_plan_sha256':sha(OUT/'profile_plan_frozen.json'),
        'intervention':'Cache ROI quantile indices/gamma; reuse one partition for median and percentiles. Preserve NumPy float32 median and float32-difference/float64-interpolation arithmetic. Signed-zero quantiles use original routine.',
        'unchanged':['IRLS3','192 fixed samples','ROI geometry','4 coefficient field','all1128 mathematical features','10Hz time summaries','model parameters','class/sample weights','source splits','statistical gates'],
        'contracts':['full1128 float32 bit exact synthetic and fixed24 real flows','public5 original20Hz and actual10Hz full sequences against original1128 caches','one complete real Civic20Hz video against original1128','same decoded pixel20/10 alignment','finite outputs and row counts'],
        'synthetic_seed':49013,'synthetic_shapes':[[96,256],[144,256],[160,256],[145,257]],
        'local_runtime_gate':plan['local_runtime_gate'],'source_gate':plan['source_gate'],'public_gate':plan['public_gate'],
        'external_stress':plan['external_stress'],'model_params':plan['model_params'],
        'final_checkpoint_or_refit_authorized':False,'cpu_threads':2,'gpu':False,'protected_before':protection()}
    save('optimization_plan_frozen.json',frozen)
    return frozen

def synthetic_tests():
    rng=np.random.default_rng(49013);flows=[];labels=[]
    for h,w in ((96,256),(144,256),(160,256),(145,257)):
        yy,xx=np.mgrid[:h,:w].astype(np.float64);X,Y=(xx-w/2)/w,(yy-h/2)/w
        cases={'zero':np.zeros((h,w,2),np.float32),'negative_zero':np.full((h,w,2),-0.,np.float32),
               'gaussian':rng.normal(0,20,(h,w,2)).astype(np.float32),
               'discrete_ties':rng.choice(np.array([-2.,-0.,0.,1.,2.],np.float32),size=(h,w,2)),
               'global_field':(np.stack((.025+.08*X+.03*Y,-.015-.03*X+.08*Y),axis=-1)*w).astype(np.float32),
               'large_finite':rng.normal(0,10000,(h,w,2)).astype(np.float32)}
        bad=np.full((h,w,2),np.nan,np.float32);cases['all_nan']=bad
        sparse=cases['gaussian'].copy();sparse[::9,::9]=np.nan;cases['sparse_nan']=sparse
        for name,f in cases.items():flows.append(f);labels.append(f'{h}x{w}_{name}')
    real=np.load(OUT/'fixed_real_flows.npz')
    for k in sorted(real.files,key=int):flows.append(real[k]);labels.append('fixed_real_'+k)
    oldbase,newbase=original.FrameFeatureComputer(),optimized.FrameFeatureComputer()
    oldextra,newextra=original.GlobalResidualComputer(),optimized.GlobalResidualComputer()
    records=[];ob=[];oe=[];nb=[];ne=[]
    with np.errstate(invalid='ignore'):
        for label,f in zip(labels,flows):
            b0,b1=oldbase(f),newbase(f);e0,e1=oldextra(f),newextra(f)
            rec={'case':label,'base144':differences(b0,b1),'extra44':differences(e0,e1)}
            records.append(rec);save('same_flow_partial.json',records)
            assert rec['base144']['bit_equal'] and rec['extra44']['bit_equal'],label
            ob.append(b0);nb.append(b1);oe.append(e0);ne.append(e1)
    centers=np.arange(len(flows),dtype=float)+.5;count=len(flows)+1
    a=np.concatenate((original.summarize(ob,centers,count),original.summarize(oe,centers,count)),axis=1)
    b=np.concatenate((optimized.summarize(nb,centers,count),optimized.summarize(ne,centers,count)),axis=1)
    temporal=differences(a,b);assert temporal['bit_equal']
    # Informational single profile of the frozen optimized implementation; no tuning follows.
    profile={}
    fixed=[real[k] for k in sorted(real.files,key=int)]
    for name,cls in [('optimized_base144',optimized.FrameFeatureComputer),('optimized_extra44',optimized.GlobalResidualComputer)]:
        c=cls();p=cProfile.Profile();p.enable()
        for _ in range(10):
            for f in fixed:c(f)
        p.disable();p.dump_stats(str(OUT/(name+'.prof')))
        stream=io.StringIO();pstats.Stats(p,stream=stream).sort_stats('cumulative').print_stats(20)
        (OUT/(name+'.txt')).write_text(stream.getvalue(),encoding='utf8')
    return {'cases':records,'temporal1128':temporal,'real_cases':len(real.files),'synthetic_cases':len(flows)-len(real.files)}

def public_tests():
    original_capture=cv2.VideoCapture
    class EverySecondCapture:
        def __init__(self,path):self.cap=original_capture(path)
        def read(self):
            ok,frame=self.cap.read()
            if ok:self.cap.read()
            return ok,frame
        def release(self):self.cap.release()
    cache=OUT/'cache_v5_fast';cache.mkdir(exist_ok=False)
    records={}
    for path in sorted((ROOT/'Baseline/data/stage3/videos').glob('*.mp4')):
        x,detail=optimized.extract_motion(path,20.,return_diagnostics=True)
        old=np.load(ROOT/'research/v5_stage3/cache_v5'/f'{path.stem}_v5.npy')
        with patch.object(optimized.cv2,'VideoCapture',EverySecondCapture):same=optimized.extract_motion(path,10.)
        rec={'full1128_vs_original':differences(old,x),'same_pixel20_10':differences(x[::2],same),**detail}
        assert rec['full1128_vs_original']['bit_equal'] and rec['same_pixel20_10']['bit_equal']
        assert x.shape[1]==1128 and np.isfinite(x).all()
        np.save(cache/f'{path.stem}_v5_fast.npy',x)
        records[path.stem]=rec;save('public_contract_partial.json',records)
        print('optimized contract',path.stem,'1128 bit exact',flush=True)
    return records

def real_external_test():
    plan=json.loads((OUT/'profile_plan_frozen.json').read_text());record=plan['external_record']
    path=ROOT/record['local']/'video.hevc'
    a=original.extract_motion(path,20.);b,detail=optimized.extract_motion(path,20.,return_diagnostics=True)
    result={'record':record,'full1128':differences(a,b),**detail}
    assert result['full1128']['bit_equal'] and np.isfinite(b).all()
    tag='ext_'+record['segment'].replace('/','_').replace('|','_')
    np.save(OUT/'cache_v5_fast'/f'{tag}_v5_fast.npy',b)
    save('real_external_contract.json',result)
    print('complete real Civic contract1128 exact',detail,flush=True)
    return result

def runtime_tests():
    sets={'public':sorted((ROOT/'artifacts/public_eval_10hz/stage3/videos').glob('*.mp4')),
          'external_runtime':[ROOT/'research/stage3_fast_benchmark/external_stress/videos/comma_repeat5.mp4']}
    result={}
    for name,paths in sets.items():
        runs={'baseline':[],'optimized':[]};details=[]
        references=[np.load(ROOT/'research/v5_stage3/cache_v5'/f'{p.stem}_corrected10_v5.npy') for p in paths] if name=='public' else [original.extract_motion(p,10.) for p in paths]
        for rep,order in enumerate((('baseline','optimized'),('optimized','baseline'),('baseline','optimized'))):
            arrays={}
            for arm in order:
                start=time.perf_counter();arrays[arm]=[(baseline if arm=='baseline' else optimized).extract_motion(p,10.) for p in paths]
                runs[arm].append(time.perf_counter()-start)
                print('optimized runtime',name,rep,arm,round(runs[arm][-1],3),flush=True)
            for p,base,new,old in zip(paths,arrays['baseline'],arrays['optimized'],references):
                d=differences(old,new);b=differences(base,new[:,:864])
                assert d['bit_equal'] and b['bit_equal'] and np.isfinite(new).all()
                details.append({'path':str(p.relative_to(ROOT)),'repeat':rep,'full1128':d,'prefix864':b})
                if name=='public' and rep==0:np.save(OUT/'cache_v5_fast'/f'{p.stem}_corrected10_v5_fast.npy',new)
            save('runtime_partial.json',{'completed':result,'current':name,'runs':runs})
        ratio=float(np.median(runs['optimized'])/np.median(runs['baseline']))
        result[name]={'seconds':runs,'median_ratio':ratio,'passed':ratio<=1.25,'rows':sum(len(a) for a in arrays['baseline']),'contracts':details}
        save('runtime_partial.json',{'completed':result})
    return result

def main():
    frozen=freeze();started=time.perf_counter();cv2.setNumThreads(2)
    try:
        same=synthetic_tests();public=public_tests();external=real_external_test();runtime=runtime_tests()
        after=protection();unchanged=all(after.get(k)==v for k,v in frozen['protected_before'].items())
        assert unchanged
        assert sha(ROOT/'research/v5_stage3/audit_report.json')==frozen['original_failed_audit_sha256']
        assert sha(ROOT/'research/v5_stage3/summary.md')==frozen['original_summary_sha256']
        assert sha(ROOT/'solution/stage3_v5.py')==frozen['original_feature_source_sha256']
        passed=all(r['passed'] for r in runtime.values())
        report={'status':'equivalent_optimization_passed_proceed_source_stress' if passed else 'rejected_equivalent_optimization_local_runtime_no_training',
            'full1128_bit_exact':True,'same_flow':same,'public':public,'real_external':external,'runtime':runtime,
            'local_runtime_gate_passed':passed,'original_failed_result_preserved':True,'protected_unchanged':unchanged,
            'seconds':time.perf_counter()-started,'training_executed':False,'final_checkpoint_created':False}
    except Exception:
        report={'status':'rejected_equivalent_optimization_contract_or_execution','error':traceback.format_exc(),
            'seconds':time.perf_counter()-started,'training_executed':False,'final_checkpoint_created':False}
        save('audit_report.json',report);raise
    save('audit_report.json',report)
    print('OPTIMIZATION FINAL',json.dumps({'status':report['status'],'ratios':{k:v['median_ratio'] for k,v in runtime.items()},'seconds':report['seconds']}),flush=True)

if __name__=='__main__':
    with threadpool_limits(limits=2):main()
