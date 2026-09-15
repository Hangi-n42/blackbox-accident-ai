"""Freeze the single V5 protocol, then check calculation and local runtime gates."""
from pathlib import Path
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[key] = '2'
import sys, json, time, hashlib
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import cv2, numpy as np
from threadpoolctl import threadpool_limits
from unittest.mock import patch
from solution import stage3_v5 as candidate
from solution import stage3_fast as baseline
from research.v4_stage3.audit import protection, differences

OUT = Path(__file__).resolve().parent

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def save(name, value):
    (OUT/name).write_text(json.dumps(value, indent=2, default=str), encoding='utf8')

def freeze():
    path = OUT/'plan_frozen.json'
    if path.exists():
        raise FileExistsError('Do not overwrite pre-result frozen protocol')
    prior = ROOT/'research/v4_stage3/experiment_plan_frozen.json'
    assert sha(prior) == 'a01a9f4d555c13a0481dc1ae75c257d890b12afb9b7dc2e3de818b1644277102'
    old = json.loads(prior.read_text())
    frozen = {'timestamp_unix': time.time(), 'authorization': 'root agreed single candidate implementation and gated evaluation',
        'status': 'frozen_before_extraction_or_results', 'candidate_count': 1,
        'design_sha256': sha(OUT/'design.md'), 'feature_source_sha256': sha(ROOT/'solution/stage3_v5.py'),
        'audit_source_sha256': sha(Path(__file__)), 'source_split_sha256': sha(prior),
        'external_stress': old['external_stress'], 'public_oof': old['public_oof'],
        'public_ids': ['OPEN_001', 'OPEN_002', 'OPEN_003', 'OPEN_004', 'OPEN_005'],
        'features': {'original_prefix':864, 'extra_raw':44, 'extra_temporal':264, 'total':1128,
            'grid':[16,12], 'x_bounds':[.03,.97], 'y_bounds':[.15,.72], 'irls_reweightings':3,
            'robust_scale':'max(median(vector_residual_norm),1e-6)', 'huber_factor':1.345,
            'min_finite_samples':16, 'temporal_windows':[5,15,31], 'temporal_lags':[5,15],
            'fit':'U=tx+aX-bY; V=ty+bX+aY; X,Y,U,V normalized by width',
            'new_raw_order':'tx,ty,a,b,residual_median,residual_p80,mean_weight,valid; 12ROI x [medianRu,medianRv,p80residualmag]'},
        'local_runtime_gate': {'repeats':3, 'statistic':'median of full-set extraction times per arm',
            'maximum_ratio':1.25, 'datasets':['public_actual10hz_2998rows','comma_repeat5_runtime_only_3000rows'],
            'order':[['baseline','candidate'],['candidate','baseline'],['baseline','candidate']],
            'not_server_guarantee':True, 'other_work':'Other stage may use CPU2 concurrently; paired alternating order reduces but does not remove contention.'},
        'source_gate': {'both_head_delta_min':0., 'stage3_delta_gt':1e-12, 'positive_routes_min':7, 'routes':12},
        'public_gate': {'both_head_delta_min':0., 'stage3_delta_gt':1e-12},
        'model_params': old['model_params'], 'sample_weight':'all ones, existing balanced class_weight; no augmentation',
        'rf_fraction_effect': {'max_features':.4,'baseline_dimensions_per_split':345,'candidate_dimensions_per_split':451,
            'interpretation':'Combined feature-expansion intervention, not pure information-only causal isolation.'},
        'training_sample_hz':2, 'external_validation_hz':10, 'gpu':False, 'threads':2,
        'final_refit_authorized':False, 'final_checkpoint_allowed':False,
        'known_duplicate_exclusion':json.loads((ROOT/'research/stage3_external_overlap.json').read_text()),
        'manifests':{str(p.relative_to(ROOT)):sha(p) for p in sorted((ROOT/'external_data/comma2k19').glob('*_manifest.json'))},
        'existing_v4_baseline_report_sha256':sha(ROOT/'research/v4_stage3/experiment_report.json'),
        'protected_before':protection()}
    save(path.name, frozen)
    return frozen

def same_pixel_tests():
    records = {}
    cache = OUT/'cache_v5'
    cache.mkdir(exist_ok=True)
    original_capture = cv2.VideoCapture
    class EverySecondCapture:
        def __init__(self, path):
            self.cap = original_capture(path)
        def read(self):
            ok, frame = self.cap.read()
            if ok:
                self.cap.read()
            return ok, frame
        def release(self):
            self.cap.release()
    for path in sorted((ROOT/'Baseline/data/stage3/videos').glob('*.mp4')):
        trained, detail = candidate.extract_motion(path,20.,return_diagnostics=True)
        cached = np.load(ROOT/'research/stage3_cache'/f'{path.stem}.npy')
        with patch.object(candidate.cv2, 'VideoCapture', EverySecondCapture):
            deployed = candidate.extract_motion(path,10.)
        rec = {'prefix_vs_training_cache':differences(trained[:,:864],cached),
            'same_pixel_20_stride2_vs_10_stride1':differences(trained[::2],deployed), **detail}
        assert rec['prefix_vs_training_cache']['bit_equal']
        assert rec['same_pixel_20_stride2_vs_10_stride1']['bit_equal']
        assert trained.shape[1] == 1128 and np.isfinite(trained).all()
        np.save(cache/f'{path.stem}_v5.npy',trained)
        records[path.stem] = rec
        save('same_pixel_partial.json',records)
        print('contract',path.stem,'prefix/time exact',detail,flush=True)
    return records

def numerical_tests():
    comp = candidate.GlobalResidualComputer()
    zero = comp(np.zeros((144,256,2),np.float32))
    assert zero.shape == (44,) and zero[7] == 1 and np.isfinite(zero).all()
    assert np.array_equal(zero[:6],np.zeros(6,np.float32))
    bad = comp(np.full((144,256,2),np.nan,np.float32))
    assert bad[7] == 0 and np.isfinite(bad).all() and comp.failures == 1
    yy,xx = np.mgrid[:144,:256].astype(float)
    X,Y=(xx-128)/256,(yy-72)/256
    truth=np.array([.025,-.015,.08,-.03])
    tx,ty,a,b=truth
    flow=np.stack((tx+a*X-b*Y,ty+b*X+a*Y),axis=-1)*256
    result=comp(flow.astype(np.float32))
    error=float(np.max(np.abs(result[:4]-truth)))
    assert error<1e-6
    return {'zero_flow_passed':True,'nonfinite_fallback_passed':True,'exact_synthetic_field_coef_max_error':error}

def runtime_tests():
    datasets={'public':sorted((ROOT/'artifacts/public_eval_10hz/stage3/videos').glob('*.mp4')),
        'external_runtime':[ROOT/'research/stage3_fast_benchmark/external_stress/videos/comma_repeat5.mp4']}
    result={}
    for name,paths in datasets.items():
        runs={'baseline':[],'candidate':[]}
        details=[]
        for repeat,order in enumerate((('baseline','candidate'),('candidate','baseline'),('baseline','candidate'))):
            arrays={}
            for arm in order:
                start=time.perf_counter()
                arrays[arm]=[(candidate if arm=='candidate' else baseline).extract_motion(p,10.) for p in paths]
                seconds=time.perf_counter()-start
                runs[arm].append(seconds)
                print('runtime',name,repeat,arm,round(seconds,3),flush=True)
            for p,x,y in zip(paths,arrays['baseline'],arrays['candidate']):
                detail=differences(x,y[:,:864])
                assert detail['bit_equal'] and np.isfinite(y).all()
                details.append({'path':str(p.relative_to(ROOT)),'repeat':repeat,**detail})
                if name=='public' and repeat==0:
                    np.save(OUT/'cache_v5'/f'{p.stem}_corrected10_v5.npy',y)
            save('runtime_partial.json',{'completed':result,'current_dataset':name,'runs':runs})
        ratio=float(np.median(runs['candidate'])/np.median(runs['baseline']))
        result[name]={'seconds':runs,'median_ratio':ratio,'passed':ratio<=1.25,'prefix_checks':details,
                      'rows':sum(len(a) for a in arrays['baseline'])}
        save('runtime_partial.json',{'completed':result})
    return result

def main():
    frozen=freeze()
    started=time.perf_counter()
    cv2.setNumThreads(2)
    numerics=numerical_tests()
    contracts=same_pixel_tests()
    runtime=runtime_tests()
    after=protection()
    # Other stages may create new ZIPs concurrently; all originally present assets must survive unchanged.
    unchanged=all(after.get(k)==v for k,v in frozen['protected_before'].items())
    assert unchanged
    passed=all(r['passed'] for r in runtime.values())
    report={'status':'audit_passed_proceed_source_stress' if passed else 'rejected_local_runtime_gate_no_training',
        'numerical_tests':numerics,'same_pixel':contracts,'runtime':runtime,'local_runtime_gate_passed':passed,
        'local_runtime_not_server_guarantee':True,'protected_unchanged':unchanged,
        'seconds':time.perf_counter()-started,'training_executed':False,'final_checkpoint_created':False}
    save('audit_report.json',report)
    print(json.dumps({'status':report['status'],'ratios':{k:v['median_ratio'] for k,v in runtime.items()},'seconds':report['seconds']}),flush=True)

if __name__=='__main__':
    with threadpool_limits(limits=2):
        main()
