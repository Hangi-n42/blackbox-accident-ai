from pathlib import Path
import sys,json,csv,time,ast,hashlib
import numpy as np
import cv2
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from solution import stage1,stage1_v4
OUT=Path(__file__).resolve().parent


def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()


def main():
    cv2.setNumThreads(2)
    audit=json.loads((OUT/'public_decoder_audit.json').read_text());rows=[]
    for source in audit:
        path=ROOT/source['path']
        t=time.perf_counter();old=stage1.sample_video(path);old_time=time.perf_counter()-t
        t=time.perf_counter();new,info=stage1_v4.sample_video_diagnostic(path);new_time=time.perf_counter()-t
        assert len(new)==len(old)==12 and all(np.array_equal(a,b) for a,b in zip(old,new))
        assert info['method']=='seek_positions_checked'
        hashes=[hashlib.sha256(f.tobytes()).hexdigest() for f in new]
        assert hashes==source['reference_rgb_sha256']
        x,_=stage1.extract_features(old);y,_=stage1_v4.extract_features(new)
        assert np.array_equal(x,y)
        rows.append({'ID':source['ID'],'sampling':info,'rgb_reference_exact':True,'feature_max_abs_difference':float(np.max(np.abs(x-y))),
                     'old_seconds':old_time,'candidate_seconds':new_time})
    tree=lambda p:next(n for n in ast.parse(p.read_text(encoding='utf-8')).body if isinstance(n,ast.FunctionDef) and n.name=='predict_stage1')
    ast_equal=ast.dump(tree(ROOT/'solution/stage1.py'))==ast.dump(tree(ROOT/'solution/stage1_v4.py'))
    assert ast_equal
    assert stage1.extract_features is stage1_v4.extract_features
    assert stage1.feature_probability is stage1_v4.feature_probability
    # An already acquired HEVC known to have broken metadata is a real fallback case.
    comma=json.loads((ROOT/'research/stage1/comma_original_diagnostic/predictions.json').read_text())[0]
    comma_sources=json.loads((ROOT/'research/stage1/comma_original_diagnostic/sources.json').read_text())['videos']
    source=next(s for s in comma_sources if s['segment']==comma['segment'])
    t=time.perf_counter();frames,info=stage1_v4.sample_video_diagnostic(ROOT/source['video_path']);elapsed=time.perf_counter()-t
    assert [hashlib.sha256(f.tobytes()).hexdigest() for f in frames]==comma['frame_sampling']['decoded_rgb_sha256']
    hevc={'segment':source['segment'],'sampling':info,'reference_rgb_exact':True,'seconds':elapsed}
    # Existing repeated-video stress artifact: speed/equality only, no generalization claim.
    stress=ROOT/'research/stage3_fast_benchmark/external_stress/videos/comma_repeat5.mp4'
    t=time.perf_counter();old=stage1.sample_video(stress);old_time=time.perf_counter()-t
    t=time.perf_counter();new,info=stage1_v4.sample_video_diagnostic(stress);new_time=time.perf_counter()-t
    assert len(new)==len(old) and all(np.array_equal(a,b) for a,b in zip(old,new))
    stress_result={'path':str(stress.relative_to(ROOT)),'sampling':info,'rgb_exact':True,'old_seconds':old_time,'candidate_seconds':new_time}
    frozen=json.loads((ROOT/'research/stage1/dlc_subset/frozen_diagnostic_manifest.json').read_text())['artifact_sha256']
    unchanged={k:sha(ROOT/k)==v for k,v in frozen.items()}
    assert all(unchanged.values())
    result={'public':rows,'public_count':len(rows),'public_rgb_frames_exact':120,
            'predict_function_ast_identical':ast_equal,'shared_feature_functions':True,
            'gpu_used':False,'output_equivalence_basis':'Identical RGB inputs, shared feature functions, identical prediction-function AST, unchanged model files; no new GPU forward required',
            'actual_hevc_recovery':hevc,'five_minute_stress':stress_result,'frozen_assets_unchanged':unchanged}
    (OUT/'candidate_validation.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps({'public_exact_frames':120,'normal_seek_path':all(r['sampling']['method']=='seek_positions_checked' for r in rows),
                      'prediction_ast_identical':ast_equal,'public_old_seconds':sum(r['old_seconds'] for r in rows),
                      'public_candidate_seconds':sum(r['candidate_seconds'] for r in rows), 'hevc':hevc,'stress':stress_result},indent=2))


if __name__=='__main__':main()
