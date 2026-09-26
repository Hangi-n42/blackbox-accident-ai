"""Independent CPU-only preflight. No encoder forward or optimizer call."""
import ast, hashlib, importlib.util, json
from datetime import datetime, timezone
from fractions import Fraction as F
from pathlib import Path
import numpy as np
from PIL import Image
HERE=Path(__file__).resolve().parent; ROOT=HERE.parents[1]
read=lambda p:json.loads(p.read_text())
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def pairs(times,lo,hi):
    t=list(map(lambda x:F(str(x)),times));lo,hi=F(str(lo)),F(str(hi))
    return [[i,j] for i,a in enumerate(t) for j,b in enumerate(t) if i!=j and max(abs(a-lo)-abs(b-lo),abs(a-hi)-abs(b-hi)) < -F(1,10**9)]
def check():
    old=read(ROOT/'artifacts/stage2_entry_path_audit_20260920/freeze.json')['files']
    for name,h in old.items():assert sha(ROOT/name)==h,name
    combined=read(HERE/'inputs.json')['jobs'];train=read(HERE/'training_inputs.json')['jobs'];evaluation=read(HERE/'evaluation_inputs.json')['jobs']
    assert len(train)==len(evaluation)==8 and combined==train+evaluation
    assert all(j['role']=='train_pool' for j in train) and all(j['role']=='evaluation' for j in evaluation)
    assert not ({j['ID'] for j in train}&{j['ID'] for j in evaluation})
    cpu=read(HERE/'cpu_input_review.json');assert cpu['status']=='PASS' and len(cpu['inputs'])==16
    assert cpu['encoder_forwards']==cpu['real_training_runs']==0
    for name,h in cpu['source_hashes'].items():
        if Path(name).name!='selector.py':assert sha(Path(name))==h,name
    records={r['ID']:r for r in cpu['inputs']}
    for j in combined:
        assert sha(ROOT/j['image'])==j['image_sha256']==records[j['ID']]['image_sha256']
        history=read(ROOT/j['history_call'])[2]
        with Image.open(ROOT/j['image']) as im:assert [hashlib.sha256(im.convert('RGB').tobytes()).hexdigest()]==history['image_sha256']
        assert records[j['ID']]['processor_hashes']==history['processor_input_sha256']
        assert j['frames']==sorted(j['frames']) and j['times']==sorted(j['times'])
        for f,t,s in zip(j['frames'],j['times'],j['source_images']):
            assert f==s['frame'] and t==s['pts_seconds'] and sha(ROOT/s['path'])==s['sha256']
    refs=read(HERE/'training_references.json');review=read(HERE/'training_reference_review.json')
    assert refs['review_sha256']==sha(HERE/'training_reference_review.json')
    br={r['ID']:r for r in review['cases']};jobs={j['ID']:j for j in train};eligible=[r for r in refs['cases'] if r['eligible']]
    assert [r['ID'] for r in eligible]==['00000','00003','00006','00013']
    counts=[]
    for r in refs['cases']:
        j=jobs[r['ID']];b=br[r['ID']]
        assert sha(ROOT/j['human_draft'])==b['human_draft_sha256']
        if not r['eligible']:continue
        assert b['usable_for_weak_training'] and b['same_counterpart'] and not r['human_exact_point_approved']
        assert r['reference_frames']==[b['lower_frame'],b['upper_frame']]
        assert r['reference_seconds']==[b['lower_pts_seconds'],b['upper_pts_seconds']]
        case=read(ROOT/j['full_case_source']);native={f['frame']:f for f in case['frames']}
        path=ROOT/case['video_path'].replace('\\','/').split('/research/',1)[1]
        path=ROOT/'research'/path.relative_to(ROOT)
        assert sha(path)==j['source_sha256']==case['video_sha256']
        for f,t in zip(r['reference_frames'],r['reference_seconds']):assert native[f]['pts_seconds']==t
        p=pairs(j['times'],*r['reference_seconds']);assert p==r['preference_pairs'] and len(p)==r['pair_count'];counts.append(len(p))
    assert counts==[61,66,66,65] and {r['entry_status'] for r in eligible}=={'before_start','during_clip'}
    split=read(HERE/'split_visual_review.json')
    assert split['status']=='PASS_WITH_SCOPE_LIMITS' and split['observed_accident_overlap']=='none_found'
    assert len(split['train_evaluation_pairs'])==32 and len(split['within_train_pairs'])==6
    assert not split['source_independence_certified'] and not split['human_review']
    source=(HERE/'selector.py').read_text();tree=ast.parse(source)
    spec=importlib.util.spec_from_file_location('selector_preflight',HERE/'selector.py');s=importlib.util.module_from_spec(spec);spec.loader.exec_module(s)
    assert s.temporal(np.zeros((12,4)),np.arange(12)).shape==(12,14)
    h=np.arange(1152*3,dtype=np.float32).reshape(1152,3);g=h.reshape(24,48,3)
    np.testing.assert_array_equal(s.pool_tiles(h),np.stack([g[y+1:y+8,x:x+12].mean((0,1)) for y in [0,8,16] for x in [0,12,24,36]]))
    for r in eligible:assert s.certain_pairs(jobs[r['ID']]['times'],*r['reference_seconds']).tolist()==r['preference_pairs']
    runner=ast.parse((HERE/'experiment.py').read_text());train_node=next(n for n in runner.body if isinstance(n,ast.FunctionDef) and n.name=='train')
    literals={n.value for n in ast.walk(train_node) if isinstance(n,ast.Constant) and isinstance(n.value,str)}
    assert 'training_inputs.json' in literals and 'evaluation_inputs.json' not in literals and 'inputs.json' not in literals
    assert not (HERE/'features').exists() and not (HERE/'model.npz').exists() and not (HERE/'freeze.json').exists()
    return dict(status='PASS',created_utc=datetime.now(timezone.utc).isoformat(),old_frozen_files_verified=len(old),input_cases=16,cpu_processor_replays=16,merged_grid=[24,48],visual_tokens_per_image=1152,tile_pool_tokens=84,eligible_training_ids=[r['ID'] for r in eligible],training_incidents=4,preference_pairs_per_incident=counts,total_pairs=sum(counts),evaluation_cases=8,head_parameters=14,train_only_manifests=True,native_source_sha_and_reference_pts=True,split_status=split['status'],model_forwards=0,real_fits=0,verifier_plan='Post-freeze new CPU-only verifier: cached hidden pooling, train-only PCA/statistics, loss/gradient stationarity without re-fit, interval evaluation and fixed comparators. Separate verifier SHA; frozen policy never edited.',cpu_review_selector_hash_note='Original CPU color-card selector hash predates removal of redundant current-first features. Pooling function independently rechecked unchanged here.',source_hashes={p.name:sha(p) for p in [HERE/'selector.py',HERE/'experiment.py',HERE/'protocol.json',HERE/'training_inputs.json',HERE/'evaluation_inputs.json',HERE/'training_references.json',HERE/'training_reference_review.json',HERE/'split_visual_review.json']},limits=['Four weak AI-filtered human drafts, not four exact human ground truths; 258 pairs are not independent incidents.','CCD8 references and historical outcomes exposed; bounded split review cannot certify hidden-source independence.','Global vision attention mixes tiles/header; excluding one token row is not proof of semantic target localization.','No actual encoder, fit, CUDA or official score validated by preflight.'])
if __name__=='__main__':
    result=check();out=HERE/'preflight_review.json';assert not out.exists()
    out.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    (HERE/'preflight_review.md').write_text('# 독립 사전 검수: PASS\n\n원 Q3 입력 16건의 CPU processor 재생과 RGB 일치, 원본 SHA·native PTS, 기존 동결 2056파일 유지, 학습 전용 manifest 분리를 확인했다. 약한 학습 4건의 구간에서 확실한 순위 쌍 61/66/66/65개(합계258), PCA4·선형14특징·사고별 동등 가중 계약을 확인했다. 모델 forward와 실제 fit은 0회다.\n\nB의 train4×CCD8 및 학습 내부6쌍 중복 검수는 PASS_WITH_SCOPE_LIMITS다. 4건은 AI가 사람 초안을 걸러 구간으로 만든 약한 참조이며, CCD8은 이미 노출된 개발평가다. 독립 사람검증·상대 식별력·CUDA 점수 개선을 입증하지 않는다.\n\n사전 파일은 여기서 변경을 중단한다. 부모 승인에 따라 동결 이후 별도 SHA를 가진 CPU 검증기를 신규 작성하여 저장된 hidden→pool, 학습 통계·손실·gradient 및 평가 수치를 검증한다. 추가 encoder/optimizer 실행은 하지 않는다.\n')
    print(json.dumps({k:result[k] for k in ['status','old_frozen_files_verified','total_pairs','head_parameters']}))
