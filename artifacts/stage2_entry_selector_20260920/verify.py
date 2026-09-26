"""Independent saved-array verification; no model load, encoder, generation or fit.

Created after the policy freeze with the parent's explicit permission. This verifier
is separately hashed; it cannot change frozen labels, choices, or model weights.
"""
import hashlib, json, sys
from collections import Counter
from datetime import datetime, timezone
from fractions import Fraction as F
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
OLD=ROOT/'artifacts/stage2_entry_path_audit_20260920'
read=lambda p:json.loads(p.read_text())
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def digest(a):return hashlib.sha256(a.tobytes()).hexdigest()
def close(a,b,atol=1e-9,rtol=1e-8):np.testing.assert_allclose(a,b,atol=atol,rtol=rtol)
def compare(a,b):
    if isinstance(a,dict):
        assert set(a)==set(b),(set(a),set(b))
        for k in a:compare(a[k],b[k])
    elif isinstance(a,list):
        assert len(a)==len(b)
        for x,y in zip(a,b):compare(x,y)
    elif isinstance(a,float):close(a,b)
    else:assert a==b,(a,b)
def freeze_check():
    f=read(HERE/'freeze.json');old=read(OLD/'freeze.json')['files']
    assert all(f['files'].get(k)==h for k,h in old.items())
    for name,h in f['files'].items():assert sha(ROOT/name)==h,name
    return f
def pairs(times,lo,hi):
    times=[F(str(x)) for x in times];lo,hi=F(str(lo)),F(str(hi))
    # |a-T|-|b-T| is monotone in T. Endpoints certify the entire interval.
    return np.array([(i,j) for i,a in enumerate(times) for j,b in enumerate(times) if i!=j and max(abs(a-lo)-abs(b-lo),abs(a-hi)-abs(b-hi)) < -F(1,10**9)],dtype=int).reshape(-1,2)
def grade(p,lo,hi):
    p,lo,hi=[F(str(x)) for x in [p,lo,hi]];near=max(lo-p,p-hi,F(0));far=max(abs(lo-p),abs(hi-p))
    return dict(result='correct' if far<=F(3,10) else 'wrong' if near>F(3,10) else 'indeterminate',minimum_error_seconds=float(near),maximum_error_seconds=float(far))
def delta(a,b,lo,hi):
    a,b,lo,hi=[F(str(x)) for x in [a,b,lo,hi]];eps=F(3,10)
    endpoints=sorted({lo,hi}|{p+d for p in [a,b] for d in [-eps,eps] if lo<=p+d<=hi})
    probes=endpoints+[(x+y)/2 for x,y in zip(endpoints,endpoints[1:])]
    changes=[int(abs(b-t)<=eps)-int(abs(a-t)<=eps) for t in probes]
    distances=[abs(b-t)-abs(a-t) for t in [lo,hi]+[p for p in [a,b] if lo<=p<=hi]]
    return dict(accuracy_delta=[min(changes),max(changes)],mae_delta_seconds=[float(min(distances)),float(max(distances))])
def aggregate(grades):
    n=len(grades);c=Counter(g['result'] for g in grades)
    return dict(n=n,**{k:c[k] for k in ['correct','wrong','indeterminate']},accuracy_bounds=[c['correct']/n,(c['correct']+c['indeterminate'])/n],mae_bounds_seconds=[sum(g[k] for g in grades)/n for k in ['minimum_error_seconds','maximum_error_seconds']])
def raw_features(emb,times,state):
    norm=emb/np.maximum(np.linalg.norm(emb,axis=2,keepdims=True),1e-12)
    z=(norm-state['embedding_mean'])@state['components'].T;out=[]
    for a,t in zip(z,times):
        previous=np.concatenate([a[:1],a[:-1]]);following=np.concatenate([a[1:],a[-1:]])
        t=np.asarray(t);relative=(t-t[0])/max(t[-1]-t[0],1e-12)
        out.append(np.column_stack([a,a-previous,following-a,relative,np.arange(12)==0]))
    return np.stack(out)
def objective(x,pref,w):
    loss=float(w@w)/2;gradient=w.copy()
    for features,p in zip(x,pref):
        d=features[p[:,0]]-features[p[:,1]];margin=d@w
        loss+=float(np.logaddexp(0,-margin).mean())/len(pref)
        gradient-=d.T@np.exp(-np.logaddexp(0,margin))/(len(p)*len(pref))
    return loss,gradient
def verify():
    assert not (HERE/'independent_verification.json').exists()
    frozen=freeze_check();jobs=read(HERE/'inputs.json')['jobs'];train=read(HERE/'training_inputs.json')['jobs'];test=read(HERE/'evaluation_inputs.json')['jobs']
    assert jobs==train+test and len(train)==len(test)==8
    assert all(j['role']=='train_pool' for j in train) and all(j['role']=='evaluation' for j in test)
    feature=read(HERE/'features/report.json');log=read(HERE/'training_report.json');evaluation=read(HERE/'evaluation.json')
    assert feature['status']==log['status']==evaluation['status']=='complete'
    assert feature['text_generations']==feature['network_attempts']==0 and feature['vision_forwards']==17 and feature['repeat_first_exact']
    assert len(feature['rows'])==16 and len({r['ID'] for r in feature['rows']})==16
    assert frozen['created_utc']<=feature['started_utc']<=feature['ended_utc']<=log['started_utc']<=log['ended_utc']
    cpu={r['ID']:r for r in read(HERE/'cpu_input_review.json')['inputs']};saved={};pool_checks=[]
    for j,r in zip(jobs,feature['rows']):
        sid=j['ID'];assert r['ID']==sid and r['role']==j['role']
        history=read(ROOT/j['history_call'])[2]
        assert r['processor_hashes']==cpu[sid]['processor_hashes']==history['processor_input_sha256']
        assert sha(ROOT/j['image'])==j['image_sha256']==cpu[sid]['image_sha256']
        hidden=np.load(HERE/'features'/f'{sid}.hidden.npy',allow_pickle=False)
        pooled=np.load(HERE/'features'/f'{sid}.npy',allow_pickle=False)
        assert hidden.dtype==pooled.dtype==np.dtype('float32') and hidden.shape==(1152,2560) and pooled.shape==(12,2560)
        assert np.isfinite(hidden).all() and np.isfinite(pooled).all()
        assert digest(hidden)==r['hidden_sha256'] and digest(pooled)==r['pooled_sha256']
        grid=hidden.reshape(24,48,2560)
        reconstructed=np.stack([grid[y+1:y+8,x:x+12].mean(axis=(0,1)) for y in (0,8,16) for x in (0,12,24,36)])
        np.testing.assert_array_equal(reconstructed,pooled)
        pool_checks.append(dict(ID=sid,hidden_bytes=hidden.nbytes,pooled_bytes=pooled.nbytes,pool_exact=True,processor_exact=True));saved[sid]=pooled
    repeat=feature['repeat_record'];first=feature['rows'][0]
    for k in ['ID','role','processor_hashes','hidden_shape','hidden_sha256','pooled_sha256']:assert repeat[k]==first[k],k
    assert log['real_training_runs']==1 and log['evaluation_examples_used_by_fit']==0 and log['optimizer_success']
    references=read(HERE/'training_references.json');assert references['review_sha256']==sha(HERE/'training_reference_review.json')
    refs=[r for r in references['cases'] if r['eligible']];ids=[r['ID'] for r in refs]
    assert ids==log['training_ids']==['00000','00003','00006','00013']
    assert log['excluded_ids']==[r['ID'] for r in references['cases'] if not r['eligible']]
    assert not set(ids)&{j['ID'] for j in test}
    tj={j['ID']:j for j in train};times=[tj[s]['times'] for s in ids]
    pref=[pairs(t,*r['reference_seconds']) for t,r in zip(times,refs)]
    assert [len(p) for p in pref]==log['pairs_per_incident']==[61,66,66,65]
    for p,r in zip(pref,refs):assert p.tolist()==r['preference_pairs']
    assert log['head_parameters']==14 and log['pca_components']==4 and log['L2']==1 and log['incident_equal_weight'] and log['train_incidents']==4
    state=dict(np.load(HERE/'model.npz',allow_pickle=False));assert set(state)=={'embedding_mean','components','feature_mean','feature_scale','weights'}
    for k,shape in [('embedding_mean',(2560,)),('components',(4,2560)),('feature_mean',(14,)),('feature_scale',(14,)),('weights',(14,))]:assert state[k].shape==shape and np.isfinite(state[k]).all()
    assert sha(HERE/'model.npz')==log['model_sha256']==evaluation['model_sha256']
    emb=np.stack([saved[s] for s in ids]);norm=emb/np.maximum(np.linalg.norm(emb,axis=2,keepdims=True),1e-12)
    mean=norm.mean((0,1));close(mean,state['embedding_mean'],atol=1e-8)
    # This SVD verifies saved training-only projection; no optimizer is run.
    _,singular,vt=np.linalg.svd((norm-mean).reshape(-1,2560),full_matrices=False)
    close(state['components']@state['components'].T,np.eye(4),atol=1e-5)
    alignment=state['components']@vt[:4].T;close(alignment@alignment.T,np.eye(4),atol=1e-5)
    explained=float((singular[:4]**2).sum()/(singular**2).sum());close(explained,log['pca_explained_ratio'],atol=1e-6)
    raw=raw_features(emb,times,state);close(raw.mean((0,1)),state['feature_mean'],atol=1e-7)
    close(np.maximum(raw.std((0,1)),1e-6),state['feature_scale'],atol=1e-7)
    x=(raw-state['feature_mean'])/state['feature_scale'];loss,grad=objective(x,pref,state['weights'])
    close(log['loss_before'],np.log(2));close(log['loss_after'],loss);close(log['gradient_max'],float(abs(grad).max()),atol=1e-8)
    assert loss<np.log(2) and float(abs(grad).max())<=1e-5
    train_scores=x@state['weights'];selected={s:tj[s]['frames'][int(i)] for s,i in zip(ids,np.argmax(train_scores,axis=1))}
    assert selected==log['train_selected_frames']
    position_losses=[float(np.mean([max(abs(t[i]-r['reference_seconds'][0]),abs(t[i]-r['reference_seconds'][1])) for t,r in zip(times,refs)])) for i in range(12)]
    position=int(np.argmin(position_losses));assert position==log['constant_position_baseline']['index']
    close(position_losses,log['constant_position_baseline']['mean_worst_mae_by_index'])
    evemb=np.stack([saved[j['ID']] for j in test]);evraw=raw_features(evemb,[j['times'] for j in test],state)
    scores=((evraw-state['feature_mean'])/state['feature_scale'])@state['weights'];chosen=np.argmax(scores,axis=1)
    traces={r['ID']:r for r in read(OLD/'path_replay.json')['rows']};rows=[]
    assert len(evaluation['rows'])==8
    for j,score,index,record in zip(test,scores,chosen,evaluation['rows']):
        sid=j['ID'];assert record['ID']==sid;close(score,record['scores'],atol=1e-8)
        baseline=j['baseline'];candidate={**baseline,'entry_frame':j['frames'][int(index)]}
        assert baseline==record['baseline'] and candidate==record['candidate'] and record['reference']==j['evaluation_reference']
        ref=j['evaluation_reference'];pts=traces[sid]['original_times'];lo,hi=[pts[str(ref[k])] for k in ['lower_frame','upper_frame']]
        a,b=pts[str(baseline['entry_frame'])],pts[str(candidate['entry_frame'])]
        timing=dict(baseline=grade(a,lo,hi),candidate=grade(b,lo,hi),**delta(a,b,lo,hi),reference_seconds=[lo,hi]);compare(timing,record['timing'])
        falsefirst=ref['status']=='during_clip' and candidate['entry_frame']==j['frames'][0]
        assert record['new_false_first']==falsefirst and record['changed']==(baseline['entry_frame']!=candidate['entry_frame']) and record['other_three_unchanged']
        assert all(candidate[k]==baseline[k] for k in ['collision_frame','entry_side','evasion_space'])
        rows.append(dict(ID=sid,timing=timing,new_false_first=falsefirst,selected_index=int(index),entry_frame=candidate['entry_frame']))
    metrics={arm:aggregate([r['timing'][arm] for r in rows]) for arm in ['baseline','candidate']}
    changes={k:[sum(r['timing'][k][i] for r in rows)/8 for i in [0,1]] for k in ['accuracy_delta','mae_delta_seconds']}
    gate=dict(gain_ids=[r['ID'] for r in rows if r['timing']['baseline']['result']=='wrong' and r['timing']['candidate']['result']=='correct'],possible_loss_ids=[r['ID'] for r in rows if r['timing']['accuracy_delta'][0]<0],new_false_first_ids=[r['ID'] for r in rows if r['new_false_first']],other_three_preserved=True)
    gate['pass']=bool(gate['gain_ids']) and not gate['possible_loss_ids'] and not gate['new_false_first_ids'] and changes['mae_delta_seconds'][1]<=0
    compare(metrics,evaluation['metrics']);compare(changes,evaluation['paired_delta_bounds']);compare(gate,evaluation['gate'])
    shortcuts={}
    for name,index in [('always_first',0),('training_best_constant_position',position)]:
        details=[dict(ID=j['ID'],entry_frame=j['frames'][index],grade=grade(j['times'][index],*r['timing']['reference_seconds'])) for j,r in zip(test,rows)]
        shortcuts[name]=dict(index=index,rows=details,metrics=aggregate([d['grade'] for d in details]))
    compare(shortcuts,evaluation['shortcut_controls']);assert evaluation['official_S2'] is None
    assert freeze_check()==frozen
    return dict(status='PASS',created_utc=datetime.now(timezone.utc).isoformat(),verifier_sha256=sha(Path(__file__)),verifier_written_after_policy_freeze=True,freeze_files=len(frozen['files']),freeze_sha256=sha(HERE/'freeze.json'),model_sha256=sha(HERE/'model.npz'),original_2056_preserved=True,stored_hidden_pool_checks=pool_checks,processor_replay_scope='16 CPU replays and color-card before freeze; actual extraction hashes matched all16; no redundant processor run after freeze',vision_forwards_reported=17,text_generations_reported=0,repeat_first_hidden_and_pool_exact=True,training_ids=ids,train_incidents=4,pairs_per_incident=[len(p) for p in pref],total_pairs=258,head_parameters=14,train_only_statistics_and_projection_verified=True,training_loss_recomputed=loss,gradient_max_recomputed=float(abs(grad).max()),stationarity_linf_limit=1e-5,gradient_l2=float(np.linalg.norm(grad)),pca_explained_ratio=explained,evaluation_rows=rows,metrics=metrics,paired_delta_bounds=changes,gate=gate,shortcut_controls=shortcuts,resources=dict(feature_wall_seconds=feature['wall_seconds'],model_load_seconds=feature['model_load_seconds'],sum_primary16_forward_seconds=sum(r['seconds'] for r in feature['rows']),repeat_forward_seconds=repeat['seconds'],maximum_mlx_peak_GB=max([r['mlx_peak_GB'] for r in feature['rows']]+[repeat['mlx_peak_GB']]),training_wall_seconds=log['wall_seconds'],training_iterations=log['iterations'],hidden_cache_bytes=sum(r['hidden_bytes'] for r in pool_checks)),environment=dict(verification_python=sys.version,verification_numpy=np.__version__,feature=feature['environment'],training=log['environment']),additional_model_loads=0,additional_encoder_forwards=0,additional_text_generations=0,additional_optimizer_fits=0,limitations=['Four AI-filtered human drafts provide weak interval preferences, not four exact human entry ground truths.','Eight CCD cases are previously exposed development references; bounded split review does not establish unseen-source generalization.','Training projection and statistics use four eligible Nexar clips only; reference exposure still influenced the overall experimental design.','Vision tokens are globally contextualized; pooled regions do not verify collision-counterpart identity.','Three non-entry outputs are cached and preserved. No new full production inference, CUDA equivalence or official S2 score.'])
if __name__=='__main__':
    r=verify();(HERE/'independent_verification.json').write_text(json.dumps(r,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    text=f"# 독립 실행·수치 검증: PASS\n\n동결 {r['freeze_files']}파일 유지, 입력 16건의 실제 processor 해시와 저장 hidden→12타일 pool 완전 일치를 확인했다. 실제 기록은 frozen vision forward17회(16입력+첫 입력 재현1), 텍스트 생성0이다. 독립 검증은 추가 모델/encoder/optimizer 실행0이다.\n\n약한 학습4건·258순위쌍·PCA4·14특징에 대해 학습 전용 projection/통계, 손실 {r['training_loss_recomputed']:.10f}, gradient 최대 {r['gradient_max_recomputed']:.3g}를 재계산했다. 8건 원프레임 argmax, 다른3값 보존, 구간 평가와 고정 위치 대조군이 원평가와 일치한다.\n\n개발 gate: {r['gate']['pass']}. 정확도 변화 구간 {r['paired_delta_bounds']['accuracy_delta']}, MAE 변화 구간 {r['paired_delta_bounds']['mae_delta_seconds']}초.\n\n4건은 AI가 사람초안을 걸러 만든 약한 참조이고, CCD8은 이미 노출된 개발자료다. 사고당 258개 독립 표본으로 해석하지 않으며, 독립 사람검증·상대 식별력·CUDA·공식 점수 향상은 확인하지 않았다. 검증기는 부모 승인으로 정책 동결 뒤 작성되었고 별도 SHA를 기록했다.\n"
    (HERE/'independent_verification.md').write_text(text)
    print(json.dumps({k:r[k] for k in ['status','freeze_files','training_loss_recomputed','gradient_max_recomputed','metrics','paired_delta_bounds','gate','resources']},ensure_ascii=False))
