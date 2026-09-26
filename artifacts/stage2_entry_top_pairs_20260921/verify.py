"""Independent train4-only saved-weight audit. No fitting, PCA, solver or model calls."""
import hashlib, json, sys
from collections import Counter
from fractions import Fraction as F
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
OLD=ROOT/'artifacts/stage2_entry_selector_20260920';IDS=['00000','00003','00006','00013']
read=lambda p:json.loads(p.read_text())
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def close(a,b):np.testing.assert_allclose(a,b,atol=1e-10,rtol=1e-9)
def same(a,b):
    if isinstance(a,dict):
        assert set(a)==set(b),(set(a),set(b))
        for k in a:same(a[k],b[k])
    elif isinstance(a,list):
        assert len(a)==len(b)
        for x,y in zip(a,b):same(x,y)
    elif isinstance(a,float):close(a,b)
    else:assert a==b,(a,b)
def loss(w,groups):
    penalty=float(w@w)/2;data=0.;gradient=w.copy();accuracies=[]
    for d in groups:
        scores=d@w;data+=float(np.logaddexp(0,-scores).mean())/4
        gradient-=d.T@np.exp(-np.logaddexp(0,scores))/(len(d)*4)
        accuracies.append(float(np.mean(scores>0)))
    return dict(total=data+penalty,l2_penalty=penalty,data_loss=data,gradient_l2=float(np.linalg.norm(gradient)),gradient_inf=float(np.max(abs(gradient))),numerical_strong_convexity_gap_bound=float(gradient@gradient)/2,pair_accuracy_per_incident=accuracies)
def evaluate(x,w,j,r,m,p):
    sc=x@w;k=int(np.argmax(sc));outside=[i for i in range(12) if i not in m]
    inside=max(m,key=lambda i:(sc[i],-i));out=max(outside,key=lambda i:(sc[i],-i))
    t=F(str(j['times'][k]));lo,hi=map(lambda v:F(str(v)),r['reference_seconds'])
    minimum=max(lo-t,t-hi,F(0));maximum=max(abs(t-lo),abs(t-hi));bestworst=min(max(abs(F(str(v))-lo),abs(F(str(v))-hi)) for v in j['times'])
    margins=(x[p[:,0]]-x[p[:,1]])@w
    return dict(selected_index=k,selected_frame=j['frames'][k],selected_seconds=float(t),in_possible_nearest_set=k in m,scores=sc.tolist(),best_inside_frame=j['frames'][inside],best_outside_frame=j['frames'][out],max_inside_score=float(sc[inside]),max_outside_score=float(sc[out]),inside_minus_outside=float(sc[inside]-sc[out]),absolute_error_seconds=[float(minimum),float(maximum)],within_0_3_status='correct_all_reference_times' if maximum<=F(3,10) else 'wrong_all_reference_times' if minimum>F(3,10) else 'indeterminate',minimax_regret_seconds=float(maximum-bestworst),top_pairs_positive=int(np.sum(margins>0)),top_pairs_nonpositive=int(np.sum(margins<=0)))
def paired(old,new,lo,hi):
    old,new,lo,hi=map(lambda v:F(str(v)),[old,new,lo,hi]);tol=F(3,10)
    points=[lo,hi]+[p for p in [old,new] if lo<=p<=hi]
    errors=[abs(new-t)-abs(old-t) for t in points]
    cuts=sorted({lo,hi}|{p+d for p in [old,new] for d in [-tol,tol] if lo<=p+d<=hi})
    checks=cuts+[(a+b)/2 for a,b in zip(cuts,cuts[1:])]
    accuracies=[int(abs(new-t)<=tol)-int(abs(old-t)<=tol) for t in checks]
    return dict(absolute_error_delta_seconds=[float(min(errors)),float(max(errors))],exact_error_delta_seconds=[str(min(errors)),str(max(errors))],accuracy_delta=[min(accuracies),max(accuracies)])
def verify():
    frozen=read(HERE/'freeze.json');snapshots={HERE/'model.npz':sha(HERE/'model.npz'),HERE/'result.json':sha(HERE/'result.json'),HERE/'fit.json':sha(HERE/'fit.json')}
    for name,h in frozen['sha256'].items():assert sha(ROOT/name)==h,name
    old=dict(np.load(OLD/'model.npz',allow_pickle=False));new=dict(np.load(HERE/'model.npz',allow_pickle=False))
    assert set(old)==set(new)=={'embedding_mean','components','feature_mean','feature_scale','weights'}
    assert sha(OLD/'model.npz')=='13122d0b662d254d02fdcd6f7f9ad816e848d9155c3ea7dfb73c1e0504e6afde'
    for k in old:
        assert new[k].dtype==old[k].dtype and new[k].shape==old[k].shape and np.isfinite(new[k]).all()
        if k!='weights':np.testing.assert_array_equal(old[k],new[k])
    assert old['weights'].shape==new['weights'].shape==(14,)
    jobs={j['ID']:j for j in read(OLD/'training_inputs.json')['jobs'] if j['ID'] in IDS};refs={r['ID']:r for r in read(OLD/'training_references.json')['cases'] if r['ID'] in IDS}
    assert set(jobs)==set(refs)==set(IDS) and all(refs[s]['eligible'] for s in IDS)
    emb=np.stack([np.load(OLD/'features'/f'{sid}.npy',allow_pickle=False) for sid in IDS])
    z=(emb/np.maximum(np.linalg.norm(emb,axis=2,keepdims=True),1e-12)-old['embedding_mean'])@old['components'].T;raw=[]
    for sid,a in zip(IDS,z):
        t=np.asarray(jobs[sid]['times']);previous=np.concatenate([a[:1],a[:-1]]);following=np.concatenate([a[1:],a[-1:]])
        raw.append(np.column_stack([a,a-previous,following-a,(t-t[0])/(t[-1]-t[0]),np.arange(12)==0]))
    x=(np.stack(raw)-old['feature_mean'])/old['feature_scale'];assert x.shape==(4,12,14)
    np.testing.assert_array_equal(x,np.load(ROOT/'artifacts/stage2_entry_feasibility_20260920/design.npy',allow_pickle=False))
    protocol=read(HERE/'protocol.json');result=read(HERE/'result.json');fit=read(HERE/'fit.json')
    assert protocol['train_ids']==IDS and protocol['L2']==1 and protocol['fits_authorized']==1 and protocol['optimizer']=='L-BFGS-B'
    assert protocol['options']=={'maxiter':500,'ftol':1e-12,'gtol':1e-8,'maxls':50}
    assert protocol['initialization']=='zeros(14), not LP witness or old weights'
    assert result['training_runs']==1 and result['ccd_evaluations']==result['encoder_forwards']==result['followup_runs']==0
    assert not result['submission_changed'] and not result['candidate_adopted']
    assert len(result['cases'])==len(protocol['cases'])==4 and fit['success']
    diffs=[];all_diffs=[];records=[];deltas=[]
    for n,sid in enumerate(IDS):
        j,r=jobs[sid],refs[sid];t=[F(str(v)) for v in j['times']];lo,hi=map(lambda v:F(str(v)),r['reference_seconds']);m=[]
        assert all(a<b for a,b in zip(t,t[1:]))
        for i,a in enumerate(t):
            lower=max([lo]+[(a+b)/2 for k,b in enumerate(t) if k<i]);upper=min([hi]+[(a+b)/2 for k,b in enumerate(t) if k>i])
            if lower<=upper:m.append(i)
        pairs=[(i,k) for i,a in enumerate(t) for k,b in enumerate(t) if i!=k and max(abs(a-lo)-abs(b-lo),abs(a-hi)-abs(b-hi))<-F(1,10**9)]
        assert [list(p) for p in pairs]==r['preference_pairs']
        top=np.array([(i,k) for i,k in pairs if i in m and k not in m]);p=np.array(pairs)
        all_diffs.append(x[n,p[:,0]]-x[n,p[:,1]]);diffs.append(x[n,top[:,0]]-x[n,top[:,1]])
        metadata=dict(ID=sid,reference_seconds=r['reference_seconds'],reference_frames=r['reference_frames'],frames=j['frames'],times=j['times'],possible_nearest_indices=m,possible_nearest_frames=[j['frames'][i] for i in m],old_pair_count=len(p),top_pairs=top.tolist(),top_pair_count=len(top))
        same(metadata,protocol['cases'][n])
        rec={**metadata,'old':evaluate(x[n],old['weights'],j,r,m,top),'new':evaluate(x[n],new['weights'],j,r,m,top)}
        same(rec,result['cases'][n]);records.append(rec)
        deltas.append(dict(ID=sid,**paired(rec['old']['selected_seconds'],rec['new']['selected_seconds'],*r['reference_seconds'])))
    assert [len(d) for d in diffs]==[11,11,11,20] and sum(map(len,diffs))==53
    recomputed_fit={'initial_new_objective':loss(np.zeros(14),diffs),'final_new_objective':loss(new['weights'],diffs)}
    for k,v in recomputed_fit.items():same(v,fit[k])
    same(fit,result['fit'])
    assert recomputed_fit['final_new_objective']['total']<recomputed_fit['initial_new_objective']['total'] and recomputed_fit['final_new_objective']['gradient_inf']<1e-5
    cross={name:loss(w,d) for name,w,d in [('old_weights_on_new_objective',old['weights'],diffs),('old_weights_on_old_objective',old['weights'],all_diffs),('new_weights_on_old_objective',new['weights'],all_diffs)]}
    for k,v in cross.items():same(v,result[k])
    summary={}
    for version in ['old','new']:
        rows=[r[version] for r in records];c=Counter(r['within_0_3_status'] for r in rows)
        summary[version]=dict(possible_nearest_top1=sum(r['in_possible_nearest_set'] for r in rows),within_0_3_counts={k:c[k] for k in ['correct_all_reference_times','wrong_all_reference_times','indeterminate']},mean_error_lower=float(np.mean([r['absolute_error_seconds'][0] for r in rows])),mean_error_upper=float(np.mean([r['absolute_error_seconds'][1] for r in rows])))
    same(summary,result['summary'])
    shared_error=[sum(F(d['exact_error_delta_seconds'][i]) for d in deltas)/4 for i in [0,1]]
    shared_accuracy=[sum(d['accuracy_delta'][i] for d in deltas)/4 for i in [0,1]]
    for name,h in frozen['sha256'].items():assert sha(ROOT/name)==h,name
    for p,h in snapshots.items():assert sha(p)==h,p
    return dict(status='PASS',verifier_sha256=sha(Path(__file__)),frozen_files_preserved=len(frozen['sha256']),result_sha256=snapshots[HERE/'result.json'],candidate_model_sha256=snapshots[HERE/'model.npz'],old_model_sha256=sha(OLD/'model.npz'),old_model_unchanged=True,pca_normalization_exactly_unchanged=True,feature_reconstruction_exact=True,train_ids=IDS,train_cases=4,top_pairs_per_case=[11,11,11,20],total_top_pairs=53,objective_and_gradient=recomputed_fit,objective_cross_comparison=cross,case_results=records,summary=summary,paired_same_unknown_time=dict(rows=deltas,mean_absolute_error_delta_seconds=[float(v) for v in shared_error],exact_mean_absolute_error_delta_seconds=list(map(str,shared_error)),mean_accuracy_delta=shared_accuracy,mean_absolute_error_definitely_worsened=shared_error[0]>0,scope='Each case shares the same unknown reference time between old/new; different cases may have different true times. Extrema of the separable mean are attained independently.'),training_runs_reported=1,additional_optimizer_fits=0,additional_lp_calls=0,additional_pca_fits=0,additional_encoder_forwards=0,ccd_files_opened=0,excluded_feature_files_opened=0,existing_source_or_models_modified=False,verification_python=sys.version,verification_numpy=np.__version__,limits=['Only train4 weak AI-filtered references; training resubstitution is not independent performance evidence.','Possible-nearest membership is distinct from inclusive0.3sec correctness; frame545 can be in M and always wrong.','Two candidate models differ only in linear weights, but pair membership changes both selected relations and their relative weight after within-case normalization.','Loss reduction on53pairs is not directly comparable to the prior258pair loss; cross-objective values are explicitly separated.','Numerical strong-convexity gap uses floating-point gradients, not rigorous interval arithmetic.','No additional validation, production adoption or official S2 score is claimed.'])
if __name__=='__main__':
    out=HERE/'independent_verification.json';assert not out.exists();r=verify();out.write_text(json.dumps(r,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    rows=r['case_results'];lines=['# 독립 train4 검증: PASS','','고정 PCA·정규화·4×12×14 특징과 P_top 11/11/11/20(53쌍)을 독립 재구성했다. 원래 모델과 동결 파일을 보존했고 새 모델의 비가중치 배열도 완전 일치했다. 추가 optimizer·LP·PCA·encoder 실행0, CCD·제외 특징 접근0이다.','','|사례|기존→새 프레임|허용 집합 마진 기존→새|새 오차 범위(초)|','|---|---:|---:|---:|']
    for row in rows:
        a,b=row['old'],row['new'];lines.append(f"|{row['ID']}|{a['selected_frame']}→{b['selected_frame']}|{a['inside_minus_outside']:.6f}→{b['inside_minus_outside']:.6f}|{b['absolute_error_seconds']}|")
    lines+=['',f"동일 미상 정답시각을 공유한 평균 절대오차 변화: {r['paired_same_unknown_time']['mean_absolute_error_delta_seconds']}초. 평균 정확도 변화: {r['paired_same_unknown_time']['mean_accuracy_delta']}. 평균 절대오차의 확정 악화 여부: {r['paired_same_unknown_time']['mean_absolute_error_definitely_worsened']}.",'','53쌍 목적함수와 gradient, top1·집합 안팎 max 점수, 전체 참조구간 오차·±0.3 판정이 원기록과 일치한다. 학습4건 재대입 결과이며 독립 성능검증·일반화·제출 채택 근거로 승격하지 않는다. 가능한 최근접 후보 집합은 정확 정답 집합이 아니다.']
    (HERE/'independent_verification.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({k:r[k] for k in ['status','frozen_files_preserved','objective_and_gradient','summary','paired_same_unknown_time']},ensure_ascii=False))
