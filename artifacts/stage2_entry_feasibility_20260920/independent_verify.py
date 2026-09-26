"""Train4-only independent feasibility witness check; no solver/fit/model calls."""
import hashlib, itertools, json, sys
from fractions import Fraction as F
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
OLD=ROOT/'artifacts/stage2_entry_selector_20260920';IDS=['00000','00003','00006','00013']
read=lambda p:json.loads(p.read_text())
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def close(a,b,atol=1e-10):np.testing.assert_allclose(a,b,rtol=1e-10,atol=atol)
def same(a,b):
    if isinstance(a,dict):
        assert set(a)==set(b)
        for k in a:same(a[k],b[k])
    elif isinstance(a,list):
        assert len(a)==len(b)
        for x,y in zip(a,b):same(x,y)
    elif isinstance(a,float):close(a,b)
    else:assert a==b,(a,b)
def objective(x,pairs,w):
    loss=0.;g=w.copy();accuracies=[]
    for a,p in zip(x,pairs):
        differences=a[p[:,0]]-a[p[:,1]];v=differences@w
        loss+=np.logaddexp(0,-v).mean()/4
        g-=differences.T@np.exp(-np.logaddexp(0,v))/(4*len(p))
        accuracies.append(float(np.mean(v>0)))
    penalty=float(w@w)/2
    return dict(pairwise_logistic_loss=float(loss),l2_penalty=penalty,original_total_objective=float(loss+penalty),gradient_l2=float(np.linalg.norm(g)),strong_convexity_suboptimality_upper_bound=float(g@g)/2,pair_accuracy_by_incident=accuracies,mean_incident_pair_accuracy=float(np.mean(accuracies)))
def check_frozen(frozen):
    for name,h in frozen['sha256'].items():assert sha(ROOT/name)==h,name
def verify():
    frozen=read(HERE/'freeze.json');check_frozen(frozen);result=read(HERE/'result.json');protocol=read(HERE/'protocol.json')
    assert protocol['train_ids']==IDS
    jobs={j['ID']:j for j in read(OLD/'training_inputs.json')['jobs'] if j['ID'] in IDS}
    refs={r['ID']:r for r in read(OLD/'training_references.json')['cases'] if r['ID'] in IDS}
    assert len(jobs)==len(refs)==4 and all(refs[i]['eligible'] for i in IDS)
    state=dict(np.load(OLD/'model.npz',allow_pickle=False));emb=np.stack([np.load(OLD/'features'/f'{sid}.npy',allow_pickle=False) for sid in IDS])
    assert emb.shape==(4,12,2560) and state['weights'].shape==(14,)
    normalized=emb/np.maximum(np.linalg.norm(emb,axis=2,keepdims=True),1e-12)
    projected=(normalized-state['embedding_mean'])@state['components'].T
    raw=[]
    for z,sid in zip(projected,IDS):
        t=np.asarray(jobs[sid]['times']);previous=np.concatenate([z[:1],z[:-1]]);following=np.concatenate([z[1:],z[-1:]])
        raw.append(np.column_stack([z,z-previous,following-z,(t-t[0])/(t[-1]-t[0]),np.arange(12)==0]))
    x=(np.stack(raw)-state['feature_mean'])/state['feature_scale'];saved=np.load(HERE/'design.npy',allow_pickle=False)
    assert x.shape==(4,12,14) and x.dtype==saved.dtype==np.dtype('float64') and np.isfinite(x).all()
    np.testing.assert_array_equal(x,saved)
    scores=x@state['weights'];cases=[];all_pairs=[]
    for n,sid in enumerate(IDS):
        j,r=jobs[sid],refs[sid];t=[F(str(v)) for v in j['times']];lo,hi=[F(str(v)) for v in r['reference_seconds']]
        assert all(a<b for a,b in zip(t,t[1:]));cells=[]
        for k,a in enumerate(t):
            # Intersect every pairwise nearest-time halfspace with reference.
            lower,upper=lo,hi
            for q,b in enumerate(t):
                if q<k:lower=max(lower,(a+b)/2)
                elif q>k:upper=min(upper,(a+b)/2)
            if lower>upper:continue
            near=max(lo-a,a-hi,F(0));far=max(abs(lo-a),abs(hi-a))
            cells.append(dict(index=k,true_time_support=[float(lower),float(upper)],exact_support=[str(lower),str(upper)],frame=j['frames'][k],time=float(a),absolute_error_range=[float(near),float(far)],within_0_3_for_some_time=near<=F(3,10),within_0_3_for_all_times=far<=F(3,10)))
        p=np.array([(i,k) for i,a in enumerate(t) for k,b in enumerate(t) if i!=k and max(abs(a-lo)-abs(b-lo),abs(a-hi)-abs(b-hi))<-F(1,10**9)],dtype=int)
        assert p.tolist()==r['preference_pairs'];all_pairs.append(p)
        index=int(scores[n].argmax());cases.append(dict(ID=sid,reference_seconds=r['reference_seconds'],reference_frames=r['reference_frames'],frames=j['frames'],times=j['times'],possible_nearest=cells,old_index=index,old_frame=j['frames'][index]))
    same(cases,protocol['cases']);same(cases,result['cases']);assert [len(p) for p in all_pairs]==[61,66,66,65]
    combos=list(itertools.product(*[[c['index'] for c in r['possible_nearest']] for r in cases]))
    top_pair_rows=[]
    for sid,c,p in zip(IDS,cases,all_pairs):
        m={o['index'] for o in c['possible_nearest']}
        selected=[[int(i),int(j)] for i,j in p if i in m and j not in m]
        top_pair_rows.append(dict(ID=sid,possible_nearest_indices=sorted(m),original_pair_count=len(p),proposed_top_pair_count=len(selected),proposed_top_pairs=selected))
    assert [r['proposed_top_pair_count'] for r in top_pair_rows]==[11,11,11,20]
    assert combos==[(6,0,0,10),(6,0,0,11)] and len(result['runs'])==result['combination_count']==result['lp_solver_calls']==2
    original=objective(x,all_pairs,state['weights']);same(original,result['original_objective']);close(np.linalg.norm(state['weights']),result['original_weight_l2'])
    witnesses=[]
    for expected,row in zip(combos,result['runs']):
        assert row['winners']==list(expected) and row['strict'] and row['success'] and row['verified'] and row['status']==0
        w=np.array(row['weights'],dtype=np.float64);assert w.shape==(14,) and np.isfinite(w).all()
        sc=x@w;top=np.argmax(sc,axis=1);assert tuple(top)==expected==tuple(row['actual_indices'])
        assert row['actual_frames']==[jobs[sid]['frames'][int(i)] for sid,i in zip(IDS,top)]
        close(sc,row['scores'])
        comparisons=[(i,k,j) for i,k in enumerate(expected) for j in range(12) if j!=k]
        # Exact rational arithmetic over stored binary floating-point numbers.
        # This avoids assuming np.longdouble is wider than float64 on arm64.
        wf=[F.from_float(float(v)) for v in w];exact=[]
        for i,k,j in comparisons:
            exact.append(sum((F.from_float(float(x[i,k,d]))-F.from_float(float(x[i,j,d])))*wf[d] for d in range(14)))
        assert len(exact)==44 and min(exact)>0
        d=np.stack([x[i,k]-x[i,j] for i,k,j in comparisons]);margin=d@w
        close(float(min(exact)),row['min_score_margin']);close(float(min(exact)),row['min_extended_precision_margin'])
        close(float((margin-1).min()),row['min_constraint_slack']);assert float(min(exact))>=1-1e-7
        norm=float(np.linalg.norm(w));close(norm,row['weight_l2']);close(np.abs(w).sum(),row['weight_l1']);close(float(margin.min())/norm,row['min_unit_l2_weight_margin'])
        obj=objective(x,all_pairs,w);same(obj,row['original_objective'])
        assert obj['original_total_objective']>original['original_total_objective']
        witnesses.append(dict(winners=list(expected),frames=row['actual_frames'],comparisons=44,all_exact_binary_float_score_margins_positive=True,minimum_exact_rational_margin=str(min(exact)),minimum_exact_margin_float=float(min(exact)),minimum_unit_l2_weight_margin=float(min(exact))/norm,original_objective_recomputed=obj,strict_feasible=True))
    assert result['strict_verified_combinations']==2 and result['conclusion']=='FEASIBLE_STRICT_TRAIN_TOP1'
    assert result['encoder_forwards']==result['ccd_evaluations']==result['training_runs']==0 and not result['submission_changed'] and not result['witness_adopted']
    assert result['frozen_files_unchanged'];check_frozen(frozen)
    return dict(status='PASS',verifier_sha256=sha(Path(__file__)),diagnostic_result_sha256=sha(HERE/'result.json'),frozen_files=len(frozen['sha256']),frozen_files_preserved=True,train_ids=IDS,feature_shape=list(x.shape),features_reconstructed_exact=True,oracle_verified='All pairwise nearest-time halfspaces intersected using rational decimal PTS; not a timing-accuracy oracle.',oracle_cases=cases,possible_nearest_combinations=2,strict_witnesses_verified=2,witnesses=witnesses,original_objective_recomputed=original,proposed_top_pair_filter=dict(definition='Existing certain preference(i,j), with i in possible-nearest set M and j not in M.',rows=top_pair_rows,total_pairs=53,executed_training_runs=0,weighting='Keep within-incident mean then equal four-incident mean; lambda1 unchanged.',limitation='Only the subset definition and count are checked. Expected top1 or generalization gain is untested.'),strong_convexity_analysis='For the fixed lambda=1 objective, Hessian = I plus a positive-semidefinite logistic Hessian, so f(w)-f(w*) <= ||gradient||^2/2. Reported numeric bound uses floating-point gradient; it is not a rigorous interval-arithmetic certificate.',longdouble_bits=np.finfo(np.longdouble).bits,float64_bits=np.finfo(np.float64).bits,positive_margin_certificate='Exact Fraction arithmetic of stored float64 feature coordinates and witness weights; no reliance on extended-precision label.',l1_optimality_independently_certified=False,scope='Witness feasibility verified. Solver-reported minimum L1 optimality not independently certified without a dual certificate; irrelevant to positive existence result.',additional_lp_solver_calls=0,additional_training_runs=0,additional_pca_fits=0,additional_encoder_forwards=0,ccd_files_opened=0,excluded_feature_files_opened=0,submission_files_changed=0,python=sys.version,numpy=np.__version__,limits=['Only four training cases with weak AI-filtered human references; no independent validation or model promotion.','00000 and00013 have no candidate guaranteed within0.3sec for every instant in their full reference intervals.','00013 frame545 can be the nearest candidate while still always exceeding0.3sec error; frame599 is only possibly within0.3sec.','A feasible strict top1 witness disproves impossibility for these frozen features and allowed training targets; it does not identify the cause of generalization failure.','Original logistic-plus-L2 objective is near its own minimizer; the top1 witness solves a different optimization problem. Relative roles of pair loss and regularization are not separated.'])
if __name__=='__main__':
    out=HERE/'independent_verification.json';assert not out.exists();r=verify();out.write_text(json.dumps(r,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    (HERE/'independent_verification.md').write_text('# 독립 표현가능성 검증: PASS\n\n허용된 학습4건만 읽어 고정 PCA/정규화의 4×12×14 특징을 완전 재구성했다. 두 조합 [561,0,0,545], [561,0,0,599] 모두 각각 44개 비교에서 저장 float64 값에 대한 정확 유리수 마진이 양수이고 최솟값은 약1이다. 원래 동결11파일을 보존했으며 추가 LP·학습·PCA·encoder 호출0, CCD·제외특징 접근0이다.\n\n원래 목적함수는 '+str(r['original_objective_recomputed']['original_total_objective'])+', gradient L2는 '+str(r['original_objective_recomputed']['gradient_l2'])+'이다. λ=1 강볼록성에 따른 수치적 최적성 격차 상계 약9.82e-15와 두 witness의 원목적함수 12.8259/19.8541을 재계산했다. 이 상계는 부동소수점 gradient에 근거하며 엄밀한 구간산술 인증은 아니다. LP의 L1 최적성은 별도로 인증하지 않았고 양의마진 witness의 존재는 직접 확인했다.\n\n가능-최근접 후보는 ±0.3초 정답과 다르다. 00013의545는 일부 진짜 시각에서 가장 가까워도 전체 구간에서 오차가0.3초를 넘는다. 00000과00013은 전체 구간에 항상 맞는 후보가 없다. 따라서 이 결과는 네 약한 학습 사례의 지정 top1 표현가능성이며, 정확 정답4/4·일반화·점수 상승·새 가중치 채택을 뜻하지 않는다.\n\n기존 pairwise+L2 목적함수와 top1 목표의 차이는 확인되지만, pair loss와 정규화 중 어느 요소가 원인인지 분리하지 않았다. np.longdouble이 float64보다 넓다는 가정 없이 정확 유리수로 마진을 검증했다.\n')
    print(json.dumps({k:r[k] for k in ['status','frozen_files','features_reconstructed_exact','strict_witnesses_verified','original_objective_recomputed','longdouble_bits','float64_bits']},ensure_ascii=False))
