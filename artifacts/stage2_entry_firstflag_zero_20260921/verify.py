"""Independent equality-constrained head audit; no optimizer/model calls."""
import hashlib,importlib.util,json,sys
from fractions import Fraction as F
from collections import Counter
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
BASE=ROOT/'artifacts/stage2_entry_l2_01_20260921';SOURCE=ROOT/'artifacts/stage2_entry_selector_20260920'
IDS=['00000','00003','00006','00013'];read=lambda p:json.loads(p.read_text())
HELPER=ROOT/'artifacts/stage2_entry_top_pairs_20260921/verify.py'
spec=importlib.util.spec_from_file_location('prior_independent_math',HELPER);h=importlib.util.module_from_spec(spec);spec.loader.exec_module(h)
same,close,evaluate,paired=h.same,h.close,h.evaluate,h.paired
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def record(w,differences,constrained):
    penalty=float(.05*(w@w));data=0.;g=.1*w.copy();accuracies=[]
    for d in differences:
        margins=d@w;data+=float(np.logaddexp(0,-margins).mean())/4
        g-=d.T@np.exp(-np.logaddexp(0,margins))/(4*len(d))
        accuracies.append(float(np.mean(margins>0)))
    free=g[:13] if constrained else g
    return dict(total=data+penalty,data_loss=data,l2_penalty=penalty,weight_l2=float(np.linalg.norm(w)),first_indicator_weight=float(w[13]),full_gradient=g.tolist(),full_gradient_l2=float(np.linalg.norm(g)),stationarity_gradient_l2=float(np.linalg.norm(free)),stationarity_gradient_inf=float(np.max(abs(free))),stationarity_coordinates='0..12 free coordinates' if constrained else '0..13 unrestricted coordinates',numerical_gap_bound=float(free@free/.2),pair_accuracy_per_incident=accuracies)
def verify():
    frozen=read(HERE/'freeze.json');files=frozen['sha256']
    for name,digest in files.items():assert sha(ROOT/name)==digest,name
    output_sha={p:sha(p) for p in [HERE/'model.npz',HERE/'result.json',HERE/'fit.json',HELPER]}
    original=dict(np.load(SOURCE/'model.npz',allow_pickle=False));baseline=dict(np.load(BASE/'model.npz',allow_pickle=False));candidate=dict(np.load(HERE/'model.npz',allow_pickle=False))
    assert set(original)==set(baseline)==set(candidate)=={'embedding_mean','components','feature_mean','feature_scale','weights'}
    for k in candidate:
        assert candidate[k].dtype==baseline[k].dtype and candidate[k].shape==baseline[k].shape and np.isfinite(candidate[k]).all()
        if k!='weights':
            np.testing.assert_array_equal(candidate[k],baseline[k]);np.testing.assert_array_equal(candidate[k],original[k])
    assert candidate['weights'].shape==(14,) and candidate['weights'][13]==0.0
    jobs={j['ID']:j for j in read(SOURCE/'training_inputs.json')['jobs'] if j['ID'] in IDS};refs={r['ID']:r for r in read(SOURCE/'training_references.json')['cases'] if r['ID'] in IDS}
    assert len(jobs)==len(refs)==4 and all(refs[s]['eligible'] for s in IDS)
    embeddings=np.stack([np.load(SOURCE/'features'/f'{s}.npy',allow_pickle=False) for s in IDS])
    z=(embeddings/np.maximum(np.linalg.norm(embeddings,axis=2,keepdims=True),1e-12)-original['embedding_mean'])@original['components'].T;raw=[]
    for sid,a in zip(IDS,z):
        t=np.asarray(jobs[sid]['times']);previous=np.concatenate([a[:1],a[:-1]]);following=np.concatenate([a[1:],a[-1:]])
        raw.append(np.column_stack([a,a-previous,following-a,(t-t[0])/(t[-1]-t[0]),np.arange(12)==0]))
    x=(np.stack(raw)-original['feature_mean'])/original['feature_scale'];assert x.shape==(4,12,14)
    np.testing.assert_array_equal(x,np.load(ROOT/'artifacts/stage2_entry_feasibility_20260920/design.npy',allow_pickle=False))
    protocol=read(HERE/'protocol.json');prior_protocol=read(BASE/'protocol.json');prior=read(BASE/'result.json');result=read(HERE/'result.json');fit=read(HERE/'fit.json')
    assert protocol['train_ids']==IDS and protocol['L2']==.1 and protocol['fits_authorized']==1
    assert protocol['bounds']==[[None,None]]*13+[[0.,0.]] and protocol['initialization']=='zeros(14), no warm start'
    assert protocol['optimizer']=='L-BFGS-B' and protocol['jac'] and protocol['options']==prior_protocol['options']=={'maxiter':500,'ftol':1e-12,'gtol':1e-8,'maxls':50}
    same(protocol['cases'],prior_protocol['cases']);assert len(result['cases'])==4
    same(result['fit'],fit);assert fit['success'] and result['training_runs']==1 and result['ccd_evaluations']==result['encoder_forwards']==result['followup_runs']==0
    assert not result['submission_changed'] and not result['candidate_adopted']
    groups={'appearance_pca':range(4),'previous_difference':range(4,8),'following_difference':range(8,12),'relative_time':[12],'first_indicator':[13]}
    differences=[];records=[];deltas=[];score_decompositions=[]
    for n,sid in enumerate(IDS):
        j,r=jobs[sid],refs[sid];times=[F(str(t)) for t in j['times']];lo,hi=map(lambda t:F(str(t)),r['reference_seconds']);allowed=[]
        for i,a in enumerate(times):
            lower=max([lo]+[(a+b)/2 for k,b in enumerate(times) if k<i]);upper=min([hi]+[(a+b)/2 for k,b in enumerate(times) if k>i])
            if lower<=upper:allowed.append(i)
        pairs=[(i,k) for i,a in enumerate(times) for k,b in enumerate(times) if i!=k and max(abs(a-lo)-abs(b-lo),abs(a-hi)-abs(b-hi))<-F(1,10**9)]
        assert [list(p) for p in pairs]==r['preference_pairs'];p=np.array([(i,k) for i,k in pairs if i in allowed and k not in allowed])
        differences.append(x[n,p[:,0]]-x[n,p[:,1]])
        meta=dict(ID=sid,reference_seconds=r['reference_seconds'],reference_frames=r['reference_frames'],frames=j['frames'],times=j['times'],possible_nearest_indices=allowed,possible_nearest_frames=[j['frames'][i] for i in allowed],old_pair_count=len(pairs),top_pairs=p.tolist(),top_pair_count=len(p))
        same(meta,protocol['cases'][n]);given=result['cases'][n]
        for k,v in meta.items():same(v,given[k])
        old=evaluate(x[n],baseline['weights'],j,r,allowed,p);new=evaluate(x[n],candidate['weights'],j,r,allowed,p)
        same(old,given['old']);same(old,prior['cases'][n]['new']);same(new,given['new']);records.append(dict(ID=sid,old=old,new=new))
        delta=paired(old['selected_seconds'],new['selected_seconds'],*r['reference_seconds']);close(delta['absolute_error_delta_seconds'],given['paired_error_change_range']);assert delta['exact_error_delta_seconds']==given['paired_error_change_exact'];deltas.append(dict(ID=sid,**delta))
        score=x[n]@candidate['weights'];inside=max(allowed,key=lambda i:(score[i],-i));outside=max([i for i in range(12) if i not in allowed],key=lambda i:(score[i],-i))
        exact={name:sum((F.from_float(float(x[n,inside,k]))-F.from_float(float(x[n,outside,k])))*F.from_float(float(candidate['weights'][k])) for k in index) for name,index in groups.items()}
        derived={k:float(v) for k,v in exact.items()};derived['sum']=float(sum(exact.values()));same(derived,given['new_inside_minus_outside_decomposition']);close(derived['sum'],new['inside_minus_outside']);assert exact['first_indicator']==0
        score_decompositions.append(dict(ID=sid,best_inside_frame=j['frames'][inside],best_outside_frame=j['frames'][outside],terms=derived,exact_total=str(sum(exact.values()))))
    assert [len(d) for d in differences]==[11,11,11,20]
    initial=record(np.zeros(14),differences,True);final=record(candidate['weights'],differences,True);old_objective=record(baseline['weights'],differences,False)
    same(initial,fit['initial']);same(final,fit['final']);same(old_objective,result['old_unrestricted_objective'])
    assert final['stationarity_gradient_inf']<1e-5 and final['total']<initial['total'] and final['total']>=old_objective['total']-1e-10
    summary={}
    for version in ['old','new']:
        rows=[r[version] for r in records];counts=Counter(r['within_0_3_status'] for r in rows)
        summary[version]=dict(possible_nearest_top1=sum(r['in_possible_nearest_set'] for r in rows),within_0_3_counts={k:counts[k] for k in ['correct_all_reference_times','wrong_all_reference_times','indeterminate']},mean_error_lower=float(np.mean([r['absolute_error_seconds'][0] for r in rows])),mean_error_upper=float(np.mean([r['absolute_error_seconds'][1] for r in rows])))
    same(summary,result['summary']);same(summary['old'],prior['summary']['new'])
    mean=[sum(F(d['exact_error_delta_seconds'][i]) for d in deltas)/4 for i in [0,1]]
    close(list(map(float,mean)),result['paired_mean_error_change_range']);assert list(map(str,mean))==result['paired_mean_error_change_exact']
    acc=[sum(d['accuracy_delta'][i] for d in deltas)/4 for i in [0,1]]
    for name,digest in files.items():assert sha(ROOT/name)==digest,name
    for p,digest in output_sha.items():assert sha(p)==digest,p
    return dict(status='PASS',verifier_sha256=sha(Path(__file__)),independent_math_helper_sha256=sha(HELPER),frozen_files_preserved=len(files),baseline_model_sha256=sha(BASE/'model.npz'),candidate_model_sha256=output_sha[HERE/'model.npz'],original_model_sha256=sha(SOURCE/'model.npz'),features_reconstructed_exact=True,all_nonweight_arrays_exactly_unchanged=True,first_indicator_weight_exact_zero=True,stored_weights=14,free_coordinates=13,train_ids=IDS,top_pairs_per_case=[11,11,11,20],total_pairs=53,initial_objective=initial,constrained_objective=final,old_unrestricted_objective=old_objective,fixed_coordinate_gradient=final['full_gradient'][13],equality_constraint_multiplier=-final['full_gradient'][13],stationarity_interpretation='For equality w13=0, stationarity applies to coordinates0..12. The multiplier cancels coordinate13 gradient; its nonzero value is not a failure to converge. Numerical gap bound uses ||g_free||²/(2lambda), denominator0.2.',case_results=records,new_score_decompositions=score_decompositions,summary=summary,paired_same_T=dict(rows=deltas,mean_absolute_error_delta_seconds=list(map(float,mean)),exact_mean_absolute_error_delta_seconds=list(map(str,mean)),mean_accuracy_delta=acc,mean_error_definitely_worse=mean[0]>0),training_runs_reported=1,additional_optimizer_fits=0,additional_pca_fits=0,additional_encoder_forwards=0,ccd_files_opened=0,excluded_feature_files_opened=0,verification_python=sys.version,verification_numpy=np.__version__,limitations=['Train4 resubstitution using weak AI-filtered intervals is not independent performance evidence.','Only explicit first-indicator coefficient is fixed. Relative time, boundary differences and contextual vision/header information still encode position.','Constrained refit changes the remaining13 weights; this is not merely zeroing one coefficient at the previous optimum.','Score-group accounting is fixed-weight arithmetic; raw margins are not calibrated confidence or temporal errors.','Numerical stationarity and strong-convexity gap use floating point, not an interval certificate.','No new validation cohort, production adoption or official score.'])
if __name__=='__main__':
    out=HERE/'independent_verification.json';assert not out.exists();r=verify();out.write_text(json.dumps(r,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    text=['# 독립 w13=0 검증: PASS','','원래14특징·PCA·정규화·53쌍·λ=0.1을 유지하고, 마지막 계수만 정확히0에 고정했음을 확인했다. 자유13좌표의 gradient와 구간점수·선택·점수분해가 원기록과 일치한다.','','|사례|기존→고정계수 재학습 프레임|새 오차범위(초)|','|---|---:|---:|']
    for row in r['case_results']:text.append(f"|{row['ID']}|{row['old']['selected_frame']}→{row['new']['selected_frame']}|{row['new']['absolute_error_seconds']}|")
    text+=['',f"동일 미상 정답시각의 평균 절대오차 변화 {r['paired_same_T']['mean_absolute_error_delta_seconds']}초, 평균 정확도 변화 {r['paired_same_T']['mean_accuracy_delta']}.",'',f"free13 gradient L2={r['constrained_objective']['stationarity_gradient_l2']:.12g}, 고정좌표 gradient={r['fixed_coordinate_gradient']:.12g}. 고정좌표의 비영 gradient는 등식제약 승수로 상쇄되며 수렴 실패가 아니다. 최적성의 수치적 격차 상계는 free13 norm²/0.2로 계산했다.",'',f"동결 {r['frozen_files_preserved']}파일·원모델 보존, 추가 optimizer·PCA·encoder·CCD/제외특징 접근0.",'','이번 변화는 계수0 제약을 둔 재학습이며 나머지13계수도 변한다. 명시적 firstflag를 제거해도 다른 특징의 위치정보는 남는다. 약한 학습4건의 재대입 결과를 일반화·제출 성능으로 승격하지 않는다.']
    (HERE/'independent_verification.md').write_text('\n'.join(text)+'\n')
    print(json.dumps({k:r[k] for k in ['status','frozen_files_preserved','fixed_coordinate_gradient','constrained_objective','summary','paired_same_T']},ensure_ascii=False))
