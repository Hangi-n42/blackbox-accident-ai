"""Independent saved-weight verification, train4 only. No optimizer/encoder calls."""
import hashlib,importlib.util,json,sys
from fractions import Fraction as F
from pathlib import Path
from collections import Counter
import numpy as np
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
BASE=ROOT/'artifacts/stage2_entry_top_pairs_20260921';ORIGINAL=ROOT/'artifacts/stage2_entry_selector_20260920'
IDS=['00000','00003','00006','00013'];read=lambda p:json.loads(p.read_text())
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
spec=importlib.util.spec_from_file_location('prior_independent_verifier',BASE/'verify.py');helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
same,close,evaluate,paired=helper.same,helper.close,helper.evaluate,helper.paired
def objective(w,differences,l2):
    penalty=float(l2*(w@w)/2);data=0.;gradient=l2*w.copy();accuracies=[]
    for d in differences:
        margins=d@w;data+=float(np.logaddexp(0,-margins).mean())/4
        gradient-=d.T@np.exp(-np.logaddexp(0,margins))/(len(d)*4)
        accuracies.append(float(np.mean(margins>0)))
    return {'lambda':l2,'total':data+penalty,'l2_penalty':penalty,'data_loss':data,'weight_l2':float(np.linalg.norm(w)),
            'gradient_l2':float(np.linalg.norm(gradient)),'gradient_inf':float(np.max(abs(gradient))),
            'numerical_strong_convexity_gap_bound':float(gradient@gradient/(2*l2)),'pair_accuracy_per_incident':accuracies}
def verify():
    frozen=read(HERE/'freeze.json');files=frozen['sha256']
    for name,h in files.items():assert sha(ROOT/name)==h,name
    snapshots={p:sha(p) for p in [HERE/'model.npz',HERE/'result.json',HERE/'fit.json',BASE/'verify.py']}
    baseline=dict(np.load(BASE/'model.npz',allow_pickle=False));new=dict(np.load(HERE/'model.npz',allow_pickle=False));original=dict(np.load(ORIGINAL/'model.npz',allow_pickle=False))
    assert set(baseline)==set(new)==set(original)=={'embedding_mean','components','feature_mean','feature_scale','weights'}
    for k in baseline:
        assert baseline[k].shape==new[k].shape and baseline[k].dtype==new[k].dtype and np.isfinite(new[k]).all()
        if k!='weights':
            np.testing.assert_array_equal(baseline[k],new[k]);np.testing.assert_array_equal(original[k],new[k])
    assert new['weights'].shape==(14,)
    jobs={j['ID']:j for j in read(ORIGINAL/'training_inputs.json')['jobs'] if j['ID'] in IDS}
    refs={r['ID']:r for r in read(ORIGINAL/'training_references.json')['cases'] if r['ID'] in IDS}
    assert len(jobs)==len(refs)==4 and all(refs[s]['eligible'] for s in IDS)
    embedding=np.stack([np.load(ORIGINAL/'features'/f'{s}.npy',allow_pickle=False) for s in IDS])
    z=(embedding/np.maximum(np.linalg.norm(embedding,axis=2,keepdims=True),1e-12)-original['embedding_mean'])@original['components'].T
    raw=[]
    for s,a in zip(IDS,z):
        t=np.array(jobs[s]['times']);previous=np.concatenate([a[:1],a[:-1]]);following=np.concatenate([a[1:],a[-1:]])
        raw.append(np.column_stack([a,a-previous,following-a,(t-t[0])/(t[-1]-t[0]),np.arange(12)==0]))
    x=(np.stack(raw)-original['feature_mean'])/original['feature_scale']
    np.testing.assert_array_equal(x,np.load(ROOT/'artifacts/stage2_entry_feasibility_20260920/design.npy',allow_pickle=False));assert x.shape==(4,12,14)
    result=read(HERE/'result.json');protocol=read(HERE/'protocol.json');prior_protocol=read(BASE/'protocol.json');prior_result=read(BASE/'result.json');fit=read(HERE/'fit.json')
    assert protocol['L2']==.1 and protocol['baseline_L2']==1 and protocol['fits_authorized']==1 and protocol['train_ids']==IDS
    assert protocol['options']==prior_protocol['options']=={'maxiter':500,'ftol':1e-12,'gtol':1e-8,'maxls':50}
    assert protocol['optimizer']=='L-BFGS-B' and protocol['jac'] and protocol['initialization']=='zeros(14); no warm start'
    same(protocol['cases'],prior_protocol['cases']);same(fit,result['fit']);assert fit['success']
    assert len(result['cases'])==4 and result['training_runs']==1 and result['additional_lambda_trials']==result['ccd_evaluations']==result['encoder_forwards']==0
    assert not result['candidate_adopted'] and not result['submission_changed']
    diffs=[];records=[];paired_rows=[];terms_checked=[]
    slices={'appearance_pca':range(4),'previous_difference':range(4,8),'following_difference':range(8,12),'relative_time':[12],'first_indicator':[13]}
    for n,sid in enumerate(IDS):
        j,r=jobs[sid],refs[sid];ts=[F(str(t)) for t in j['times']];lo,hi=map(lambda t:F(str(t)),r['reference_seconds']);m=[]
        for i,a in enumerate(ts):
            lower=max([lo]+[(a+b)/2 for k,b in enumerate(ts) if k<i]);upper=min([hi]+[(a+b)/2 for k,b in enumerate(ts) if k>i])
            if lower<=upper:m.append(i)
        all_pairs=[(i,k) for i,a in enumerate(ts) for k,b in enumerate(ts) if i!=k and max(abs(a-lo)-abs(b-lo),abs(a-hi)-abs(b-hi))<-F(1,10**9)]
        assert [list(p) for p in all_pairs]==r['preference_pairs'];p=np.array([(i,k) for i,k in all_pairs if i in m and k not in m])
        diffs.append(x[n,p[:,0]]-x[n,p[:,1]])
        metadata=dict(ID=sid,reference_seconds=r['reference_seconds'],reference_frames=r['reference_frames'],frames=j['frames'],times=j['times'],possible_nearest_indices=m,possible_nearest_frames=[j['frames'][i] for i in m],old_pair_count=len(all_pairs),top_pairs=p.tolist(),top_pair_count=len(p))
        same(metadata,protocol['cases'][n]);saved=result['cases'][n]
        for k,v in metadata.items():same(v,saved[k])
        oldrow=evaluate(x[n],baseline['weights'],j,r,m,p);newrow=evaluate(x[n],new['weights'],j,r,m,p)
        same(oldrow,saved['old']);same(oldrow,prior_result['cases'][n]['new']);same(newrow,saved['new'])
        d=paired(oldrow['selected_seconds'],newrow['selected_seconds'],*r['reference_seconds'])
        close(d['absolute_error_delta_seconds'],saved['paired_error_change_range']);assert d['exact_error_delta_seconds']==saved['paired_error_change_exact']
        paired_rows.append(dict(ID=sid,**d));group_record={'ID':sid}
        for label,w,ev in [('old',baseline['weights'],oldrow),('new',new['weights'],newrow)]:
            score=x[n]@w;inside=max(m,key=lambda i:(score[i],-i));outside=max([i for i in range(12) if i not in m],key=lambda i:(score[i],-i))
            terms={name:sum((F.from_float(float(x[n,inside,k]))-F.from_float(float(x[n,outside,k])))*F.from_float(float(w[k])) for k in ix) for name,ix in slices.items()}
            total=sum(terms.values());expected={k:float(v) for k,v in terms.items()};expected['sum']=float(total)
            same(expected,saved['inside_minus_outside_score_decomposition'][label]);close(float(total),ev['inside_minus_outside'])
            group_record[label]=dict(best_inside_frame=j['frames'][inside],best_outside_frame=j['frames'][outside],groups=expected,exact_total=str(total),unit_weight_l2_margin=float(total)/float(np.linalg.norm(w)))
        terms_checked.append(group_record);records.append(dict(ID=sid,old=oldrow,new=newrow))
    assert [len(d) for d in diffs]==[11,11,11,20]
    objectives={f'{label}_weights_lambda_{lam}':objective(w,diffs,lam) for label,w in [('old',baseline['weights']),('new',new['weights'])] for lam in [1.,.1]}
    same(objectives,result['objectives']);same(objective(np.zeros(14),diffs,.1),fit['initial']);same(objectives['new_weights_lambda_0.1'],fit['final'])
    assert fit['final']['total']<fit['initial']['total'] and objectives['new_weights_lambda_0.1']['gradient_inf']<1e-5
    summary={}
    for version in ['old','new']:
        rows=[r[version] for r in records];count=Counter(r['within_0_3_status'] for r in rows)
        summary[version]=dict(possible_nearest_top1=sum(r['in_possible_nearest_set'] for r in rows),within_0_3_counts={k:count[k] for k in ['correct_all_reference_times','wrong_all_reference_times','indeterminate']},mean_error_lower=float(np.mean([r['absolute_error_seconds'][0] for r in rows])),mean_error_upper=float(np.mean([r['absolute_error_seconds'][1] for r in rows])))
    same(summary,result['summary']);same(summary['old'],prior_result['summary']['new'])
    delta=[sum(F(r['exact_error_delta_seconds'][i]) for r in paired_rows)/4 for i in [0,1]]
    close(list(map(float,delta)),result['paired_mean_error_change_range']);assert list(map(str,delta))==result['paired_mean_error_change_exact']
    acc=[sum(r['accuracy_delta'][i] for r in paired_rows)/4 for i in [0,1]]
    for name,h in files.items():assert sha(ROOT/name)==h,name
    for path,h in snapshots.items():assert sha(path)==h,path
    return dict(status='PASS',verifier_sha256=sha(Path(__file__)),independent_math_helper_sha256=sha(BASE/'verify.py'),frozen_files_preserved=len(files),model_sha256=snapshots[HERE/'model.npz'],baseline_model_sha256=sha(BASE/'model.npz'),baseline_is_previous_53_pair_lambda1=True,original_model_sha256=sha(ORIGINAL/'model.npz'),original_model_unchanged=True,all_nonweight_arrays_exactly_unchanged=True,features_reconstructed_exact=True,train_ids=IDS,top_pairs_per_case=[11,11,11,20],total_pairs=53,objectives=objectives,case_results=records,score_decomposition=terms_checked,summary=summary,paired_same_T=dict(rows=paired_rows,mean_absolute_error_delta_seconds=list(map(float,delta)),exact_mean_absolute_error_delta_seconds=list(map(str,delta)),mean_accuracy_delta=acc),training_runs_reported=1,additional_optimizer_fits=0,additional_lambda_trials=0,additional_pca_fits=0,additional_encoder_forwards=0,ccd_files_opened=0,excluded_feature_files_opened=0,verification_python=sys.version,verification_numpy=np.__version__,limitations=['Train4 resubstitution on weak AI-filtered reference intervals; no independent validation.','Possible-nearest membership is not0.3sec correctness; 00013 frame545 remains outside tolerance for every reference time.','A raw score margin has no common calibration across different weight scales. Unit-L2 margins are arithmetic direction summaries, not probability or error estimates.','Score decomposition is fixed-weight arithmetic, not feature ablation or independent causal evidence.','Strong-convexity gap uses floating-point gradient and denominator2lambda=0.2, not a rigorous interval certificate.','Lambda0.1 and prior lambda1 are successive exposed training experiments; this verification does not select another lambda or claim generalization.'])
if __name__=='__main__':
    out=HERE/'independent_verification.json';assert not out.exists();r=verify();out.write_text(json.dumps(r,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    text=['# 독립 λ=0.1 검증: PASS','','직전53쌍 λ=1 모델을 기준으로, 학습4건의 고정 특징·P/M/P_top53·PCA·정규화·참조구간 보존을 확인했다. 모델의 다른 배열은 원본과 완전 일치하며 새 가중치의 목적함수·gradient·집합 안팎 마진·±0.3 판정과 점수 분해가 원기록과 일치한다.','','|사례|λ1→λ0.1 선택 프레임|새 오차 범위(초)|새 허용집합 포함|','|---|---:|---:|---:|']
    for row in r['case_results']:text.append(f"|{row['ID']}|{row['old']['selected_frame']}→{row['new']['selected_frame']}|{row['new']['absolute_error_seconds']}|{row['new']['in_possible_nearest_set']}|")
    text+=['',f"같은 미상 정답시각 기준 평균 절대오차 변화 {r['paired_same_T']['mean_absolute_error_delta_seconds']}초, 평균 정확도 변화 {r['paired_same_T']['mean_accuracy_delta']}. 가능한 최근접 집합 포함은 정확한 진입시각 정답과 다르다.",'',f"동결 {r['frozen_files_preserved']}파일과 기존 모델을 보존했다. 추가 optimizer·PCA·encoder 실행0, CCD·제외 특징 접근0이다.",'','λ가 달라지면 가중치 규모와 raw score margin도 달라진다. 이를 직접 시간오차·확신도 변화로 해석하지 않았다. 점수항 분해는 고정 결과의 산술이며 feature 제거 실험이 아니다. 약한 학습4건의 재대입 개선은 독립 성능검증이나 제출 채택 근거가 아니다.']
    (HERE/'independent_verification.md').write_text('\n'.join(text)+'\n')
    print(json.dumps({k:r[k] for k in ['status','frozen_files_preserved','summary','paired_same_T','objectives']},ensure_ascii=False))
