"""Independent train4 aggregation audit, without fitting or model forwards."""
import ast,hashlib,importlib.util,json,sys
from collections import Counter
from fractions import Fraction as F
from pathlib import Path
import numpy as np
from scipy.special import logsumexp
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
BASE=ROOT/'artifacts/stage2_entry_l2_01_20260921';SOURCE=ROOT/'artifacts/stage2_entry_selector_20260920'
HELPER=ROOT/'artifacts/stage2_entry_top_pairs_20260921/verify.py'
IDS=['00000','00003','00006','00013'];read=lambda p:json.loads(p.read_text())
spec=importlib.util.spec_from_file_location('independent_interval_math',HELPER);h=importlib.util.module_from_spec(spec);spec.loader.exec_module(h)
same,close,evaluate,paired=h.same,h.close,h.evaluate,h.paired
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def calculate(w,diffs,kind):
    penalty=float(.05*(w@w));data=0.;g=.1*w.copy();parts=[]
    for d in diffs:
        margin=d@w;n=len(d)
        if kind=='hard':
            logits=np.concatenate([np.array([0.]),-margin-np.log(n)])
            li=float(np.logaddexp.reduce(logits));coefficient=np.exp(logits[1:]-li)/4
        else:
            li=float(np.logaddexp(0,-margin).mean());coefficient=(1-np.tanh(margin/2))/(8*n)
        data+=li/4;g-=d.T@coefficient;parts.append((li,coefficient,margin))
    rec=dict(aggregation=kind,total=data+penalty,data_loss=data,l2_penalty=penalty,weight_l2=float(np.linalg.norm(w)),gradient_l2=float(np.linalg.norm(g)),gradient_inf=float(np.max(abs(g))),numerical_gap_bound=float(g@g/.2),pair_accuracy_per_incident=[float(np.mean(m>0)) for _,_,m in parts])
    return rec,g,parts
def verify():
    frozen=read(HERE/'freeze.json');files=frozen['sha256']
    for name,digest in files.items():assert sha(ROOT/name)==digest,name
    snapshots={p:sha(p) for p in [HERE/'model.npz',HERE/'result.json',HERE/'fit.json',HELPER]}
    original=dict(np.load(SOURCE/'model.npz',allow_pickle=False));old=dict(np.load(BASE/'model.npz',allow_pickle=False));new=dict(np.load(HERE/'model.npz',allow_pickle=False))
    assert set(old)==set(new)==set(original)=={'embedding_mean','components','feature_mean','feature_scale','weights'}
    for k in new:
        assert new[k].shape==old[k].shape and new[k].dtype==old[k].dtype and np.isfinite(new[k]).all()
        if k!='weights':np.testing.assert_array_equal(new[k],old[k]);np.testing.assert_array_equal(new[k],original[k])
    assert old['weights'].shape==new['weights'].shape==(14,)
    jobs={j['ID']:j for j in read(SOURCE/'training_inputs.json')['jobs'] if j['ID'] in IDS};refs={r['ID']:r for r in read(SOURCE/'training_references.json')['cases'] if r['ID'] in IDS}
    assert len(jobs)==len(refs)==4 and all(refs[s]['eligible'] for s in IDS)
    e=np.stack([np.load(SOURCE/'features'/f'{s}.npy',allow_pickle=False) for s in IDS]);z=(e/np.maximum(np.linalg.norm(e,axis=2,keepdims=True),1e-12)-original['embedding_mean'])@original['components'].T;raw=[]
    for sid,a in zip(IDS,z):
        t=np.array(jobs[sid]['times']);previous=np.concatenate([a[:1],a[:-1]]);following=np.concatenate([a[1:],a[-1:]])
        raw.append(np.column_stack([a,a-previous,following-a,(t-t[0])/(t[-1]-t[0]),np.arange(12)==0]))
    x=(np.stack(raw)-original['feature_mean'])/original['feature_scale'];assert x.shape==(4,12,14)
    np.testing.assert_array_equal(x,np.load(ROOT/'artifacts/stage2_entry_feasibility_20260920/design.npy',allow_pickle=False))
    protocol=read(HERE/'protocol.json');prior_protocol=read(BASE/'protocol.json');prior=read(BASE/'result.json');result=read(HERE/'result.json');fit=read(HERE/'fit.json')
    assert protocol['train_ids']==IDS and protocol['L2']==.1 and protocol['fits_authorized']==1 and protocol['bounds'] is None
    assert protocol['initialization']=='zeros(14), no warm start' and protocol['optimizer']=='L-BFGS-B' and protocol['jac']
    assert protocol['options']==prior_protocol['options']=={'maxiter':500,'ftol':1e-12,'gtol':1e-8,'maxls':50}
    same(protocol['cases'],prior_protocol['cases']);same(fit,result['fit']);assert fit['success'] and len(result['cases'])==4
    assert result['training_runs']==1 and result['ccd_evaluations']==result['encoder_forwards']==result['followup_runs']==0
    assert not result['submission_changed'] and not result['candidate_adopted']
    diffs=[];records=[];deltas=[];decompositions=[]
    groups={'appearance_pca':range(4),'previous_difference':range(4,8),'following_difference':range(8,12),'relative_time':[12],'first_indicator':[13]}
    for n,sid in enumerate(IDS):
        j,r=jobs[sid],refs[sid];ts=[F(str(t)) for t in j['times']];lo,hi=map(lambda t:F(str(t)),r['reference_seconds']);m=[]
        for i,a in enumerate(ts):
            lower=max([lo]+[(a+b)/2 for k,b in enumerate(ts) if k<i]);upper=min([hi]+[(a+b)/2 for k,b in enumerate(ts) if k>i])
            if lower<=upper:m.append(i)
        p=[(i,k) for i,a in enumerate(ts) for k,b in enumerate(ts) if i!=k and max(abs(a-lo)-abs(b-lo),abs(a-hi)-abs(b-hi))<-F(1,10**9)]
        assert [list(q) for q in p]==r['preference_pairs'];top=np.array([(i,k) for i,k in p if i in m and k not in m])
        diffs.append(x[n,top[:,0]]-x[n,top[:,1]])
        meta=dict(ID=sid,reference_seconds=r['reference_seconds'],reference_frames=r['reference_frames'],frames=j['frames'],times=j['times'],possible_nearest_indices=m,possible_nearest_frames=[j['frames'][k] for k in m],old_pair_count=len(p),top_pairs=top.tolist(),top_pair_count=len(top))
        same(meta,protocol['cases'][n]);given=result['cases'][n]
        for k,v in meta.items():same(v,given[k])
        a=evaluate(x[n],old['weights'],j,r,m,top);b=evaluate(x[n],new['weights'],j,r,m,top)
        same(a,given['old']);same(a,prior['cases'][n]['new']);same(b,given['new']);records.append(dict(ID=sid,old=a,new=b))
        delta=paired(a['selected_seconds'],b['selected_seconds'],*r['reference_seconds']);close(delta['absolute_error_delta_seconds'],given['paired_error_change_range']);assert delta['exact_error_delta_seconds']==given['paired_error_change_exact'];deltas.append(dict(ID=sid,**delta))
        terms_row={'ID':sid}
        for label,w,ev in [('old',old['weights'],a),('new',new['weights'],b)]:
            scores=x[n]@w;inside=max(m,key=lambda k:(scores[k],-k));outside=max([k for k in range(12) if k not in m],key=lambda k:(scores[k],-k))
            terms={name:sum((F.from_float(float(x[n,inside,k]))-F.from_float(float(x[n,outside,k])))*F.from_float(float(w[k])) for k in ix) for name,ix in groups.items()}
            derived={k:float(v) for k,v in terms.items()};derived['sum']=float(sum(terms.values()));same(derived,given['score_decomposition'][label]);close(derived['sum'],ev['inside_minus_outside']);terms_row[label]=derived
        decompositions.append(terms_row)
    assert [len(d) for d in diffs]==[11,11,11,20]
    objectives={};weight_analysis={};gradients={}
    for label,w in [('old',old['weights']),('new',new['weights'])]:
        linear,gold,oparts=calculate(w,diffs,'old');hard,ghard,hparts=calculate(w,diffs,'hard')
        objectives[f'{label}_weights_old_objective']=linear;objectives[f'{label}_weights_hard_objective']=hard;gradients[label]=ghard
        rows=[]
        for c,(oloss,ocoef,margin),(hloss,hcoef,_) in zip(protocol['cases'],oparts,hparts):
            assert hloss>=oloss-1e-12 and 0<float(ocoef.sum())<.25 and 0<float(hcoef.sum())<.25
            share=hcoef/hcoef.sum();expected=np.exp(-margin-np.logaddexp.reduce(-margin));close(share,expected)
            pairs=[dict(preferred_frame=c['frames'][i],other_frame=c['frames'][j],margin=float(v),old_gradient_coefficient=float(o),hard_gradient_coefficient=float(h),old_within_incident_share=float(o/ocoef.sum()),hard_within_incident_share=float(h/hcoef.sum())) for (i,j),v,o,h in zip(c['top_pairs'],margin,ocoef,hcoef)]
            rows.append(dict(ID=c['ID'],old_incident_loss=oloss,hard_incident_loss=hloss,old_coefficient_sum=float(ocoef.sum()),hard_coefficient_sum=float(hcoef.sum()),pairs=pairs))
        weight_analysis[f'at_{label}_weights']=rows
    same(objectives,result['objectives']);same(weight_analysis,result['gradient_weight_analysis'])
    zero_old,g0old,_=calculate(np.zeros(14),diffs,'old');zero,g0,_=calculate(np.zeros(14),diffs,'hard')
    close(zero_old['total'],zero['total']);close(g0old,g0);close(zero['total'],np.log(2));same(zero,fit['initial']);same(objectives['new_weights_hard_objective'],fit['final'])
    assert fit['final']['total']<fit['initial']['total'] and fit['final']['gradient_inf']<1e-5
    # Evaluate actual runner objective numerically, never importing or invoking minimize.
    tree=ast.parse((HERE/'run.py').read_text());node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='objective')
    ns={'np':np,'logsumexp':logsumexp};exec(compile(ast.Module(body=[node],type_ignores=[]),str(HERE/'run.py'),'exec'),ns)
    finite=[];epsilon=1e-6
    for label,w in [('old',old['weights']),('new',new['weights'])]:
        value,g=ns['objective'](w,diffs);close(value,objectives[f'{label}_weights_hard_objective']['total']);close(g,gradients[label])
        fd=np.array([(ns['objective'](w+epsilon*np.eye(14)[k],diffs)[0]-ns['objective'](w-epsilon*np.eye(14)[k],diffs)[0])/(2*epsilon) for k in range(14)])
        np.testing.assert_allclose(fd,gradients[label],atol=1e-8,rtol=0)
        finite.append(dict(weights=label,epsilon=epsilon,maximum_absolute_error=float(np.max(abs(fd-gradients[label]))),actual_runner_function=True,objective_evaluations=28,optimizer_calls=0))
    summary={}
    for version in ['old','new']:
        rows=[r[version] for r in records];c=Counter(r['within_0_3_status'] for r in rows)
        summary[version]=dict(possible_nearest_top1=sum(r['in_possible_nearest_set'] for r in rows),within_0_3_counts={k:c[k] for k in ['correct_all_reference_times','wrong_all_reference_times','indeterminate']},mean_error_lower=float(np.mean([r['absolute_error_seconds'][0] for r in rows])),mean_error_upper=float(np.mean([r['absolute_error_seconds'][1] for r in rows])))
    same(summary,result['summary']);same(summary['old'],prior['summary']['new'])
    mean=[sum(F(d['exact_error_delta_seconds'][i]) for d in deltas)/4 for i in [0,1]];close(list(map(float,mean)),result['paired_mean_error_change_range']);assert list(map(str,mean))==result['paired_mean_error_change_exact']
    acc=[sum(d['accuracy_delta'][i] for d in deltas)/4 for i in [0,1]]
    for name,digest in files.items():assert sha(ROOT/name)==digest,name
    for p,digest in snapshots.items():assert sha(p)==digest,p
    return dict(status='PASS',verifier_sha256=sha(Path(__file__)),independent_math_helper_sha256=sha(HELPER),frozen_files_preserved=len(files),baseline_model_sha256=sha(BASE/'model.npz'),candidate_model_sha256=snapshots[HERE/'model.npz'],features_reconstructed_exact=True,nonweight_arrays_exactly_unchanged=True,train_ids=IDS,top_pair_counts=[11,11,11,20],total_pairs=53,objectives=objectives,finite_difference_checks=finite,zero_initial_old_new_loss_and_gradient_match=True,same_weight_gradient_coefficient_analysis=weight_analysis,case_results=records,score_decomposition=decompositions,summary=summary,paired_same_T=dict(rows=deltas,mean_absolute_error_delta_seconds=list(map(float,mean)),mean_accuracy_delta=acc),training_runs_reported=1,additional_optimizer_fits=0,additional_pca_fits=0,additional_encoder_forwards=0,model_weight_file_updates=0,ccd_files_opened=0,excluded_feature_files_opened=0,verification_python=sys.version,verification_numpy=np.__version__,limitations=['Same-weight coefficient comparisons isolate the aggregation formula mathematically. They are not independent generalization evidence.','Both within-incident relative pair shares and total coefficient mass change, despite equal1/4 incident loss weighting.','Fixed zero initialization has identical initial loss and gradient under both aggregations; subsequent curvatures and coefficients differ.','Hard and old losses have different aggregation; their scalar values are compared across both objectives explicitly. Jensen inequality hard>=old was checked per incident at each fixed weight vector.','Central differences are numerical objective evaluations only, not additional fits or model weight updates.','Train4 weak-reference resubstitution remains distinct from independent human validation, counterpart understanding and official S2.'])
if __name__=='__main__':
    out=HERE/'independent_verification.json';assert not out.exists();r=verify();out.write_text(json.dumps(r,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    text=['# 독립 harder-pair 집계 검증: PASS','','학습4건의 고정14특징·PCA·정규화·P/M/53쌍과 비가중치 배열 보존을 확인했다. 새 log-mean-exp 집계는 np.logaddexp.reduce로 별도 계산했고, 실제 runner objective의 baseline/new 가중치 중앙차분(ε=1e-6)으로 gradient를 확인했다. 추가 optimizer 호출은0이다.','','|사례|기존→새 프레임|새 ±0.3 판정|','|---|---:|---|']
    for row in r['case_results']:text.append(f"|{row['ID']}|{row['old']['selected_frame']}→{row['new']['selected_frame']}|{row['new']['within_0_3_status']}|")
    text+=['',f"같은 T 기준 평균 절대오차 변화 {r['paired_same_T']['mean_absolute_error_delta_seconds']}초, 정확도 변화 {r['paired_same_T']['mean_accuracy_delta']}. 선택 프레임 변화와 시간 정확도 개선은 없다.",'','동일 가중치에서 옛/새 집계의 pair gradient 계수와 비중을 전수 대조했다. 새 집계는 어려운 쌍의 상대 비중뿐 아니라 사고별 계수 총합도 바꾼다. 초기 zero 가중치에서 두 식의 손실·gradient는 같으며, 이후의 계수 분포는 달라진다. 이는 식과 결과의 산술 검증이며 일반화 효과를 입증하지 않는다.','',f"동결 {r['frozen_files_preserved']}파일·기존 모델 보존. 추가 학습·PCA·encoder·CCD/제외특징 접근0."]
    (HERE/'independent_verification.md').write_text('\n'.join(text)+'\n')
    print(json.dumps({k:r[k] for k in ['status','frozen_files_preserved','finite_difference_checks','summary','paired_same_T']},ensure_ascii=False))
