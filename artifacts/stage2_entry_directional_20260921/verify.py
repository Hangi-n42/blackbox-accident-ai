"""Independent fixed-weight directional audit. Train4 only; no fit/forward."""
import hashlib,json,sys
from fractions import Fraction as F
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
BASE=ROOT/'artifacts/stage2_entry_l2_01_20260921';SOURCE=ROOT/'artifacts/stage2_entry_selector_20260920'
IDS=['00000','00003','00006','00013'];read=lambda p:json.loads(p.read_text())
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def close(a,b,atol=1e-12):np.testing.assert_allclose(a,b,atol=atol,rtol=1e-10)
def same(a,b):
    if isinstance(a,dict):
        assert set(a)==set(b)
        for k in a:same(a[k],b[k])
    elif isinstance(a,list):
        assert len(a)==len(b)
        for x,y in zip(a,b):same(x,y)
    elif isinstance(a,float):close(a,b)
    else:assert a==b,(a,b)
def verify():
    frozen=read(HERE/'freeze.json');files=frozen['sha256']
    for name,h in files.items():assert sha(ROOT/name)==h,name
    snapshots={p:sha(p) for p in [BASE/'model.npz',HERE/'result.json']}
    model=dict(np.load(BASE/'model.npz',allow_pickle=False));w=model['weights'];assert w.shape==(14,)
    jobs={j['ID']:j for j in read(SOURCE/'training_inputs.json')['jobs'] if j['ID'] in IDS}
    refs={r['ID']:r for r in read(SOURCE/'training_references.json')['cases'] if r['ID'] in IDS}
    assert len(jobs)==len(refs)==4 and all(refs[s]['eligible'] for s in IDS)
    emb=np.stack([np.load(SOURCE/'features'/f'{s}.npy',allow_pickle=False) for s in IDS])
    z=(emb/np.maximum(np.linalg.norm(emb,axis=2,keepdims=True),1e-12)-model['embedding_mean'])@model['components'].T;raw=[]
    for sid,a in zip(IDS,z):
        t=np.asarray(jobs[sid]['times']);previous=np.concatenate([a[:1],a[:-1]]);following=np.concatenate([a[1:],a[-1:]])
        raw.append(np.column_stack([a,a-previous,following-a,(t-t[0])/(t[-1]-t[0]),np.arange(12)==0]))
    x=(np.stack(raw)-model['feature_mean'])/model['feature_scale']
    assert x.shape==(4,12,14);np.testing.assert_array_equal(x,np.load(ROOT/'artifacts/stage2_entry_feasibility_20260920/design.npy',allow_pickle=False))
    result=read(HERE/'result.json');base_protocol=read(BASE/'protocol.json');assert result['lambda']==base_protocol['L2']==.1
    assert [c['ID'] for c in base_protocol['cases']]==IDS and jobs['00000']['frames'][6]==561 and jobs['00000']['frames'][0]==0
    target=x[0,6]-x[0,0];length=float(np.linalg.norm(target));direction=target/length
    assert length>0;close(direction@direction,1);close(direction,result['direction'])
    target_margin=float(target@w);close(target_margin,result['target_current_margin']);close(length,result['target_margin_directional_derivative'])
    groups={'appearance_pca':slice(0,4),'previous_difference':slice(4,8),'following_difference':slice(8,12),'relative_time':slice(12,13),'first_indicator':slice(13,14)}
    for name,s in groups.items():close(float(np.linalg.norm(direction[s])),result['direction_by_feature_group'][name])
    pair_rows=[];incidents=[];objective_data=0.;data_gradient=np.zeros(14);all_diffs=[];pair_counts=[]
    for n,sid in enumerate(IDS):
        j,r=jobs[sid],refs[sid];ts=[F(str(t)) for t in j['times']];lo,hi=map(lambda t:F(str(t)),r['reference_seconds']);m=[]
        for i,a in enumerate(ts):
            lower=max([lo]+[(a+b)/2 for k,b in enumerate(ts) if k<i]);upper=min([hi]+[(a+b)/2 for k,b in enumerate(ts) if k>i])
            if lower<=upper:m.append(i)
        p=[(i,k) for i,a in enumerate(ts) for k,b in enumerate(ts) if i!=k and max(abs(a-lo)-abs(b-lo),abs(a-hi)-abs(b-hi))<-F(1,10**9)]
        assert [list(a) for a in p]==r['preference_pairs'];top=[(i,k) for i,k in p if i in m and k not in m]
        assert [list(a) for a in top]==base_protocol['cases'][n]['top_pairs'];assert m==base_protocol['cases'][n]['possible_nearest_indices']
        pair_counts.append(len(top));d=np.array([x[n,i]-x[n,k] for i,k in top]);all_diffs.append(d)
        margins=d@w;sigmoid=np.exp(-np.logaddexp(0,margins));objective_data+=float(np.logaddexp(0,-margins).mean())/4
        gradients=-d*sigmoid[:,None]/(4*len(top));incident=gradients.sum(axis=0);data_gradient+=incident;projections=gradients@direction
        incidents.append(dict(ID=sid,pairs=len(top),gradient=incident.tolist(),projection=float(incident@direction),support_sum=float(projections[projections<0].sum()),oppose_sum=float(projections[projections>0].sum()),support_count=int(np.sum(projections<0)),oppose_count=int(np.sum(projections>0))))
        for (i,k),margin,gradient,projection,change in zip(top,margins,gradients,projections,d@direction):
            pair_rows.append(dict(ID=sid,winner_index=i,loser_index=k,winner_frame=j['frames'][i],loser_frame=j['frames'][k],current_margin=float(margin),margin_directional_derivative=float(change),gradient=gradient.tolist(),loss_directional_derivative=float(projection),projection_by_feature_group={name:float(gradient[s]@direction[s]) for name,s in groups.items()}))
    assert pair_counts==[11,11,11,20] and len(pair_rows)==53;same(pair_rows,result['pairs']);same(incidents,result['incidents'])
    reg=.1*w;total=data_gradient+reg;total_projection=float(total@direction)
    close(float(data_gradient@direction),result['data_projection']);close(float(reg@direction),result['regularization_projection']);close(total_projection,result['total_projection']);close(float(np.linalg.norm(total)),result['total_gradient_l2'])
    close(objective_data+float(.05*(w@w)),result['objective'])
    independent_whole=.1*w.copy()
    for d in all_diffs:independent_whole-=d.T@np.exp(-np.logaddexp(0,d@w))/(4*len(d))
    close(independent_whole,total,1e-14)
    projection_sum=sum(r['loss_directional_derivative'] for r in pair_rows)+float(reg@direction)
    close(projection_sum,total_projection,1e-14)
    for inc in incidents:close(inc['support_sum']+inc['oppose_sum'],inc['projection'],1e-14)
    direct=[r for r in pair_rows if r['ID']=='00000' and r['winner_index']==6 and r['loser_index']==0]
    assert len(direct)==1;same(direct[0],result['direct_target_pair'])
    direct_formula=-float(np.exp(-np.logaddexp(0,target_margin)))*length/(4*11)
    close(direct_formula,direct[0]['loss_directional_derivative']);close(direct[0]['margin_directional_derivative'],length)
    reg_formula=.1*target_margin/length;close(reg_formula,result['regularization_projection'])
    assert direct_formula<0 and reg_formula<0 and float(data_gradient@direction)>0
    assert result['training_runs']==result['encoder_forwards']==result['ccd_accesses']==result['weight_updates']==result['submission_changes']==0
    assert result['frozen_files_preserved']==len(files)
    for name,h in files.items():assert sha(ROOT/name)==h,name
    for p,h in snapshots.items():assert sha(p)==h,p
    return dict(status='PASS',verifier_sha256=sha(Path(__file__)),result_sha256=snapshots[HERE/'result.json'],model_sha256=snapshots[BASE/'model.npz'],frozen_files_preserved=len(files),feature_reconstruction_exact=True,train_ids=IDS,top_pairs_per_incident=pair_counts,pairs_verified=53,direction_norm=float(np.linalg.norm(direction)),target_current_margin=target_margin,target_margin_directional_derivative=length,incidents=incidents,pairs=pair_rows,direct_pair_projection=direct_formula,direct_pair_closed_form='-sigmoid(-target_margin)*norm(target_difference)/(4*11)',regularization_projection=reg_formula,regularization_identity='lambda*target_margin/norm(target_difference)',data_projection=float(data_gradient@direction),total_projection=total_projection,pair_and_regularization_projection_sum=projection_sum,total_gradient_l2=float(np.linalg.norm(total)),objective=objective_data+float(.05*(w@w)),largest_opposing_pairs=sorted(pair_rows,key=lambda r:r['loss_directional_derivative'],reverse=True)[:6],sign_convention='Derivative of loss with respect to positive motion along d: negative lowers that component locally; positive raises it. d increases f561 minus f0 margin.',weight_updates=0,additional_optimizer_fits=0,additional_pca_fits=0,additional_encoder_forwards=0,ccd_files_opened=0,excluded_feature_files_opened=0,verification_python=sys.version,verification_numpy=np.__version__,limitations=['One local direction at fixed weights, not an intervention, ablation or additional training experiment.','winner/loser fields denote reference-preferred pairs, not necessarily the current score order.','Regularization supports this particular correction direction; this does not measure the outcome of changing lambda and reoptimizing.','Near-zero total projection reflects cancellation at the current objective solution; it does not prove that f561 becomes the overall top1 after a finite move.','Weak training references and global visual features remain unchanged; no independent validation, CCD evaluation or official score.'])
if __name__=='__main__':
    out=HERE/'independent_verification.json';assert not out.exists();r=verify();out.write_text(json.dumps(r,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    lines=['# 독립 방향미분 검증: PASS','','고정 λ=0.1 가중치와 학습4건만으로 14특징·P/M/53쌍·d=(x561−x0)/norm을 독립 재구성했다. 모든 pair gradient, 사고별 합, 그룹별 투영과 전체 합이 원기록과 일치한다.','','|항|loss gradient의 d 투영|','|---|---:|']
    for row in r['incidents']:lines.append(f"|{row['ID']}|{row['projection']:.12f}|")
    lines += [f"|L2|{r['regularization_projection']:.12f}|",f"|전체|{r['total_projection']:.12g}|",'',f"직접 (f561,f0) 쌍은 {r['direct_pair_projection']:.12f}다. 음수는 +d 방향에서 해당 loss가 감소함을, 양수는 증가함을 뜻한다. 직접 쌍과 L2는 이 교정 방향을 지지하고, 데이터항 전체는 반대한다. L2 투영=λ×현재margin/norm 관계도 확인했다.",'',f"동결 {r['frozen_files_preserved']}파일·기존 가중치 보존. 추가 optimizer·PCA·encoder·CCD/제외특징 접근0.",'','고정 결과의 한 방향에 대한 국소 미분이다. 이를 λ 변경의 인과 효과, 유한 이동 후 전체 top1 개선, 독립 성능 또는 일반화 증거로 해석하지 않는다.']
    (HERE/'independent_verification.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({k:r[k] for k in ['status','frozen_files_preserved','pairs_verified','direct_pair_projection','data_projection','regularization_projection','total_projection','total_gradient_l2']},ensure_ascii=False))
