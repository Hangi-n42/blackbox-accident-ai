"""Separate arithmetic audit of the completed fit's score decomposition."""
import hashlib,json
from pathlib import Path
from fractions import Fraction as F
import numpy as np
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];OLD=ROOT/'artifacts/stage2_entry_selector_20260920'
read=lambda p:json.loads(p.read_text())
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def close(a,b):np.testing.assert_allclose(a,b,rtol=1e-10,atol=1e-10)
def main():
    out=HERE/'score_verification.json';assert not out.exists()
    freeze=read(HERE/'freeze.json')['sha256']
    for name,h in freeze.items():assert sha(ROOT/name)==h
    primary=read(HERE/'independent_verification.json');assert primary['status']=='PASS'
    model_paths=[OLD/'model.npz',HERE/'model.npz'];original=[sha(p) for p in model_paths]
    state,new=[dict(np.load(p,allow_pickle=False)) for p in model_paths]
    assert original==[primary['old_model_sha256'],primary['candidate_model_sha256']]
    x=np.load(ROOT/'artifacts/stage2_entry_feasibility_20260920/design.npy',allow_pickle=False)
    assert primary['feature_reconstruction_exact'] and x.shape==(4,12,14)
    result=read(HERE/'result.json');analysis=read(HERE/'score_analysis.json')
    refs={r['ID']:r for r in read(OLD/'training_references.json')['cases'] if r['ID'] in ['00000','00003','00006','00013']}
    definitions={'appearance_pca':range(0,4),'previous_difference':range(4,8),'following_difference':range(8,12),'relative_time':[12],'first_candidate_indicator':[13]}
    rows=[];gradient_totals={'old258':F(0),'new53':F(0)}
    for z,c,record,delta in zip(x,result['cases'],analysis['cases'],primary['paired_same_unknown_time']['rows']):
        assert c['ID']==record['ID']==delta['ID']
        scores=z@new['weights'];winner=int(np.argmax(scores));inside=max(c['possible_nearest_indices'],key=lambda j:(scores[j],-j))
        assert record['winner_frame']==c['frames'][winner] and record['best_inside_frame']==c['frames'][inside]
        # Exact arithmetic over the cached floating-point numbers for each group.
        terms={name:sum((F.from_float(float(z[winner,k]))-F.from_float(float(z[inside,k])))*F.from_float(float(new['weights'][k])) for k in indices) for name,indices in definitions.items()}
        for name,value in terms.items():close(float(value),record['winner_minus_best_inside_by_feature_group'][name])
        close(float(sum(terms.values())),record['winner_minus_best_inside_total'])
        close(record['paired_absolute_error_change_range'],delta['absolute_error_delta_seconds'])
        assert record['paired_absolute_error_change_exact']==delta['exact_error_delta_seconds']
        gradients={}
        for label,p in [('old258',refs[c['ID']]['preference_pairs']),('new53',c['top_pairs'])]:
            d=[F.from_float(float(z[i,13]))-F.from_float(float(z[j,13])) for i,j in p]
            contribution=-sum(d)/(8*len(d));gradient_totals[label]+=contribution
            derived=dict(first_indicator_gradient_contribution_to_total=float(contribution),positive_first_indicator_pairs=sum(v>0 for v in d),negative_first_indicator_pairs=sum(v<0 for v in d),pair_count=len(d))
            given=record['zero_initialization_first_indicator_gradient'][label]
            for k,v in derived.items():close(v,given[k])
            gradients[label]=derived
        rows.append(dict(ID=c['ID'],winner_frame=record['winner_frame'],best_inside_frame=record['best_inside_frame'],exact_group_score_contributions={k:str(v) for k,v in terms.items()},group_score_contributions={k:float(v) for k,v in terms.items()},total=float(sum(terms.values())),zero_initial_gradient=gradients))
    for name,index in [('first_indicator_weight',13),('relative_time_weight',12)]:
        close(analysis[name]['old'],state['weights'][index]);close(analysis[name]['new'],new['weights'][index])
    paired=primary['paired_same_unknown_time']
    close(analysis['paired_mean_absolute_error_change_range'],paired['mean_absolute_error_delta_seconds'])
    assert analysis['paired_mean_absolute_error_change_exact']==paired['exact_mean_absolute_error_delta_seconds']
    assert analysis['additional_fits']==analysis['ccd_accesses']==0
    for name,h in freeze.items():assert sha(ROOT/name)==h
    assert original==[sha(p) for p in model_paths]
    payload=dict(status='PASS',verifier_sha256=sha(Path(__file__)),analysis_sha256=sha(HERE/'score_analysis.json'),primary_verification_sha256=sha(HERE/'independent_verification.json'),models_and_frozen13_preserved=True,rows=rows,total_zero_first_indicator_gradient={k:float(v) for k,v in gradient_totals.items()},first_indicator_weight=analysis['first_indicator_weight'],paired_mean_absolute_error_delta_seconds=paired['mean_absolute_error_delta_seconds'],additional_optimizer_calls=0,additional_encoder_forwards=0,ccd_or_excluded_feature_accesses=0,limits=['Exact score-term accounting at fixed weights, not an ablation or proof of training causality.','First-indicator values are standardized; coefficient alone differs from first-versus-other score contribution.','Changing pair membership changes which relations receive mass after within-incident averaging. Four incidents remain equally weighted.','A positive first-indicator effect in00000 does not explain00013, where both competing candidates have firstflag0.'])
    out.write_text(json.dumps(payload,ensure_ascii=False,indent=2)+'\n')
    (HERE/'score_verification.md').write_text('# 점수 분해 독립 검증: PASS\n\n고정 float64 특징·가중치를 정확 유리수로 옮겨 그룹별 점수차 합과 원기록 일치를 확인했다. 00000의 f0−f561 차이는 first flag +1.031368과 나머지 -0.032507의 합 +0.998861이다. 00013의 f109−f545는 first flag0이며 prev +0.180264, next +0.149400 등이 합쳐 +0.175378이다.\n\n초기 sigmoid0.5·사고평균1/4에 따른 first항 gradient -mean(d)/8도 각 사고별로 일치한다. 이는 고정 결과의 산술 분해이며 feature 제거 실험이나 학습 원인의 인과검증은 아니다. 같은 T의 평균 MAE 변화 [+0.612399,+1.408247]초, 기존 모델·동결13 보존을 다시 확인했다. 추가 학습·모델 호출·CCD 접근0이다.\n')
    print(json.dumps({k:payload[k] for k in ['status','total_zero_first_indicator_gradient','first_indicator_weight','paired_mean_absolute_error_delta_seconds']}))
if __name__=='__main__':main()
