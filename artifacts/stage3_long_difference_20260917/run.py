"""Frozen single ablation; reuse all23 caches and existing label/metric implementations."""
import importlib.util, json, time, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from sklearn.base import clone
from sklearn.exceptions import ConvergenceWarning
from threadpoolctl import threadpool_limits

O=Path(__file__).resolve().parent; R=O.parents[1]; B=R/'artifacts/stage3_training_basis_20260917'
spec=importlib.util.spec_from_file_location('basis',B/'run.py'); basis=importlib.util.module_from_spec(spec); spec.loader.exec_module(basis)
read,write,sha,ACCEL=basis.read,basis.write,basis.sha,basis.ACCEL

def metrics(truth,pred):
    result=basis.old.metrics(truth,pred)
    cm=np.asarray(result['confusion']); counts=cm.sum(axis=1)
    result['support']=dict(zip(map(str,ACCEL),map(int,counts)))
    result['recall']={str(c):float(cm[i,i]/counts[i]) if counts[i] else None for i,c in enumerate(ACCEL)}
    return result

def main():
    assert not (O/'freeze.json').exists(),'Fresh directory required'
    cases=read(B/'cases.json'); source_files=read(R/'artifacts/pipeline_diagnosis_20260917/freeze.json')['files']
    assert all(sha(R/p)==h for p,h in source_files.items())
    old=pd.read_csv(B/'scored_rows.csv'); audit={}; runs=[]
    for row in cases:
        sub=old[old.id==row['id']].sort_values('sample_index')
        audit[row['id']]={'route':row['route'],'vehicle':row['vehicle'],'role':row['role'],**metrics(sub.truth,sub.expanded)}
        indices=sub.loc[(sub.truth=='CONSTANT')&(sub.expanded!='CONSTANT'),'sample_index'].to_numpy()
        for group in np.split(indices,np.flatnonzero(np.diff(indices)>1)+1):
            if len(group):runs.append({'id':row['id'],'first_index':int(group[0]),'last_index':int(group[-1]),'n':len(group),'sample_bin_duration_s':len(group)/10,'note':'scored stable proxy region only; excluded rows break runs'})
    write(O/'baseline_audit.json',audit); write(O/'constant_error_runs.json',runs)
    (O/'EVALUATION_CONTRACT.md').write_text('''# 固定比較 / 고정 비교 기준

- 공식 공개50시각과 comma 센서 대체 정답은 따로 평가한다. 종합 대회 점수를 계산하지 않는다.
- 주 방향437학습/3246비교, 역방향646학습/2204비교. 기존23구간·마스크·학습2Hz·평가10Hz 유지.
- 13787중5450명확 시각만 평가한다. 제외 시각의 정답을 채우지 않는다. route별 전체4범주 F1에는 지원 수를 병기한다.
- 후보는 기존864특징의 마지막144열(평균15의 ±15 차분)만 제외한720특징이다. 평균31·짧은차분·공간영역·라벨·분류기 설정은 고정한다.
- 사전 보수적 다음 단계 조건: 주방향 F1 개선, 등속오류 감소, 양방향 가감속 반전 비증가, 역방향 F1 비악화, 양방향 감속/정지 recall 비악화, 양방향 공개 F1 비악화. 독립 검증 통과나 운영 채택 조건으로 부르지 않는다.
- 위 조건은 실험 선택을 위한 보수적 규칙이며 통계적 유의성 기준이 아니다. 단일 미충족도 숨기지 않고 trade-off를 보고한다.
- 전환 거리/미검출은 임계값·동기화가 불확실한 보조 진단이며 공식 항목이나 자동 탈락 조건이 아니다.
- 기존 예측을 본 개발 노출 자료다. 결과 확인 후 임계값·후보를 바꾸지 않는다. 운영 모델·조향은 그대로 보존한다.
''')
    columns={'control':np.arange(864),'drop_long':np.arange(720)}
    assert len(columns['drop_long'])==5*12*4*3
    write(O/'freeze.json',{'script_sha256':sha(Path(__file__)),'contract_sha256':sha(O/'EVALUATION_CONTRACT.md'),'cases_sha256':sha(B/'cases.json'),'labels_sha256':{r['labels_npz']:sha(B/r['labels_npz']) for r in cases},'source_files':source_files,'candidate':'remove last144 columns only','features':{k:len(v) for k,v in columns.items()},'before_candidate_fit':True})
    features={r['id']:basis.loadcache(R/r['feature_cache'] if r['feature_cache'] else B/(r['id']+'_features.npy')) for r in cases}
    labels={r['id']:np.load(B/r['labels_npz'])['accel_candidate'] for r in cases}
    models={}; fitting=[]
    for direction,role,checkpoint,expected in [('primary','train','expanded_rav4',437),('reverse','comparison','reverse_civic',646)]:
        template=joblib.load(B/(checkpoint+'.joblib'))['accel']
        train=[r for r in cases if r['role']==role]
        X=np.concatenate([features[r['id']][r['training_indices']] for r in train]); Y=np.concatenate([labels[r['id']][r['training_indices']] for r in train])
        assert len(Y)==expected and X.shape[1]==864 and np.isfinite(X).all()
        for variant,ix in columns.items():
            model=clone(template); start=time.perf_counter()
            with warnings.catch_warnings(record=True) as ws,threadpool_limits(limits=2):
                warnings.simplefilter('always'); model.fit(X[:,ix],Y)
            assert not any(issubclass(w.category,ConvergenceWarning) for w in ws)
            models[direction,variant]=model
            joblib.dump({'accel':model,'feature_indices':ix,'not_production_package':True},O/f'{direction}_{variant}.joblib')
            fitting.append({'direction':direction,'variant':variant,'n':len(Y),'seconds':time.perf_counter()-start,'iterations':model.named_steps['logisticregression'].n_iter_.tolist(),'warnings':[str(w.message) for w in ws]})
            if variant=='control':
                for r in cases:assert np.array_equal(model.predict(features[r['id']]),template.predict(features[r['id']])),'Control reproduction failed'
            print('fit',direction,variant,len(Y),flush=True)
    scored=[]; full=[]; lookup={}
    for r in cases:
        id=r['id']; direction='primary' if r['role']=='comparison' else 'reverse'; x=features[id]
        pred={v:models[direction,v].predict(x[:,ix]).astype(int) for v,ix in columns.items()}; lookup[id]=pred
        for i in range(len(x)):full.append({'id':id,'direction':direction,'sample_index':i,**{v:str(ACCEL[p[i]]) for v,p in pred.items()}})
        for i in r['evaluation_indices']:scored.append({'id':id,'scope':direction,'sample_index':i,'truth':str(ACCEL[labels[id][i]]),**{v:str(ACCEL[p[i]]) for v,p in pred.items()}})
    public=pd.read_csv(R/'Baseline/data/stage3/labels.csv')
    for id,g in public.groupby('ID'):
        x=np.load(R/'artifacts/data_pilot_20260916/comma_experiment/mac_public_features'/(id+'.npy'))
        for direction in ['primary','reverse']:
            pred={v:models[direction,v].predict(x[:,ix]).astype(int) for v,ix in columns.items()}
            for z in g.itertuples():scored.append({'id':id,'scope':'public_'+direction,'sample_index':int(z.sample_index),'truth':z.accel_label,**{v:str(ACCEL[p[z.sample_index]]) for v,p in pred.items()}})
    s=pd.DataFrame(scored); results={scope:{v:metrics(g.truth,g[v]) for v in columns} for scope,g in s.groupby('scope')}
    route_results={scope+'/'+id:{v:metrics(g.truth,g[v]) for v in columns} for (scope,id),g in s.groupby(['scope','id'])}
    transitions=[]
    for e in read(B/'transition_manifest.json'):transitions.append(dict(e,**{v:basis.prior.crossing(lookup[e['id']][v],e) for v in columns}))
    tr={}
    for direction,role in [('primary','comparison'),('reverse','train')]:
        ev=[e for e in transitions if e['role']==role]; both=[e for e in ev if all(e[v] is not None for v in columns)]
        tr[direction]={'n':len(ev),'both_detected':len(both),**{v:{'missing':sum(e[v] is None for e in ev),'matched_distance_s':float(np.mean([e[v] for e in both])) if both else None} for v in columns}}
    a=results['primary']; gates={'primary_f1_improved':a['drop_long']['macro_f1']>a['control']['macro_f1'],'primary_constant_errors_reduced':a['drop_long']['constant_errors']<a['control']['constant_errors']}
    for direction in ['primary','reverse']:
        a=results[direction]; p=results['public_'+direction]
        gates[direction+'_opposite_no_increase']=a['drop_long']['opposite_accel_errors']<=a['control']['opposite_accel_errors']
        gates[direction+'_public_no_regression']=p['drop_long']['macro_f1']>=p['control']['macro_f1']
        for c in ['DECELERATING','STOPPED']:gates[direction+'_'+c+'_recall_no_regression']=a['drop_long']['recall'][c]>=a['control']['recall'][c]
    gates['reverse_f1_no_regression']=results['reverse']['drop_long']['macro_f1']>=results['reverse']['control']['macro_f1']
    assert abs(results['primary']['control']['macro_f1']-read(B/'report.json')['metrics']['primary_civic_all']['expanded']['macro_f1'])<1e-12
    s.to_csv(O/'scored_rows.csv',index=False); pd.DataFrame(full).to_csv(O/'predictions.csv',index=False)
    changed=s[s.control!=s.drop_long].copy(); changed['effect']=np.where(changed.drop_long==changed.truth,'fixed',np.where(changed.control==changed.truth,'regressed','wrong_to_wrong')); changed.to_csv(O/'changed_rows.csv',index=False)
    write(O/'route_results.json',route_results); write(O/'transitions.json',transitions); write(O/'runtime.json',fitting)
    freeze=read(O/'freeze.json'); assert all(sha(B/p)==h for p,h in freeze['labels_sha256'].items())
    assert all(sha(R/p)==h for p,h in source_files.items())
    write(O/'report.json',{'results':results,'transitions':tr,'gates':gates,'decision':'further_review_only' if all(gates.values()) else 'do_not_adopt_tradeoffs_or_regression','control_reproduced':True,'production_and_labels_unchanged':True,'exit_status':0})
    print(json.dumps({'results':results,'gates':gates},ensure_ascii=False,indent=2))

if __name__=='__main__':main()
