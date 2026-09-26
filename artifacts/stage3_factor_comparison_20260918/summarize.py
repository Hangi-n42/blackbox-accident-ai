from pathlib import Path
import json,hashlib
import numpy as np,pandas as pd,joblib
O=Path(__file__).resolve().parent;R=O.parents[1]
metrics=pd.concat([pd.read_csv(O/(e+'_metrics.csv')) for e in ['dis','raft']],ignore_index=True)
rows=[];gates={}
for name in metrics.variant.unique():
 gates[name]=[]
 for fold in [1,2,3]:
  x=metrics[(metrics.variant==name)&(metrics.fold==fold)].set_index('scope');b=metrics[(metrics.variant=='dis_base')&(metrics.fold==fold)].set_index('scope')
  gates[name].append({'fold':fold,'heldout_f1_improves':bool(x.loc['heldout','macro_f1']>b.loc['heldout','macro_f1']),'opposite_no_increase_all_scopes':bool((x.loc[['heldout','rav4','civic'],'opposite']<=b.loc[['heldout','rav4','civic'],'opposite']).all()),'both_directions_no_increase_all_scopes':bool((x.loc[['heldout','rav4','civic'],['accel_to_decel','decel_to_accel']]<=b.loc[['heldout','rav4','civic'],['accel_to_decel','decel_to_accel']]).all().all()),'public_f1_nonregression':bool(x.loc['public','macro_f1']>=b.loc['public','macro_f1'])})
  rows.append({'variant':name,'fold':fold,'F1':x.loc['heldout','macro_f1'],'delta_F1':x.loc['heldout','macro_f1']-b.loc['heldout','macro_f1'],'opposite':int(x.loc['heldout','opposite']),'delta_opposite':int(x.loc['heldout','opposite']-b.loc['heldout','opposite']),'accel_to_decel':int(x.loc['heldout','accel_to_decel']),'decel_to_accel':int(x.loc['heldout','decel_to_accel']),'public_F1':x.loc['public','macro_f1'],'public_delta_F1':x.loc['public','macro_f1']-b.loc['public','macro_f1']})
summary=pd.DataFrame(rows);summary.to_csv(O/'comparison.csv',index=False)
selected=pd.read_csv(R/'artifacts/stage3_reversal_diagnosis_20260917/selected_74.csv');context=pd.read_csv(R/'artifacts/stage3_reversal_diagnosis_20260917/full_context.csv');controls=context[context.strict_mask&(context.proxy_truth==0)&(context.rav4_prediction==0)&(context.mixed_budget_rav4_prediction==0)]
diagnostic=[]
for name,parts in json.load(open(O/'freeze.json'))['variants'].items():
 engine=name.split('_')[0];m=joblib.load(O/'fold_1'/(name+'.joblib'))['accel']
 for id in ['expanded_14','extra_01']:
  a=np.load(O/engine/(id+'.npz'));x=np.concatenate([a[p] for p in parts],axis=1);pred=m.predict(x);proba=m.predict_proba(x)
  for scope,s in [('prior74',selected),('prior101controls',controls)]:
   ix=s[s.id==id]['index'].to_numpy(dtype=int)
   diagnostic.append({'variant':name,'id':id,'scope':scope,'n':len(ix),'accelerating':int((pred[ix]==0).sum()),'decelerating':int((pred[ix]==1).sum()),'constant':int((pred[ix]==2).sum()),'stopped':int((pred[ix]==3).sum())})
pd.DataFrame(diagnostic).to_csv(O/'known_case_diagnostics.csv',index=False)
timing={}
for engine in ['dis','raft']:
 a=[json.load(open(p)) for p in (O/engine).glob('*_timing.json')];timing[engine]={'clips':len(a),'frames':sum(r['frames'] for r in a),'pairs':sum(r['pairs'] for r in a),'seconds':sum(r['seconds'] for r in a)}
f=json.load(open(O/'freeze.json'));assert all(hashlib.file_digest(open(R/p,'rb'),'sha256').hexdigest()==h for p,h in {**f['inputs'],**f['production']}.items());passed=[name for name,v in gates.items() if name!='dis_base' and all(all(value for key,value in fold.items() if key!='fold') for fold in v)]
json.dump({'gates':gates,'passed_candidates':passed,'timing':timing,'production_and_inputs_unchanged':True,'no_promotion':True},open(O/'summary.json','w'),indent=2)
lines=['# Stage3 특징·광류 모델 분리 비교','', '2026-09-18. 가감속 분류만 비교했다. 운영 모델 변경 없음.','', '## 결과','', '|조건|fold|F1|기준 대비|반전|반전 증감|공개 F1|','|---|---:|---:|---:|---:|---:|---:|']
for r in summary.itertuples():lines.append(f'|{r.variant}|{r.fold}|{r.F1:.6f}|{r.delta_F1:+.6f}|{r.opposite}|{r.delta_opposite:+d}|{r.public_F1:.6f}|')
lines+=['','채택 조건을 모두 통과한 후보: '+(', '.join(passed) or '없음')+'.','', '## 통제 및 범위','', '- 특징 비교는 DIS를 유지했다. RAFT 비교는 기존864특징을 유지했다. 분류기 구조·하이퍼파라미터·학습 시각을 고정하고 계수만 각 조건에 맞춰 적합했다.', '- 각fold 학습385개. 평가 시각은 fold1 1612개, fold2 1167개, fold3 561개. 중복 날짜가 있어 합산 독립 표본으로 해석하지 않는다.', '- 기존23개 comma 단편 및 공개5개 단편. 센서 대체 정답과 공개 공식 정답은 별도 보고한다. 개발 노출된 자료로 독립 일반화 검증이 아니다.', '- DIS 기존864특징 전체 캐시와 정확히 일치. 재학습 기준 예측이 과거 mixed_budget_rav4와 모든fold/공개자료에서 일치.', '- 각 방향 반전 및 차량별 지표는 dis_metrics.csv/raft_metrics.csv, 새로 생긴 반전과 수정된 반전도 같은 파일에 기록.', '- 기존74반전/101정상 진단은 known_case_diagnostics.csv. 진단 두영상만 좋아지는 조건을 선택하지 않는다.', '- 특징 차원이 달라지는 영향도 포함한다. affine 계수·잔차를 물리적 자차 운동의 정확한 분리로 해석하지 않는다.', '- RAFT는 동일 회색조를3채널 복제하여 광류 교체만 통제했다. 공식 RGB 벤치마크 성능을 재현했다고 주장하지 않는다. 우리 영상의 정답 광류가 없어 광류 정확도 자체는 평가하지 못한다.', '- Depth Anything 추가는 이번 비교에서 수행하지 않았다. 2D 특징 후보의 실패만으로 깊이 추정 필요성이 입증되는 것은 아니다.', '', '## 실행 및 검증','', 'run.py dis 및 run.py raft로 재현. 상세설계 METHOD.md, 고정파일 freeze.json, 실제실행 *_execution.log, CPU/MPS 실제입력 점검 raft_runtime_check.json.','', '전체 추출 시간(디코딩+특징계산 포함; DIS는 모든추가특징 포함, RAFT는864만 포함하므로 순수 광류속도 비교가 아님):']
for k,v in timing.items():lines.append(f'- {k}: {v["clips"]} clips, {v["frames"]} frames, {v["seconds"]:.2f}s.')
lines+=['','입력/라벨/운영파일 해시 보존 검사 통과. 모델 교체·제출 없음.']
(O/'REPORT.md').write_text('\n'.join(lines)+'\n');print(summary.to_string(index=False));print('passed',passed);print('timing',timing)
