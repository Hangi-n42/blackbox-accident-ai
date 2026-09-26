"""Explain frozen model differences on the same cached inputs; no fitting."""
import json, hashlib, importlib.util
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import confusion_matrix, f1_score
from threadpoolctl import threadpool_limits

O=Path(__file__).resolve().parent; R=O.parents[1]; B=R/'artifacts/stage3_training_basis_20260917'
read=lambda p:json.loads(p.read_text())
write=lambda p,x:p.write_text(json.dumps(x,ensure_ascii=False,indent=2))
sha=lambda p:hashlib.file_digest(p.open('rb'),'sha256').hexdigest()
N=np.array(['ACCELERATING','DECELERATING','CONSTANT','STOPPED'])

def metric(y,p):
    cm=confusion_matrix(y,p,labels=range(4)); n=cm.sum(1)
    return {'n':len(y),'macro_f1':float(f1_score(y,p,labels=range(4),average='macro',zero_division=0)),
            'confusion':cm.tolist(),'support':dict(zip(N.tolist(),n.tolist())),
            'recall':{str(k):float(cm[i,i]/n[i]) if n[i] else None for i,k in enumerate(N)}}

def main():
    assert not (O/'results.json').exists()
    cases=read(B/'cases.json'); models={v:joblib.load(B/f'{file}.joblib')['accel'] for v,file in [('rav4','expanded_rav4'),('civic','reverse_civic')]}
    inputs=[B/'cases.json',B/'expanded_rav4.joblib',B/'reverse_civic.joblib']+[B/r['labels_npz'] for r in cases]
    production=read(R/'artifacts/pipeline_diagnosis_20260917/freeze.json')['files']
    assert all(sha(R/p)==h for p,h in production.items())
    write(O/'freeze.json',{'purpose':'same inputs both frozen models; no fit or label change','files':{str(p.relative_to(R)):sha(p) for p in inputs},'production':production,
        'selection_rule':'longest consecutive CONSTANT and DECELERATING run for each vehicle and correctness category; visualize longest four per vehicle at most, distinct cases preferred','limitation':'own-vehicle predictions include fit rows and neighboring fit-context rows; not independent validation','script_sha256':sha(Path(__file__))})
    rows=[]; cache={}; sensors={}; train=[]
    previous=pd.read_csv(B/'cross_vehicle_predictions.csv')
    for c in cases:
        p=R/c['feature_cache'] if c['feature_cache'] else B/(c['id']+'_features.npy')
        a=np.load(p);x=a['features'] if isinstance(a,np.lib.npyio.NpzFile) else a
        d=np.load(B/c['labels_npz']);cache[c['id']]=x;sensors[c['id']]=d
        pred={k:m.predict(x).astype(int) for k,m in models.items()}
        held='civic' if c['role']=='train' else 'rav4'; own='rav4' if c['role']=='train' else 'civic'
        old=previous[previous.id==c['id']].sort_values('sample_index')
        assert np.array_equal(N[pred[held]],old.vehicle_held_out_prediction.to_numpy())
        fit=set(c['training_indices'])
        for i in c['evaluation_indices']:
            y=int(d['accel_candidate'][i]);good={k:int(p[i])==y for k,p in pred.items()}
            status='both_correct' if all(good.values()) else 'both_wrong' if not any(good.values()) else 'rav4_only_correct' if good['rav4'] else 'civic_only_correct'
            rows.append({'id':c['id'],'vehicle':own,'index':i,'truth':y,'rav4':int(pred['rav4'][i]),'civic':int(pred['civic'][i]),'category':status,'exact_fit_row_for_own_model':i in fit,'speed_mps':float(d['speed_smoothed'][i]),'accel_proxy':float(d['acceleration_proxy'][i]),'steering_deg':float(d['steering_smoothed'][i])})
        for i in c['training_indices']:train.append({'vehicle':own,'truth':int(d['accel_candidate'][i]),'speed_mps':float(d['speed_smoothed'][i]),'accel_proxy':float(d['acceleration_proxy'][i]),'steering_deg':float(d['steering_smoothed'][i])})
    s=pd.DataFrame(rows);results={};runs=[]
    for vehicle,g in s.groupby('vehicle'):
        results[vehicle]={'models':{k:metric(g.truth,g[k]) for k in models},'correctness_categories':g.category.value_counts().to_dict(),
            'fit_rows_only':metric(g[g.exact_fit_row_for_own_model].truth,g[g.exact_fit_row_for_own_model][vehicle]),
            'nonfit_rows_same_training_routes':metric(g[~g.exact_fit_row_for_own_model].truth,g[~g.exact_fit_row_for_own_model][vehicle])}
        results[vehicle]['by_truth']={str(N[t]):{'n':len(a),'categories':a.category.value_counts().to_dict()} for t,a in g.groupby('truth')}
    for (id,truth,category),g in s[s.truth.isin([1,2]) & (s.category!='both_correct')].groupby(['id','truth','category']):
        ix=g.sort_values('index')['index'].to_numpy()
        for block in np.split(ix,np.flatnonzero(np.diff(ix)>1)+1):
            if len(block):runs.append({'id':id,'vehicle':g.vehicle.iloc[0],'truth':str(N[truth]),'category':category,'start':int(block[0]),'end':int(block[-1]),'n':len(block),'bin_duration_s':len(block)/10})
    runs.sort(key=lambda a:(-a['n'],a['id'],a['start']))
    selected=[]
    for vehicle in ['rav4','civic']:
        available=[x for x in runs if x['vehicle']==vehicle]; keys=set();ids=set()
        for x in available:
            key=(x['truth'],x['category'])
            if key in keys or x['id'] in ids:continue
            selected.append(x);keys.add(key);ids.add(x['id'])
            if len(ids)==4:break
    # Same public rows are unseen by both checkpoints (but exposed for model development).
    p=pd.read_csv(B/'public_predictions.csv');truth=[list(N).index(t) for t in p.truth]
    results['public']={k:metric(truth,[list(N).index(t) for t in p[col]]) for k,col in [('rav4','expanded_rav4'),('civic','reverse_civic')]}
    results['public']['by_truth']={t:{'n':len(g),'rav4_predictions':g.expanded_rav4.value_counts().to_dict(),'civic_predictions':g.reverse_civic.value_counts().to_dict()} for t,g in p.groupby('truth')}
    # Exact linear margins using a shared centering reference, no feature intervention or retraining.
    ref=np.mean(np.concatenate([cache[c['id']][c['training_indices']] for c in cases]),axis=0)
    for e in selected:
        c=next(c for c in cases if c['id']==e['id']);i=(e['start']+e['end'])//2;x=cache[e['id']][i];d=sensors[e['id']];y=int(d['accel_candidate'][i]);other='civic' if e['vehicle']=='rav4' else 'rav4'
        wrong=int(models[other].predict(x[None])[0]);wrong=wrong if wrong!=y else int(models[e['vehicle']].predict(x[None])[0])
        assert wrong!=y
        e.update(midpoint=i,original_path=c['raw_path'],route=c['route'],license=c['license'],source_url=c['source_url'],truth_type=c['truth_type'],wrong_contrast=str(N[wrong]),models={})
        lo=max(0,i-23);hi=min(len(d['time']),i+24)
        e['sensor_context']={'speed_start':float(d['speed_smoothed'][lo]),'speed_end':float(d['speed_smoothed'][hi-1]),'accel_min':float(np.nanmin(d['acceleration_proxy'][lo:hi])),'accel_max':float(np.nanmax(d['acceleration_proxy'][lo:hi])),'same_proxy_class_fraction':float(np.mean(d['accel_candidate'][lo:hi]==y)),'valid_speed_fraction':float(np.mean(d['speed_valid'][lo:hi])),'context_indices':[lo,hi-1]}
        for name,m in models.items():
            scaler=m.named_steps['standardscaler'];lr=m.named_steps['logisticregression'];weights=lr.coef_/scaler.scale_[None];bias=lr.intercept_-weights@scaler.mean_
            delta=weights[wrong]-weights[y];baseline=float((weights@ref+bias)[wrong]-(weights@ref+bias)[y]);contribution=delta*(x-ref)
            score=m.decision_function(x[None])[0];margin=float(score[wrong]-score[y]);error=float(abs(baseline+contribution.sum()-margin));assert error<1e-5, error
            e['models'][name]={'prediction':str(N[int(m.predict(x[None])[0])]),'probabilities':dict(zip(N.tolist(),map(float,m.predict_proba(x[None])[0]))),'wrong_minus_truth_margin':margin,'float32_scaler_reconstruction_error':error,'common_reference_margin':baseline,'temporal_contributions':dict(zip(['raw','mean5','mean15','mean31','diff5','diff15'],map(float,contribution.reshape(6,144).sum(1))))}
        # Distribution descriptors are diagnostic, not a proof of visual covariate causality.
    distribution={}
    for (v,y),g in pd.DataFrame(train).groupby(['vehicle','truth']):distribution[v+'/'+str(N[y])]={'n':len(g),**{f:{str(q):float(g[f].quantile(q)) for q in [.05,.5,.95]} for f in ['speed_mps','accel_proxy','steering_deg']}}
    s.to_csv(O/'same_input_predictions.csv',index=False);write(O/'results.json',results);write(O/'error_runs.json',runs);write(O/'selected_cases.json',selected);write(O/'training_sensor_distribution.json',distribution)
    assert all(sha(R/p)==h for p,h in production.items())
    print(json.dumps(results,indent=2));print('selected',[(e['id'],e['truth'],e['category'],e['n']) for e in selected])

if __name__=='__main__':
    with threadpool_limits(limits=2):main()
