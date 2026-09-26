"""One frozen classifier-only contrast; existing cached features and baseline fits."""
from pathlib import Path
import sys,time,json,copy
import numpy as np,pandas as pd,joblib
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.pipeline import make_pipeline
from sklearn.metrics import f1_score
from sklearn.utils.class_weight import compute_sample_weight
from threadpoolctl import threadpool_limits
O=Path(__file__).resolve().parent;R=O.parents[2];P=R/'artifacts/stage3_state_learning_20260919'
sys.path.insert(0,str(P));import run as prior
write,read,sha=prior.write,prior.read,prior.sha
PARAMS=dict(max_leaf_nodes=7,max_depth=3,min_samples_leaf=30,max_iter=100,learning_rate=.05,l2_regularization=1,early_stopping=False,class_weight='balanced',random_state=42)
def opposite(y,p):return ((y==0)&(p==1))|((y==1)&(p==0))
def metric(y,p):
    m=prior.metrics(np.asarray(y),np.asarray(p))
    if not m['opposite_denominator']:m['opposite_rate']=None
    return m

def main():
    if (O/'freeze.json').exists():raise FileExistsError('Do not overwrite frozen experiment')
    started=time.perf_counter();cs,_=prior.data('dis');splits=prior.make_splits(cs);old=read(P/'freeze.json')
    protected={**old['inputs'],**old['protected']}
    assert all(sha(R/p)==h for p,h in protected.items())
    sources={};
    for sp in splits:
        source=P/'coverage_control/curated_usable'/sp['name']
        for path in [source/'model.joblib',source/'training_manifest.json',*[source/f'{id}.npz' for id in sp['held']]]:sources[str(path.relative_to(R))]=sha(path)
    write(O/'freeze.json',{'params':PARAMS,'script_sha256':sha(Path(__file__)),'fixed':'cached DIS864; same curated rows2395external+40public perLOVO; saved train-only scaler; balanced class weights, source weights1; same5public/3date splits, labels, evaluation rows and fixed historical steering; no early-stop/threshold/seed search','splits':splits,'protected':protected,'baseline_sources':sources,'evaluation':'Official-formula public development S3; dates external-only proxy, not independent/private. Each date/vehicle reported separately. Baseline all probabilities replay must match exactly.','decision':'Report S3 gain separately from risk gate. A development candidate requires S3 gain>=0.01, zero newly created public opposites, no externalfold F1 decrease, and opposite-rate increase<=0.01 in each fold/vehicle. One setting only; no tuning on outer outcomes. No automatic production promotion.'})
    records=[];fitlog=[]
    hist=pd.read_csv(R/'research/stage3_oof_external.csv');steer={(r.ID,r.sample_index):prior.STEER.index(r.steer_forest) for r in hist.itertuples()}
    for sp in splits:
        source=P/'coverage_control/curated_usable'/sp['name'];selection=read(source/'training_manifest.json')['selection']
        expected=[(id,i) for id in sp['train'] for i in (cs[id]['training_indices'] if cs[id]['public'] else range(0,cs[id]['n'],5)) if cs[id]['y'][i]>=0]
        assert list(map(tuple,selection))==expected
        if sp['name'].startswith('OPEN'):assert len(selection)==2435
        x=np.stack([cs[id]['x'][i] for id,i in selection]);y=np.array([cs[id]['y'][i] for id,i in selection]);assert set(y)==set(range(4))
        reference=joblib.load(source/'model.joblib');sc=reference.named_steps['standardscaler'];z=sc.transform(x)
        assert reference.named_steps['logisticregression'].class_weight=='balanced'
        w=compute_sample_weight('balanced',y);assert np.isclose(w.sum(),len(y))
        t=time.perf_counter();model=make_pipeline(copy.deepcopy(sc),HistGradientBoostingClassifier(**PARAMS))
        # Fit only classifier; do not refit or reweight the reference scaler.
        model[-1].fit(z,y);elapsed=time.perf_counter()-t
        folder=O/sp['name'];folder.mkdir();joblib.dump(model,folder/'hgb.joblib')
        write(folder/'training_manifest.json',{'selection':selection,'class_counts':np.bincount(y,minlength=4).tolist(),'class_weight_factors':(len(y)/(4*np.bincount(y,minlength=4))).tolist(),'scaler_source':str((source/'model.joblib').relative_to(R))})
        fitlog.append({'fold':sp['name'],'n_train':len(y),'fit_seconds':elapsed,'linear_train':metric(y,reference.predict(x)),'hgb_train':metric(y,model.predict(x))})
        for id in sp['held']:
            c=cs[id];base=reference.predict_proba(c['x']);assert np.array_equal(base,np.load(source/f'{id}.npz')['prob'])
            candidate=model.predict_proba(c['x']);assert np.isfinite(candidate).all() and np.allclose(candidate.sum(1),1)
            np.savez_compressed(folder/f'{id}.npz',linear=base,hgb=candidate)
            for i in c['evaluation_indices']:
                r={'fold':sp['name'],'id':id,'vehicle':c['vehicle'],'sample_index':int(i),'time_s':i/10,'truth':int(c['y'][i]),'linear':int(base[i].argmax()),'hgb':int(candidate[i].argmax()),'sensor_accel':float(c['sensor'][i])}
                for name,p in [('linear',base),('hgb',candidate)]:r.update({f'{name}_p{j}':float(p[i,j]) for j in range(4)})
                if c['public']:r.update(steer_truth=int(c['steer'][i]),steer_prediction=steer[id,i])
                records.append(r)
        print(sp['name'],len(y),'fit_seconds',round(elapsed,2),flush=True)
    df=pd.DataFrame(records);baseopp=opposite(df.truth,df.linear);newopp=opposite(df.truth,df.hgb)
    df['new_opposite']=newopp&~baseopp;df['resolved_opposite']=baseopp&~newopp
    df['corrected']=(df.linear!=df.truth)&(df.hgb==df.truth);df['new_wrong']=(df.linear==df.truth)&(df.hgb!=df.truth)
    df['opposite_to_correct']=baseopp&(df.hgb==df.truth);df['opposite_to_other_wrong']=baseopp&~newopp&(df.hgb!=df.truth)
    df.to_csv(O/'predictions.csv',index=False);df[df.hgb!=df.linear].to_csv(O/'changed_predictions.csv',index=False)
    scores=[];changes=[];episodes=[]
    scopes=[('public_oof',df[df.fold.str.startswith('OPEN')]),*list(df.groupby('fold',sort=False))]
    for scope,part in scopes:
        for group,g in [('all',part),*list(part.groupby('vehicle'))] if not scope.startswith('OPEN') and scope!='public_oof' else [('all',part)]:
            for name in ['linear','hgb']:
                m=metric(g.truth,g[name]);r={'variant':name,'scope':scope,'group':group,**m}
                if scope=='public_oof':
                    mask=g.truth!=3;sf=f1_score(g.loc[mask,'steer_truth'],g.loc[mask,'steer_prediction'],labels=range(3),average='macro',zero_division=0);r.update(steer_f1=float(sf),S3=.7*m['macro_f1']+.3*sf)
                scores.append(r)
            changes.append({'scope':scope,'group':group,**{k:int(g[k].sum()) for k in ['new_opposite','resolved_opposite','corrected','new_wrong','opposite_to_correct','opposite_to_other_wrong']}})
    # Group consecutive external evaluation times; these episodes are diagnostics, not independent events.
    for (fold,id),g in df[~df.fold.str.startswith('OPEN')].groupby(['fold','id']):
        for flag in ['new_opposite','resolved_opposite']:
            sub=g[g[flag]].sort_values('sample_index');segments=(sub.sample_index.diff().fillna(999)>5).cumsum()
            for _,e in sub.groupby(segments):episodes.append({'fold':fold,'id':id,'type':flag,'start_s':float(e.time_s.min()),'end_s':float(e.time_s.max()),'n':len(e),'sensor_accel_min':float(e.sensor_accel.min()),'sensor_accel_max':float(e.sensor_accel.max())})
    pd.DataFrame(scores).to_csv(O/'metrics.csv',index=False);write(O/'metrics.json',scores);write(O/'changes.json',changes);pd.DataFrame(episodes).to_csv(O/'episodes.csv',index=False);write(O/'execution.json',fitlog)
    s=pd.DataFrame(scores);public=s[(s.scope=='public_oof')&(s.group=='all')].set_index('variant');gain=float(public.loc['hgb','S3']-public.loc['linear','S3']);assert np.isclose(public.loc['linear','S3'],.808950934648875)
    ext=s[s.scope.str.startswith('date')];pairs=ext.pivot(index=['scope','group'],columns='variant',values=['macro_f1','opposite_rate'])
    gate={'S3_gain_ge_001':gain>=.01,'no_new_public_opposite':int(df.loc[df.fold.str.startswith('OPEN'),'new_opposite'].sum())==0,'external_eachfold_F1_nondecrease':all(pairs.loc[(f,'all'),('macro_f1','hgb')]>=pairs.loc[(f,'all'),('macro_f1','linear')] for f in ['date_1','date_2','date_3']),'each_date_vehicle_opposite_rate_increase_le_001':bool(((pairs['opposite_rate']['hgb']-pairs['opposite_rate']['linear']).dropna()<=.01).all())}
    assert all(sha(R/p)==h for p,h in {**protected,**sources}.items())
    write(O/'decision.json',{'public_S3_gain':gain,'score_improved':gain>0,'development_candidate_gate':gate,'eligible':all(gate.values()),'production_changed':False,'no_further_tuning':True})
    write(O/'checks.json',{'completed':True,'same_rows_scaler_class_weights':True,'baseline_probabilities_exact':True,'inputs_baseline_production_preserved':True,'fits':8,'settings':1,'elapsed_seconds':time.perf_counter()-started})
    print(s.query("group=='all'")[['variant','scope','macro_f1','opposite','S3']].to_string(index=False),flush=True)
if __name__=='__main__':
    with threadpool_limits(limits=2):main()
