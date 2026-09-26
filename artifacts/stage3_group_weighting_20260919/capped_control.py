"""One prospective weight-concentration control after the original experiment."""
from pathlib import Path
import importlib.util,time
import numpy as np,pandas as pd,joblib
from scipy.optimize import brentq
from sklearn.metrics import f1_score
from threadpoolctl import threadpool_limits
HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('group_experiment',HERE/'run.py');g=importlib.util.module_from_spec(spec);spec.loader.exec_module(g)
O=HERE/'capped_control';NAME='vehicle_route_capped'

def cap(cs,sel):
    y,allw=g.weighting(cs,sel);raw=allw['vehicle_route'];w=raw.copy();ext=np.array([not cs[id]['public'] for id,i in sel])
    for c in range(4):
        mask=ext&(y==c)
        if not mask.any():continue
        v=raw[mask];factor=brentq(lambda a:np.clip(a*v,.25,4).sum()-len(v),1e-8,1e8)
        w[mask]=np.clip(factor*v,.25,4)
        assert np.isclose(w[mask].sum(),mask.sum())
    assert np.all(w[~ext]==1) and np.all((w>=.25)&(w<=4)) and np.isclose(w.sum(),len(w))
    return y,w

def main():
    O.mkdir(exist_ok=True)
    if (O/'freeze.json').exists():raise FileExistsError('Preserve previous result')
    f=g.read(HERE/'freeze.json');cs,_=g.prior.data('dis');template=joblib.load(g.prior.B/'expanded_rav4.joblib')['accel']
    g.write(O/'freeze.json',{'source_sha256':g.sha(Path(__file__)),'parent_freeze_sha256':g.sha(HERE/'freeze.json'),
        'timing':'Added after original weighting results; trigger was training-only maximum weight49.375. This is development follow-up, not independent validation.',
        'fixed':'Exactly one extra candidate, vehicle_route weights clipped to [.25,4] with a positive classwise scale solved from training weights to preserve external mass perclass. Public weight1, unweighted scaler, C and all labels/splits unchanged. No bound grid. Reuse all prior inner/outer predictions, fit only the new candidate.',
        'selection':'Same inner3date folds/constraints as parent, now among uniform/vehicle/vehicle_route/half_vehicle_route/vehicle_route_capped. Original selection preserved in parent. No outer score passed to selector.'})
    start=time.perf_counter();newinner=[];newpred=[];selections={};allweights=[]
    oldinner=pd.read_csv(HERE/'inner_metrics.csv');oldpred=pd.read_csv(HERE/'predictions.csv');g.NAMES=g.NAMES+[NAME]
    for sp in f['outer']:
        dst=O/sp['name'];dst.mkdir()
        for ins in f['inner'][sp['name']]:
            sel=g.rows(cs,ins['train']);y,w=cap(cs,sel);x=np.stack([cs[id]['x'][i] for id,i in sel]);m=g.fit(template,x,y,w)
            mf=pd.DataFrame(sel,columns=['id','sample_index']);mf['truth']=y;mf['weight']=w;mf.to_csv(dst/f'{ins["name"]}_weights.csv',index=False)
            scores,_=g.stats(cs,ins['held'],m);newinner.extend({'outer':sp['name'],'inner':ins['name'],'variant':NAME,**r} for r in scores)
        records=oldinner[oldinner.outer==sp['name']].to_dict('records')+[r for r in newinner if r['outer']==sp['name']]
        chosen,audit=g.select(records);selections[sp['name']]={'chosen':chosen,'audit':audit};g.write(dst/'inner_selection.json',selections[sp['name']])
        sel=g.rows(cs,sp['train']);y,w=cap(cs,sel);x=np.stack([cs[id]['x'][i] for id,i in sel]);m=g.fit(template,x,y,w);joblib.dump(m,dst/'model.joblib')
        ref=joblib.load(g.P/'coverage_control/curated_usable'/sp['name']/'model.joblib')
        for attr in ['mean_','scale_']:assert np.array_equal(getattr(m.named_steps['standardscaler'],attr),getattr(ref.named_steps['standardscaler'],attr))
        mf=pd.DataFrame(sel,columns=['id','sample_index']);mf['truth']=y;mf['weight']=w;mf.to_csv(dst/'training_weights.csv',index=False)
        allweights.append({'outer':sp['name'],'min_weight':float(w.min()),'max_weight':float(w.max()),'sum_weight':float(w.sum()),'n':len(w),'effective_sample_n':float(w.sum()**2/(w@w))})
        for id in sp['held']:
            c=cs[id];p=m.predict_proba(c['x']);np.savez_compressed(dst/f'{id}.npz',prob=p)
            for i in c['evaluation_indices']:newpred.append({'variant':NAME,'fold':sp['name'],'id':id,'vehicle':c['vehicle'],'sample_index':i,'truth':int(c['y'][i]),'prediction':int(p[i].argmax()),**{f'p{j}':float(p[i,j]) for j in range(4)}})
        print(sp['name'],'chosen',chosen,flush=True)
    combined=pd.concat([oldpred[oldpred.variant!='selected'],pd.DataFrame(newpred)],ignore_index=True);selected=[]
    for sp in f['outer']:
        a=combined[(combined.fold==sp['name'])&(combined.variant==selections[sp['name']]['chosen'])].copy();a['selected_variant']=a.variant;a['variant']='selected';selected.append(a)
    combined=pd.concat([combined,*selected],ignore_index=True);combined.to_csv(O/'predictions.csv',index=False)
    pd.concat([oldinner,pd.DataFrame(newinner)],ignore_index=True).to_csv(O/'inner_metrics.csv',index=False);g.write(O/'selections.json',selections);pd.DataFrame(allweights).to_csv(O/'weight_checks.csv',index=False)
    hist=pd.read_csv(g.R/'research/stage3_oof_external.csv');lookup={(r.ID,r.sample_index):r for r in hist.itertuples()};results=[]
    for name,z in combined.groupby('variant',sort=False):
        pub=z[z.fold.str.startswith('OPEN')];keep=pub.truth.to_numpy()!=3;truth=np.array([g.prior.STEER.index(lookup[r.id,r.sample_index].steer_label) for r in pub.itertuples()]);pred=np.array([g.prior.STEER.index(lookup[r.id,r.sample_index].steer_forest) for r in pub.itertuples()]);sf=f1_score(truth[keep],pred[keep],labels=range(3),average='macro',zero_division=0);a=g.prior.metrics(pub.truth,pub.prediction)
        results.append({'variant':name,'scope':'public_oof','group':'all',**a,'S3':.7*a['macro_f1']+.3*sf})
        for fold,part in z[~z.fold.str.startswith('OPEN')].groupby('fold'):
            for group,sub in [('all',part),*list(part.groupby('vehicle'))]:results.append({'variant':name,'scope':fold,'group':group,**g.prior.metrics(sub.truth,sub.prediction)})
    pd.DataFrame(results).to_csv(O/'metrics.csv',index=False)
    assert all(g.sha(g.R/p)==h for p,h in {**f['inputs'],**f['protected']}.items())
    g.write(O/'checks.json',{'completed':True,'original_results_inputs_production_unchanged':True,'class_mass_scaler_and_rows_preserved':True,'weights_bounded':True,'elapsed_seconds':time.perf_counter()-start})
    print(pd.DataFrame(results).query("group=='all'")[['variant','scope','macro_f1','opposite','S3']].to_string(index=False),flush=True)

if __name__=='__main__':
    with threadpool_limits(limits=2):main()
