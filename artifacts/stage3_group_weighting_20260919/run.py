"""Change source weights only; frozen DIS/labels/scaler/class mass/model recipe."""
from pathlib import Path
import sys,json,hashlib,time,warnings
import numpy as np,pandas as pd,joblib
from sklearn.base import clone
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import f1_score
from threadpoolctl import threadpool_limits
O=Path(__file__).resolve().parent;R=O.parents[1];P=R/'artifacts/stage3_state_learning_20260919'
sys.path.insert(0,str(P))
import run as prior
NAMES=['uniform','vehicle','vehicle_route','half_vehicle_route']
read=prior.read;write=prior.write;sha=prior.sha

def rows(cs,ids):
    return [(id,i) for id in ids for i in (cs[id]['training_indices'] if cs[id]['public'] else range(0,cs[id]['n'],5)) if cs[id]['y'][i]>=0]

def weighting(cs,sel):
    y=np.array([cs[id]['y'][i] for id,i in sel]);ext=np.array([not cs[id]['public'] for id,i in sel])
    vehicle=np.array([cs[id]['vehicle'] for id,i in sel]);route=np.array([cs[id].get('route',id) for id,i in sel])
    w={name:np.ones(len(sel)) for name in NAMES}
    for label in range(4):
        cm=(y==label)&ext;n=int(cm.sum());vs=np.unique(vehicle[cm])
        for v in vs:
            vm=cm&(vehicle==v);rs=np.unique(route[vm]);w['vehicle'][vm]=n/(len(vs)*vm.sum())
            for r in rs:
                rm=vm&(route==r);w['vehicle_route'][rm]=n/(len(vs)*len(rs)*rm.sum())
    w['half_vehicle_route']=(1+w['vehicle_route'])/2
    for name,z in w.items():
        assert np.isfinite(z).all() and (z>0).all() and np.all(z[~ext]==1)
        for label in range(4):assert np.isclose(z[(y==label)&ext].sum(),((y==label)&ext).sum())
        assert np.isclose(z.sum(),len(sel))
    return y,w

def inner_splits(cs,sp):
    by={}
    for id in sp['train']:
        if not cs[id]['public']:by.setdefault(cs[id]['vehicle'],set()).add(cs[id]['route'].split('|')[1][:10])
    groups=[set() for _ in range(3)]
    for v,days in sorted(by.items()):
        days=sorted(days,key=lambda d:hashlib.sha256(('s3-weight-inner:'+v+d).encode()).hexdigest())
        for j,day in enumerate(days):groups[j%3].add((v,day))
    result=[]
    for k,g in enumerate(groups):
        held=[id for id in sp['train'] if not cs[id]['public'] and (cs[id]['vehicle'],cs[id]['route'].split('|')[1][:10]) in g]
        train=[id for id in sp['train'] if id not in held];sel=rows(cs,train)
        assert set(held).isdisjoint(sp['held']) and set(held).isdisjoint(train)
        assert set(cs[id]['y'][i] for id,i in sel)==set(range(4))
        result.append({'name':f'inner_{k+1}','train':train,'held':held})
    return result

def stats(cs,held,model):
    rec=[]
    for id in held:
        c=cs[id];indices=c['evaluation_indices'];p=model.predict(c['x'][indices])
        rec.extend({'id':id,'vehicle':c['vehicle'],'sample_index':i,'truth':int(c['y'][i]),'prediction':int(a)} for i,a in zip(indices,p))
    df=pd.DataFrame(rec);out=[]
    for group,g in [('all',df),*list(df.groupby('vehicle'))]:out.append({'group':group,**prior.metrics(g.truth,g.prediction)})
    return out,df

def fit(template,x,y,w):
    m=clone(template)
    with warnings.catch_warnings(record=True) as ws:
        warnings.simplefilter('always');m.fit(x,y,logisticregression__sample_weight=w)
    assert not any(issubclass(a.category,ConvergenceWarning) for a in ws)
    return m

def select(inner):
    d=pd.DataFrame(inner);b=d[d.variant=='uniform'];audit=[]
    for name in NAMES:
        q=d[d.variant==name];vehicles=q[q.group!='all'];bv=b[b.group!='all']
        gain=float(vehicles.macro_f1.mean()-bv.macro_f1.mean())
        risk=[]
        for v,g in vehicles.groupby('group'):
            bg=bv[bv.group==v];risk.append(float(g.opposite_rate.mean()-bg.opposite_rate.mean()))
        foldg=[]
        for k,g in q[q.group=='all'].groupby('inner'):
            foldg.append(float(g.macro_f1.iloc[0]-b[(b.inner==k)&(b.group=='all')].macro_f1.iloc[0]))
        eligible=(gain>=-1e-12 and max(risk)<=.01 and min(foldg)>=-.02)
        audit.append({'variant':name,'mean_vehicle_fold_F1_gain':gain,'max_vehicle_mean_opposite_rate_increase':max(risk),'worst_fold_F1_delta':min(foldg),'eligible':bool(eligible)})
    # Stable ties favor original uniform weights. No outer/public score enters this function.
    chosen=max([a for a in audit if a['eligible']],key=lambda a:a['mean_vehicle_fold_F1_gain'])['variant']
    return chosen,audit

def main():
    if (O/'freeze.json').exists():raise FileExistsError('Do not overwrite this frozen experiment')
    start=time.perf_counter();cs,pub=prior.data('dis');splits=prior.make_splits(cs);old=read(P/'freeze.json')
    assert all(sha(R/p)==h for p,h in {**old['inputs'],**old['protected']}.items())
    frozen={'source_sha256':sha(Path(__file__)),'fixed':'same curated usable2Hz rows (2395external+40public perLOVO), DIS864, existing unweighted scaler, LogisticRegression C=.03/class_weight=balanced, source labels and evaluation masks; no clipping, new features, labels, downloads, production change',
        'variants':{'uniform':'old weights1','vehicle':'within each external class equal vehicle loss mass; n_class/(V_class*n_vehicle_class)',
                    'vehicle_route':'within each external class equal vehicle and then route mass; n_class/(V_class*R_vehicle_class*n_route_class)',
                    'half_vehicle_route':'0.5*uniform +0.5*vehicle_route; fixed before results'},
        'controls':'external mass per class, total sample weight, public weight1 and class_weight factors remain equal; scaler is always unweighted on exactly same rows; no vehicle-only resampling or hidden regularization change',
        'inner_selection':'3hash-distributed vehicle-date folds within each outertrain, whole routes stay together; maximize mean vehicle/fold strict proxy4classMacroF1 subject to no mean vehicle opposite-rate increase>.01 and no innerfold F1 drop>.02, averageF1 nondecrease; tie uniform; no outerlabels/publicheld labels; no threshold grid or new candidate after results',
        'partial_gate':'publicS3>=expanded0.808950934648875; publicopposites<=2, no new public opposite from previously correct; each externalfoldF1>=expanded, each externalfold/vehicle opposite count<=expanded, date1 Civic count<275',
        'promotion_gate':'Additionally meet previous strict-reference gates: publicS3>=strict+.01, no new public opposites vs strict, each externalfoldF1>=strict, eachfold/vehicle opposite-rate<=strict+.01. Fixed candidates are diagnostics; only train-only selected policy eligible for promotion.',
        'limitations':'all sources previously development exposed; official50 sparse labels +fixed historical steer47OOF; no private/independent score; repeated outerdates not independent; inner usesproxy, may not represent official labels',
        'inputs':{**old['inputs'],str((P/'run.py').relative_to(R)):sha(P/'run.py')},'protected':old['protected'],
        'outer':splits,'inner':{sp['name']:inner_splits(cs,sp) for sp in splits}}
    write(O/'freeze.json',frozen)
    template=joblib.load(prior.B/'expanded_rav4.joblib')['accel'];selections={};allinner=[];allpred=[];timings=[];mass=[]
    for sp in splits:
        folder=O/sp['name'];folder.mkdir();inner=[]
        for ins in frozen['inner'][sp['name']]:
            sel=rows(cs,ins['train']);y,w=weighting(cs,sel);x=np.stack([cs[id]['x'][i] for id,i in sel]);refsc=None
            manifest=pd.DataFrame(sel,columns=['id','sample_index']);manifest['truth']=y
            for name,z in w.items():manifest[name]=z
            manifest.to_csv(folder/f'{ins["name"]}_train_weights.csv',index=False)
            for name in NAMES:
                m=fit(template,x,y,w[name]);sc=m.named_steps['standardscaler']
                if refsc is None:refsc=(sc.mean_,sc.scale_)
                assert np.array_equal(sc.mean_,refsc[0]) and np.array_equal(sc.scale_,refsc[1])
                measures,_=stats(cs,ins['held'],m)
                inner.extend({'outer':sp['name'],'inner':ins['name'],'variant':name,**a} for a in measures)
        chosen,audit=select(inner);selections[sp['name']]={'chosen':chosen,'audit':audit};allinner.extend(inner)
        # Save selection before any outer evaluation for this split.
        write(folder/'inner_selection.json',selections[sp['name']])
        sel=rows(cs,sp['train']);y,w=weighting(cs,sel);x=np.stack([cs[id]['x'][i] for id,i in sel])
        source=P/'coverage_control/curated_usable'/sp['name']
        assert sel==[tuple(a) for a in read(source/'training_manifest.json')['selection']]
        mf=pd.DataFrame(sel,columns=['id','sample_index']);mf['truth']=y;mf['vehicle']=[cs[id]['vehicle'] for id,i in sel]
        for name,z in w.items():mf[name]=z
        mf.to_csv(folder/'training_weights.csv',index=False)
        ref=joblib.load(source/'model.joblib');refsc=ref.named_steps['standardscaler']
        for name in NAMES:
            t=time.perf_counter();m=ref if name=='uniform' else fit(template,x,y,w[name]);sc=m.named_steps['standardscaler']
            assert np.array_equal(sc.mean_,refsc.mean_) and np.array_equal(sc.scale_,refsc.scale_)
            if name!='uniform':joblib.dump(m,folder/f'{name}.joblib')
            mass.append({'outer':sp['name'],'variant':name,'train_n':len(sel),'sum_weight':float(w[name].sum()),'min_weight':float(w[name].min()),'max_weight':float(w[name].max()),'effective_sample_n':float(w[name].sum()**2/(w[name]@w[name]))})
            for id in sp['held']:
                c=cs[id];prob=m.predict_proba(c['x']);indices=c['evaluation_indices']
                if name=='uniform':assert np.array_equal(prob,np.load(source/f'{id}.npz')['prob'])
                np.savez_compressed(folder/f'{name}_{id}.npz',prob=prob)
                for i in indices:
                    row={'variant':name,'fold':sp['name'],'id':id,'vehicle':c['vehicle'],'sample_index':i,'truth':int(c['y'][i]),'prediction':int(prob[i].argmax()),**{f'p{j}':float(prob[i,j]) for j in range(4)}}
                    allpred.append(row)
                    if name==chosen:allpred.append({**row,'variant':'selected','selected_variant':chosen})
            timings.append({'fold':sp['name'],'variant':name,'seconds':time.perf_counter()-t})
        print(sp['name'],'chosen',chosen,'train',len(sel),flush=True)
    df=pd.DataFrame(allpred);df.to_csv(O/'predictions.csv',index=False);pd.DataFrame(allinner).to_csv(O/'inner_metrics.csv',index=False)
    pd.DataFrame(mass).to_csv(O/'weight_checks.csv',index=False);write(O/'selections.json',selections)
    # Official steering stays fixed, reusing same historical out-of-fold predictions.
    hist=pd.read_csv(R/'research/stage3_oof_external.csv');labels={r.ID:{int(a.sample_index):a for a in g.itertuples()} for r,g in []} if False else {(r.ID,r.sample_index):r for r in hist.itertuples()}
    metrics=[]
    for name,g in df.groupby('variant',sort=False):
        pg=g[g.fold.str.startswith('OPEN')];yp=[prior.STEER.index(labels[r.id,r.sample_index].steer_label) for r in pg.itertuples()];pp=[prior.STEER.index(labels[r.id,r.sample_index].steer_forest) for r in pg.itertuples()];mask=pg.truth.to_numpy()!=3
        sf=f1_score(np.array(yp)[mask],np.array(pp)[mask],labels=range(3),average='macro',zero_division=0);acc=prior.metrics(pg.truth,pg.prediction)
        metrics.append({'variant':name,'scope':'public_oof','group':'all',**acc,'steer_F1':float(sf),'S3':.7*acc['macro_f1']+.3*sf})
        for fold,part in g[~g.fold.str.startswith('OPEN')].groupby('fold'):
            for group,sub in [('all',part),*list(part.groupby('vehicle'))]:metrics.append({'variant':name,'scope':fold,'group':group,**prior.metrics(sub.truth,sub.prediction)})
    pd.DataFrame(metrics).to_csv(O/'metrics.csv',index=False);write(O/'metrics.json',metrics)
    assert all(sha(R/p)==h for p,h in {**frozen['inputs'],**frozen['protected']}.items())
    write(O/'checks.json',{'completed':True,'elapsed_seconds':time.perf_counter()-start,'baseline_all_probabilities_exact':True,'same_rows_and_scaler':True,'external_per_class_and_total_mass_equal':True,'inner_route_vehicle_date_disjoint':True,'publicheld_outerheld_not_used_in_selection':True,'inputs_production_unchanged':True,'execution':timings})
    print(pd.DataFrame(metrics).query("group=='all'")[['variant','scope','macro_f1','opposite','S3']].to_string(index=False),flush=True)

if __name__=='__main__':
    with threadpool_limits(limits=2):main()
