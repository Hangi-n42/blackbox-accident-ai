"""One frozen nested-OOF class-bias control; no production modifications."""
from pathlib import Path
import sys, json, hashlib, time, warnings
import numpy as np
import pandas as pd
import joblib
from scipy.optimize import minimize, check_grad
from scipy.special import logsumexp, softmax
from sklearn.base import clone
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import f1_score
from threadpoolctl import threadpool_limits

O = Path(__file__).resolve().parent
R = O.parents[1]
P = R / 'artifacts/stage3_state_learning_20260919'
sys.path.insert(0, str(P))
import run as prior

LAMBDA = 1.0  # Fixed mean NLL + 0.5 * lambda * sum(bias**2); no sweep.
read, write, sha = prior.read, prior.write, prior.sha


def selection(cs, ids):
    return [(id, i) for id in ids
            for i in (cs[id]['training_indices'] if cs[id]['public'] else range(0, cs[id]['n'], 5))
            if cs[id]['y'][i] >= 0]


def objective(u, logits, y):
    b = np.r_[u, -np.sum(u)]
    z = logits + b
    loss = np.mean(logsumexp(z, axis=1) - z[np.arange(len(y)), y]) + .5 * LAMBDA * (b @ b)
    error = softmax(z, axis=1)
    error[np.arange(len(y)), y] -= 1
    grad_b = error.mean(axis=0) + LAMBDA * b
    return loss, grad_b[:3] - grad_b[3]


def metrics(y, pred):
    m = prior.metrics(np.asarray(y), np.asarray(pred))
    if not m['opposite_denominator']:
        m['opposite_rate'] = None
    m['support'] = np.bincount(y, minlength=4).tolist()
    return m


def main():
    start = time.perf_counter()
    assert not (O / 'freeze.json').exists(), 'Do not overwrite this experiment'
    cs, _ = prior.data('dis')
    public = [id for id, c in cs.items() if c['public']]
    splits = prior.make_splits(cs)
    old = read(P / 'freeze.json')
    paths = {**old['inputs'], **old['protected']}
    assert all(sha(R / path) == h for path, h in paths.items())
    paths[str((P / 'run.py').relative_to(R))] = sha(P / 'run.py')
    paths[str((prior.B / 'expanded_rav4.joblib').relative_to(R))] = sha(prior.B / 'expanded_rav4.joblib')
    for id in public:
        path = P / 'coverage_control/curated_usable' / id / 'model.joblib'
        paths[str(path.relative_to(R))] = sha(path)
    freeze = {
        'source_sha256': sha(Path(__file__)), 'lambda': LAMBDA,
        'objective': 'mean unweighted official-label NLL + .5 * lambda * ||b||^2; b=(u0,u1,u2,-sum(u)); 3DOF; BFGS zero init, gtol1e-9, maxiter1000; accept precision-loss status only if max_abs_gradient<=1e-7; no temperature, clipping, class balancing, smoothing or search',
        'fixed': 'DIS864, external curated usable2Hz labels, unweighted StandardScaler, balanced LogisticRegression C=.03/max_iter2000/random_state42, historical steer OOF unchanged',
        'public': 'outerLOVO10 held; internalLOVO among other4videos: external2395+3public30 train;40 OOF logits calibrate; cached outer external2395+40 model reused',
        'date': 'external held date excluded from ALL base fits; base trains external date train + all50 public;5 inner public LOVO generate50 OOF calibration logits with same external date train. Compare raw vs calibrated SAME base. Not comparable to previous external-only model as same training condition.',
        'candidate_gate': 'public S3 strictly improves; no new public opposite; at least2 public videos have higher accel macroF1; each date macroF1 nondecrease; each date/vehicle opposite-rate increase<=.01. Passing is development candidate only, no production promotion.',
        'limitations': 'all public50 and external23 previously development exposed; dates overlap; public-external source overlap beyond excluded known match not established; no private or independent generalization claim; zero reversal denominator NA',
        'paths': paths, 'splits': splits,
    }
    write(O / 'freeze.json', freeze)  # Before any new candidate output.
    z = np.array([[1., -.2, .4, 0.], [-1., 2., .1, .2]])
    y = np.array([0, 2])
    err = check_grad(lambda u: objective(u,z,y)[0], lambda u: objective(u,z,y)[1], np.array([.1,-.1,.2]))
    assert err < 1e-6, err
    template = joblib.load(prior.B / 'expanded_rav4.joblib')['accel']
    models = {}; training_log = []; predrows = []; calibration_log = []

    def fit(ids):
        key = tuple(ids)
        if key in models:
            return models[key]
        selected = selection(cs, ids)
        x = np.stack([cs[id]['x'][i] for id, i in selected])
        labels = np.array([cs[id]['y'][i] for id, i in selected])
        assert set(labels) == set(range(4))
        name = hashlib.sha256(json.dumps(ids).encode()).hexdigest()[:16]
        folder = O / 'base_models' / name
        if folder.exists():
            saved=read(folder/'training_manifest.json')
            assert saved['ids']==ids and [tuple(a) for a in saved['selection']]==selected
            model=joblib.load(folder/'model.joblib')
            assert model.get_params()['logisticregression__C']==.03
            training_log.append({'key':name,'rows':len(selected),'reused_after_precision_stop':True,'model_sha256':sha(folder/'model.joblib')})
            models[key]=(model,name)
            return model,name
        folder.mkdir(parents=True)
        write(folder / 'training_manifest.json', {'ids': ids, 'selection': selected, 'class_counts': np.bincount(labels, minlength=4).tolist()})
        model = clone(template); t = time.perf_counter()
        with warnings.catch_warnings(record=True) as ws:
            warnings.simplefilter('always')
            model.fit(x, labels, logisticregression__sample_weight=np.ones(len(labels)))
        assert not any(issubclass(w.category, ConvergenceWarning) for w in ws)
        assert np.array_equal(model.classes_, np.arange(4))
        joblib.dump(model, folder / 'model.joblib')
        training_log.append({'key': name, 'rows': len(selected), 'seconds': time.perf_counter()-t, 'iterations': model[-1].n_iter_.tolist()})
        models[key] = (model, name)
        return model, name

    for sp in splits:
        folder = O / sp['name']; folder.mkdir()
        ext = [id for id in sp['train'] if not cs[id]['public']]
        pubs = [id for id in public if id not in sp['held']]
        assert set(ext + pubs).isdisjoint(sp['held'])
        records = []; insplits = []
        for held in pubs:
            train = ext + [id for id in pubs if id != held]
            assert set(train).isdisjoint(sp['held'] + [held])
            base, key = fit(train)
            ix = cs[held]['evaluation_indices']
            logits = base.decision_function(cs[held]['x'][ix])
            assert np.allclose(softmax(logits,axis=1), base.predict_proba(cs[held]['x'][ix]),atol=1e-12)
            records.extend({'id': held, 'sample_index': i, 'truth': int(cs[held]['y'][i]), 'model': key,
                            **{f'z{j}': float(logit[j]) for j in range(4)}} for i, logit in zip(ix, logits))
            insplits.append({'held': held, 'train': train, 'model': key})
        frame = pd.DataFrame(records)
        assert len(frame) == len(pubs)*10 and not frame.duplicated(['id','sample_index']).any()
        logits = frame[[f'z{j}' for j in range(4)]].to_numpy()
        labels = frame.truth.to_numpy()
        result = minimize(objective, np.zeros(3), args=(logits, labels), jac=True,
                          method='BFGS', options={'gtol':1e-9, 'maxiter':1000})
        assert result.success or (result.status==2 and np.abs(result.jac).max()<=1e-7), result.message
        bias = np.r_[result.x, -result.x.sum()]
        assert np.isfinite(bias).all() and abs(bias.sum()) < 1e-12
        frame.to_csv(folder / 'calibration_oof.csv', index=False)
        audit = {'fold':sp['name'], 'bias':bias.tolist(), 'class_counts':np.bincount(labels,minlength=4).tolist(),
                 'objective_before':float(objective(np.zeros(3),logits,labels)[0]), 'objective_after':float(result.fun),
                 'gradient_max':float(np.abs(result.jac).max()), 'iterations':int(result.nit),
                 'optimizer_status':int(result.status),'optimizer_message':str(result.message),'inner':insplits}
        assert audit['objective_after'] <= audit['objective_before'] + 1e-12
        write(folder/'calibrator.json',audit); calibration_log.append(audit)
        # Freeze each learned bias before evaluating its outer cases.
        if sp['name'].startswith('OPEN'):
            source = P/'coverage_control/curated_usable'/sp['name']
            assert selection(cs,ext+pubs) == [tuple(a) for a in read(source/'training_manifest.json')['selection']]
            base = joblib.load(source/'model.joblib')
            write(folder/'base_reference.json',{'path':str((source/'model.joblib').relative_to(R)), 'sha256':sha(source/'model.joblib')})
        else:
            base,key = fit(ext+pubs)
            write(folder/'base_reference.json',{'model':key})
        for id in sp['held']:
            c=cs[id]; ix=np.array(c['evaluation_indices'])
            logits=base.decision_function(c['x'])[ix]; raw=base.predict_proba(c['x'])[ix]; calibrated=softmax(logits+bias,axis=1)
            assert np.allclose(softmax(logits,axis=1),raw,atol=1e-12)
            if sp['name'].startswith('OPEN'):
                assert np.array_equal(raw,np.load(source/f'{id}.npz')['prob'][ix])
            np.savez_compressed(folder/f'{id}.npz',sample_index=ix,logits=logits,raw=raw,calibrated=calibrated)
            for k,i in enumerate(ix):
                predrows.append({'fold':sp['name'],'id':id,'sample_index':int(i),'vehicle':c['vehicle'],
                                 'truth':int(c['y'][i]),'raw':int(raw[k].argmax()),'calibrated':int(calibrated[k].argmax()),
                                 **{f'raw_p{j}':float(raw[k,j]) for j in range(4)},
                                 **{f'cal_p{j}':float(calibrated[k,j]) for j in range(4)}})
        print(sp['name'],'calibration_n',len(frame),'bias',np.round(bias,4).tolist(),flush=True)
    df=pd.DataFrame(predrows)
    opposite=lambda pred: ((df.truth==0)&(pred==1))|((df.truth==1)&(pred==0))
    before=opposite(df.raw); after=opposite(df.calibrated)
    df['new_opposite']=after&~before;df['fixed_opposite']=before&~after
    df['new_wrong']=(df.raw==df.truth)&(df.calibrated!=df.truth)
    df['recovered_correct']=(df.raw!=df.truth)&(df.calibrated==df.truth)
    df.to_csv(O/'predictions.csv',index=False)
    df[df.raw!=df.calibrated].to_csv(O/'changes.csv',index=False)
    hist=pd.read_csv(R/'research/stage3_oof_external.csv')
    lookup={(r.ID,r.sample_index):r for r in hist.itertuples()}
    scores=[]
    for name in ['raw','calibrated']:
        pg=df[df.fold.str.startswith('OPEN')]
        y=[prior.STEER.index(lookup[r.id,r.sample_index].steer_label) for r in pg.itertuples()]
        pr=[prior.STEER.index(lookup[r.id,r.sample_index].steer_forest) for r in pg.itertuples()]
        mask=pg.truth.to_numpy()!=3
        sf=float(f1_score(np.array(y)[mask],np.array(pr)[mask],labels=range(3),average='macro',zero_division=0))
        a=metrics(pg.truth,pg[name]);scores.append({'variant':name,'scope':'public_oof','group':'all',**a,'S3':.7*a['macro_f1']+.3*sf,'steer_F1':sf})
        for fold,g in df.groupby('fold'):
            groups=[('all',g)] if fold.startswith('OPEN') else [('all',g),*list(g.groupby('vehicle'))]
            for group,s in groups:
                scores.append({'variant':name,'scope':fold,'group':group,**metrics(s.truth,s[name])})
    write(O/'metrics.json',scores);pd.DataFrame(scores).to_csv(O/'metrics.csv',index=False)
    write(O/'execution.json',training_log)
    assert all(sha(R/path)==h for path,h in paths.items())
    assert sha(Path(__file__))==freeze['source_sha256']
    write(O/'checks.json',{'completed':True,'inputs_and_production_unchanged':True,'source_frozen':True,
                           'public_baseline_probabilities_exact':True,'nested_held_sources_excluded':True,
                           'zero_bias_matches_baseline':True,'calibration_gradient_check_error':float(err),
                           'base_fits_converged':True,'bias_stationarity_verified':True,'unique_base_models':len(training_log),
                           'reused_after_precision_stop':sum(a.get('reused_after_precision_stop',False) for a in training_log),
                           'cached_public_outer_models':5,'elapsed_seconds':time.perf_counter()-start})
    print(pd.DataFrame(scores).query("group=='all'")[['variant','scope','macro_f1','opposite','S3']].to_string(index=False),flush=True)


if __name__=='__main__':
    with threadpool_limits(limits=2):
        main()
