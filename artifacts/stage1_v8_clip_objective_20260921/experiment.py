"""One loss-unit ablation; fixed V8 baseline and no post-result tuning."""
import os
os.environ['OMP_NUM_THREADS']='2'
os.environ['OPENBLAS_NUM_THREADS']='2'
os.environ['HF_HUB_OFFLINE']='1'
os.environ['TRANSFORMERS_OFFLINE']='1'
import collections,json,sys,time,resource
from pathlib import Path
import numpy as np
from scipy.special import expit,log_expit,logsumexp
from scipy.optimize import minimize,check_grad
from sklearn.metrics import f1_score,confusion_matrix
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'scripts/data'))
import experiment_stage1_anchored_head as h

def write(name,data):h.write(OUT/name,data)

def objective(theta,x,y,w,center):
    z=x@theta[:-1]+theta[-1]
    sign=2*y-1
    logq=log_expit(sign[:,None]*z)
    logsum=logsumexp(logq,axis=1)
    delta=theta-center
    loss=-w@(logsum-np.log(x.shape[1]))+.005*(delta@delta)
    dz=-w[:,None]*sign[:,None]*np.exp(logq-logsum[:,None])*(1-expit(sign[:,None]*z))
    grad=np.r_[np.einsum('bv,bvf->f',dz,x),dz.sum()]+.01*delta
    return float(loss),grad

def check_hashes(pro):
    for path,digest in pro['frozen_sha256'].items():assert h.sha(ROOT/path)==digest,path

def metrics(rs,p):
    m=h.metric(rs,p);y=np.array([r['label']=='recapture' for r in rs]);pred=np.array(p)>=.5
    tn,fp,fn,tp=confusion_matrix(y,pred,labels=[False,True]).ravel()
    assert (m['fp'],m['fn'])==(fp,fn)
    if y.any() and (~y).any():assert abs(m['macro_f1']-f1_score(y,pred,average='macro'))<1e-12
    m['class_f1']={name:float(v) for name,v in zip(['original','recapture'],f1_score(y,pred,labels=[False,True],average=None,zero_division=0))}
    return m

def evaluate(domain,split,baseline,candidate):
    x,rs=h.dataset(domain,split);scores={}
    for name,theta in [('v8',baseline),('clip_bce',candidate)]:
        p=expit(x@theta[:-1]+theta[-1]);scores[name]=[float(p[r['offset']:r['offset']+r['count']].mean()) for r in rs]
    report={}
    for name,p in scores.items():
        report[name]={'overall':metrics(rs,p),'modes':{mode:metrics([r for r in rs if r['mode']==mode],[v for r,v in zip(rs,p) if r['mode']==mode]) for mode in sorted({r['mode'] for r in rs})}}
        report[name]['by_group']={g:metrics([r for r in rs if r['group']==g],[v for r,v in zip(rs,p) if r['group']==g]) for g in sorted({r['group'] for r in rs})}
        for key in ['camera','cluster']:
            if all(key in r for r in rs):report[name]['by_'+key]={g:metrics([r for r in rs if r[key]==g],[v for r,v in zip(rs,p) if r[key]==g]) for g in sorted({r[key] for r in rs})}
    rows=[];changes=[]
    for i,r in enumerate(rs):
        row=r|{'scores':{name:p[i] for name,p in scores.items()}};rows.append(row)
        b=scores['v8'][i]>=.5;c=scores['clip_bce'][i]>=.5
        if b!=c:changes.append(row|{'change':'corrected' if c==(r['label']=='recapture') else 'new_error'})
    report['changes']={'corrected':sum(r['change']=='corrected' for r in changes),'new_errors':sum(r['change']=='new_error' for r in changes),'unique_videos_changed':len({r['video'] for r in changes})}
    write(domain+'_'+split+'_predictions.json',rows);write(domain+'_'+split+'_changes.json',changes)
    write(domain+'_'+split+'_summary.json',report)
    return report

def main():
    pro=h.read(OUT/'protocol.json');budget=h.read(OUT/'work_budget.json')
    assert time.time()<budget['deadline_unix'];assert not (OUT/'training.json').exists()
    check_hashes(pro)
    baseline=np.load(h.OUT/'anchored_0.01.npz')['theta'];center=np.load(h.OUT/'v7_theta.npy')
    # Reproduce cached V8 scores before evaluating any candidate.
    replay={}
    for domain in ['vd','dlc','comma']:
        x,rs=h.dataset(domain,'development');p=expit(x@baseline[:-1]+baseline[-1]);old=h.read(h.OUT/f'{domain}_development_predictions.json')
        lookup={(r['video'],r['mode']):r['scores']['anchored_0.01'] for r in old}
        err=max(abs(float(p[r['offset']:r['offset']+r['count']].mean())-lookup[(r['video'],r['mode'])]) for r in rs)
        assert err<1e-6;replay[domain]=err
    arrays=[];ys=[];weights=[];counts={};group_sets={}
    for domain in ['vd','dlc','comma']:
        x,rs=h.dataset(domain,'train');cnt=collections.Counter((r['label'],r['group']) for r in rs)
        unique={label:len({r['group'] for r in rs if r['label']==label}) for label in ['original','recapture']}
        offsets=[]
        for r in rs:
            assert r['count']==24
            sl=slice(r['offset'],r['offset']+24);offsets.extend(range(r['offset'],r['offset']+24));arrays.append(x[sl]);ys.append(int(r['label']=='recapture'))
            weights.append(pro['train_mass'][domain+'_'+r['label']]/unique[r['label']]/cnt[(r['label'],r['group'])])
        assert sorted(offsets)==list(range(len(x)))
        _,dev=h.dataset(domain,'development');train_groups={r['group'] for r in rs};dev_groups={r['group'] for r in dev};assert not train_groups&dev_groups
        if domain=='dlc':assert not {r['cluster'] for r in rs}&{r['cluster'] for r in dev}
        group_sets[domain]={'train':sorted(train_groups),'development':sorted(dev_groups)}
        counts[domain]={'feature_rows':len(x),'condition_records':len(rs),'groups':len(train_groups),'videos':len({r['video'] for r in rs})}
    x=np.stack(arrays);y=np.array(ys);w=np.array(weights)
    assert x.shape==(201,24,512) and np.isfinite(x).all()
    assert np.isclose(w.sum(),1) and np.isclose(w[y==1].sum(),.5)
    # Same reconstructed per-view objective must have the recorded V8 stationary point.
    oldloss,oldgrad=h.objective(baseline,x.reshape(-1,512),np.repeat(y,24),np.repeat(w/24,24),center,.01)
    expected=h.read(h.OUT/'training.json')['models']['anchored_0.01']['loss']
    assert abs(oldloss-expected)<1e-10 and abs(oldgrad).max()<2e-6
    rng=np.random.default_rng(20260921);small=rng.normal(size=(4,3,5));sy=np.array([0,1,0,1]);sw=np.array([.1,.2,.3,.4]);st=rng.normal(size=6);sc=rng.normal(size=6)
    ge=check_grad(lambda a:objective(a,small,sy,sw,sc)[0],lambda a:objective(a,small,sy,sw,sc)[1],st)
    assert ge<1e-6
    # Independent torch autodiff on synthetic inputs only, not a second training run.
    import torch
    tt=torch.tensor(st,requires_grad=True);tx=torch.tensor(small);ty=torch.tensor(sy);tw=torch.tensor(sw);tc=torch.tensor(sc)
    means=torch.sigmoid(tx@tt[:-1]+tt[-1]).mean(1)
    loss=(-tw*(ty*torch.log(means)+(1-ty)*torch.log1p(-means))).sum()+.005*((tt-tc)**2).sum();loss.backward()
    av,ag=objective(st,small,sy,sw,sc);assert abs(av-loss.item())<1e-12 and np.max(abs(ag-tt.grad.numpy()))<1e-12
    write('preflight.json',{'baseline_replay_error':replay,'old_objective_loss_reproduced':oldloss,'old_gradient_max':float(abs(oldgrad).max()),'gradient_finite_difference_error':float(ge),'torch_autograd_pass':True,'train_counts':counts,'split_groups':group_sets,'total_mass':float(w.sum()),'positive_mass':float(w[y==1].sum()),'all_checks_pass':True})
    start=time.monotonic();calls=0
    def bounded(theta):
        nonlocal calls
        calls+=1
        if time.monotonic()-start>180 or time.time()>budget['deadline_unix']:raise TimeoutError('Frozen time cap')
        return objective(theta,x,y,w,center)
    opts={k:v for k,v in pro['optimizer'].items() if k!='method'}
    result=minimize(bounded,center.copy(),jac=True,method='L-BFGS-B',options=opts)
    train_time=time.monotonic()-start
    assert result.success and np.isfinite(result.x).all(),str(result.message)
    np.savez(OUT/'candidate.npz',theta=result.x)
    write('training.json',{'success':True,'runs':1,'iterations':int(result.nit),'function_calls':calls,'seconds':train_time,'loss':float(result.fun),'gradient_max':float(abs(result.jac).max()),'distance_to_v7':float(np.linalg.norm(result.x-center)),'distance_to_v8':float(np.linalg.norm(result.x-baseline)),'sha256':h.sha(OUT/'candidate.npz'),'objective_at_v8':objective(baseline,x,y,w,center)[0],'nonconvex_global_optimum_not_proven':True})
    evaluation_start=time.monotonic()
    reports={d:evaluate(d,'development',baseline,result.x) for d in ['vd','dlc','comma']}
    for d in ['vd','dlc','comma']:evaluate(d,'train',baseline,result.x)
    reasons=[]
    primary={n:float(np.mean([reports[d][n]['modes']['full_frame']['macro_f1'] for d in ['vd','dlc']])) for n in ['v8','clip_bce']}
    if primary['clip_bce']<=primary['v8']:reasons.append('No strict full-frame primary improvement')
    for d,r in reports.items():
        for mode in h.MODES:
            for k in (['fp'] if d=='comma' else ['fp','fn']):
                b=r['v8']['modes'][mode][k];c=r['clip_bce']['modes'][mode][k]
                if c>b:reasons.append(f'{d}/{mode}/{k} increased {b}->{c}')
    development_pass=not reasons;guards={}
    if development_pass:
        for d,s in [('vd','new_val'),('vd','new_tcl'),('public','guard')]:
            rr=evaluate(d,s,baseline,result.x);guards[d+'_'+s]=rr
            for k in ['fp','fn']:
                if rr['clip_bce']['overall'][k]>rr['v8']['overall'][k]:reasons.append(f'{d}_{s}/{k} increased')
            if d=='public':
                newfps=[r for r in h.read(OUT/f'{d}_{s}_changes.json') if r['change']=='new_error' and r['label']=='original']
                if newfps:reasons.append('New public original FP')
    check_hashes(pro)
    decision={'development_gate_pass':development_pass,'all_gates_pass':not reasons,'decision':'KEEP_FOR_ADDITIONAL_CONFIRMATION' if not reasons else 'REJECT','reasons':reasons,'primary_full_frame_domain_average':primary,'development':reports,'guards':guards,'guard_candidate_scored':bool(guards),'unseen_confirmation':'Not available; reserved incomplete data not scored','evaluation_seconds':time.monotonic()-evaluation_start,'peak_process_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'max_candidate_runs':1,'original_hashes_unchanged':True,'official_candidate_score':'Not measured','no_submission_or_model_replacement':True}
    write('decision.json',decision);print(json.dumps({k:v for k,v in decision.items() if k not in ['development','guards']},indent=2),flush=True)

if __name__=='__main__':main()
