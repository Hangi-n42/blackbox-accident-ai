"""Small shared pairwise entry ranker; fixed candidates, frozen visual encoder."""
from fractions import Fraction as F
import numpy as np

PCA_DIM=4
L2=1.0
PAIR_EPS=F(1,10**9)

def pool_tiles(hidden):
    assert hidden.ndim==2 and hidden.shape[0]==1152
    grid=hidden.reshape(24,48,-1)
    # Omit each tile's first 32-pixel row (frame-number header); global attention can still carry header information.
    return np.stack([grid[(i//4)*8+1:(i//4+1)*8,(i%4)*12:(i%4+1)*12].mean((0,1)) for i in range(12)])

def certain_pairs(times,lo,hi):
    times=[F(str(t)) for t in times];lo,hi=F(str(lo)),F(str(hi));assert lo<=hi
    result=[]
    for i,a in enumerate(times):
        for j,b in enumerate(times):
            if i==j:continue
            points={lo,hi,*[p for p in [a,b] if lo<=p<=hi]}
            # Both errors use the SAME unknown true instant, not independent ends of error intervals.
            if max(abs(a-t)-abs(b-t) for t in points)<-PAIR_EPS:result.append((i,j))
    return np.array(result,dtype=int).reshape(-1,2)

def project(embeddings,mean,components):
    norms=np.linalg.norm(embeddings,axis=2,keepdims=True)
    return (embeddings/np.maximum(norms,1e-12)-mean)@components.T

def temporal(z,times):
    assert z.shape==(12,PCA_DIM) and len(times)==12
    previous=np.vstack([z[:1],z[:-1]]);following=np.vstack([z[1:],z[-1:]])
    times=np.asarray(times);relative=(times-times[0])/max(times[-1]-times[0],1e-12)
    return np.column_stack([z,z-previous,following-z,relative,np.arange(12)==0])

def design(embeddings,times,state):
    z=project(embeddings,state['embedding_mean'],state['components'])
    raw=np.stack([temporal(x,t) for x,t in zip(z,times)])
    return (raw-state['feature_mean'])/state['feature_scale']

def fit(embeddings,times,references):
    from scipy.optimize import minimize
    assert len(embeddings)==len(times)==len(references)
    norm=embeddings/np.maximum(np.linalg.norm(embeddings,axis=2,keepdims=True),1e-12)
    mean=norm.mean((0,1));_,singular,vt=np.linalg.svd((norm-mean).reshape(-1,norm.shape[-1]),full_matrices=False)
    assert len(singular)>=PCA_DIM and singular[PCA_DIM-1]>1e-10
    state={'embedding_mean':mean,'components':vt[:PCA_DIM]}
    projected=project(embeddings,mean,state['components'])
    raw=np.stack([temporal(x,t) for x,t in zip(projected,times)])
    state.update(feature_mean=raw.mean((0,1)),feature_scale=np.maximum(raw.std((0,1)),1e-6))
    x=design(embeddings,times,state);pairs=[certain_pairs(t,*r) for t,r in zip(times,references)]
    assert all(len(p) for p in pairs),'No certain training pairs; do not force targets'
    diffs=[a[p[:,0]]-a[p[:,1]] for a,p in zip(x,pairs)]
    def objective(w):
        loss=L2*np.dot(w,w)/2;grad=L2*w.copy()
        for d in diffs:
            margin=d@w;loss+=np.logaddexp(0,-margin).mean()/len(diffs)
            sigmoid=np.exp(-np.logaddexp(0,margin))
            grad-=(d.T@sigmoid)/len(d)/len(diffs)
        return float(loss),grad
    initial=np.zeros(x.shape[-1]);before=objective(initial)[0]
    opt=minimize(objective,initial,jac=True,method='L-BFGS-B',options={'maxiter':500,'ftol':1e-12,'gtol':1e-8,'maxls':50})
    assert opt.success and np.isfinite(opt.fun) and opt.fun<before
    state['weights']=opt.x
    log=dict(loss_before=before,loss_after=float(opt.fun),gradient_max=float(np.max(np.abs(opt.jac))),iterations=int(opt.nit),optimizer_success=bool(opt.success),optimizer_message=str(opt.message),head_parameters=len(opt.x),pca_components=PCA_DIM,pca_projection_parameters=state['components'].size,train_incidents=len(embeddings),pairs_per_incident=[len(p) for p in pairs],incident_equal_weight=True,L2=L2,pca_explained_ratio=float((singular[:PCA_DIM]**2).sum()/(singular**2).sum()))
    return state,log

def choose(embeddings,times,state):
    scores=design(embeddings,times,state)@state['weights']
    return scores,np.argmax(scores,axis=1)

def self_check():
    assert certain_pairs([0,1,2],0,0).tolist()==[[0,1],[0,2],[1,2]]
    assert certain_pairs([0,2],0,2).shape==(0,2)
    assert certain_pairs([0,2],1,1).shape==(0,2)
    assert certain_pairs([0,3],0,1).tolist()==[[0,1]]
    hidden=np.zeros((24,48,2))
    for i in range(12):hidden[(i//4)*8:(i//4+1)*8,(i%4)*12:(i%4+1)*12]=i
    np.testing.assert_array_equal(pool_tiles(hidden.reshape(1152,2))[:,0],np.arange(12))
    rng=np.random.default_rng(42);e=rng.normal(size=(3,12,16));t=np.tile(np.arange(12),(3,1));refs=[(0,0),(5,5),(9,10)]
    state,log=fit(e,t,refs);scores,ids=choose(e,t,state)
    assert scores.shape==(3,12) and ids.shape==(3,) and log['loss_after']<log['loss_before']
    # Per-example inference cannot acquire statistics from an unrelated evaluation example.
    first=choose(e[:1],t[:1],state)[0];mixed=choose(np.concatenate([e[:1],e[1:]*100]),t,state)[0][:1]
    np.testing.assert_allclose(first,mixed,rtol=0,atol=0)
    print('Pair interval, tile order, optimizer and inference isolation checks PASS')
if __name__=='__main__':self_check()
