"""One fixed CPU-only residual temporal experiment; no production writes."""
import os
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='2'
os.environ['CUDA_VISIBLE_DEVICES']=''
import sys,time,json,hashlib,importlib.util
sys.dont_write_bytecode=True
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
OUT=Path(__file__).resolve().parent/'dtype_fix_run'
sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
import cv2
from scipy.ndimage import uniform_filter1d
from sklearn.metrics import f1_score,confusion_matrix
from threadpoolctl import threadpool_limits
from research import stage3_temporal_experiment as old
from solution import stage3_v5_compatible as motion
torch.set_num_threads(2);torch.set_num_interop_threads(2);cv2.setNumThreads(2)
torch.manual_seed(42);np.random.seed(42)

def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def write(name,obj):
    p=OUT/name
    with p.open('x',encoding='utf-8') as f:json.dump(obj,f,ensure_ascii=False,indent=2)
def recover(x):
    centers=np.arange(2,len(x),2)-1
    return x[centers,:144].copy(),centers
def reconstruct(raw,centers,n):
    parts=[raw]+[uniform_filter1d(raw,size=w,axis=0,mode='nearest') for w in (5,15,31)]
    ix=np.arange(len(raw));sm=parts[2]
    for lag in (5,15):parts.append((sm[np.minimum(ix+lag,len(ix)-1)]-sm[np.maximum(ix-lag,0)])/(2*lag/10))
    a=np.concatenate(parts,axis=1)
    return np.stack([np.interp(np.arange(n),centers,a[:,j]) for j in range(864)],1).astype(np.float32)
def direct_raw(path):
    cap=cv2.VideoCapture(str(path));dis=cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_FAST)
    calc=motion.FrameFeatureComputer();prev=None;rows=[];i=0
    while True:
        ok,frame=cap.read()
        if not ok:break
        idx=i;i+=1
        if idx%2:continue
        gray=cv2.cvtColor(cv2.resize(frame,(256,max(96,int(round(frame.shape[0]*256/frame.shape[1]))))),cv2.COLOR_BGR2GRAY)
        if prev is not None:rows.append(calc(dis.calc(prev,gray,None)*10))
        prev=gray
    cap.release();return np.array(rows),i

class Residual(nn.Module):
    def __init__(self):
        super().__init__();self.proj=nn.Conv1d(144,32,1)
        self.layers=nn.ModuleList([nn.Conv1d(32,32,3,dilation=d) for d in (1,2,4,8)])
        self.out=nn.Conv1d(32,7,1);nn.init.zeros_(self.out.weight);nn.init.zeros_(self.out.bias)
    def forward(self,x):
        x=F.relu(self.proj(x))
        for layer,d in zip(self.layers,(1,2,4,8)):x=F.relu(layer(F.pad(x,(d,d),mode='replicate')))
        return self.out(x).squeeze(0).T

def interpolate(y,centers,indices):
    pos=torch.tensor(np.clip((np.asarray(indices)-centers[0])/2,0,len(centers)-1),dtype=torch.float32)
    lo=pos.long();hi=torch.clamp(lo+1,max=len(centers)-1);w=(pos-lo).unsqueeze(1)
    return y[lo]*(1-w)+y[hi]*w
def score(rows,ps):
    result={}
    for task,n in [('accel',4),('steer',3)]:
        ys=[];pred=[]
        for r,p in zip(rows,ps):
            mask=r['accel']!=3 if task=='steer' else np.ones(len(r['accel']),bool)
            ys.extend(r[task][mask]);pred.extend(p[task].argmax(1)[mask])
        result[task]={'macro_f1':float(f1_score(ys,pred,labels=list(range(n)),average='macro',zero_division=0)),
            'gt_counts':np.bincount(ys,minlength=n).tolist(),'confusion':confusion_matrix(ys,pred,labels=list(range(n))).tolist(),
            'class_f1':f1_score(ys,pred,labels=list(range(n)),average=None,zero_division=0).tolist()}
    result['stage3']=.7*result['accel']['macro_f1']+.3*result['steer']['macro_f1'];return result

def main():
    OUT.mkdir(exist_ok=False)
    start=time.perf_counter()
    rows=old.load_external();train,val=old.split_routes(rows)
    split=json.loads((ROOT/'research/stage3_temporal_v1/external_split.json').read_text())
    assert [r['route'] for r in train]==split['train'] and [r['route'] for r in val]==split['validation']
    sources={r['segment']:r for p in sorted((ROOT/'external_data/comma2k19').glob('*_manifest.json')) for r in json.loads(p.read_text())}
    protected=[ROOT/'model/stage3/motion_model.joblib',ROOT/'solution/stage3.py',ROOT/'solution/stage3_v5_compatible.py']
    binding={str(p.relative_to(ROOT)):sha(p) for p in protected+[Path(__file__),ROOT/'research/stage3_temporal_experiment.py',ROOT/'research/stage3_temporal_v1/external_split.json',ROOT/'research/stage3_external_overlap.json',ROOT/'Baseline/data/stage3/labels.csv']}
    inventory=[]
    for p in sorted((ROOT/'research/stage3_cache').glob('*.npy')):
        a=np.load(p,mmap_mode='r');inventory.append({'path':str(p.relative_to(ROOT)),'shape':list(a.shape),'dtype':str(a.dtype),'sha256':sha(p)})
    for r in rows:
        p=ROOT/'research/stage3_cache'/('ext_'+r['segment'].replace('/','_').replace('|','_')+'.npy')
        r['full']=np.load(p);r['raw'],r['centers']=recover(r['full'])
        base=ROOT/sources[r['segment']]['local']
        for name in ('video.hevc','global_pose/frame_times','processed_log/CAN/speed/t','processed_log/CAN/speed/value','processed_log/CAN/steering_angle/t','processed_log/CAN/steering_angle/value'):
            p=base/name;binding[str(p.relative_to(ROOT))]=sha(p)
    public_path=ROOT/'Baseline/data/stage3/videos/OPEN_001.mp4';binding[str(public_path.relative_to(ROOT))]=sha(public_path)
    plan={'status':'frozen_before_audit_and_fit','created_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'train_routes':split['train'],'heldout_routes':split['validation'],
        'split_history':'17/6 split used in earlier development; route held out of this fit, not fresh independent validation; both vehicles in training',
        'raw144_recovery':'20Hz stride2 pair centers are decoded index 1,3,...; select exact center rows prefix144; no reshape, no interpolation inversion',
        'network':'144->32 point projection, ReLU, Conv32 kernel3 dilations1,2,4,8 ReLU replicate padding; output7 zero initialization; receptive field31',
        'seed':42,'epochs':40,'optimizer':'AdamW lr0.001 weight_decay0.0001','loss':'0.7 weighted accel CE +0.3 weighted steer CE; train-set balanced class weights; steer excludes GTSTOPPED; full route gradient accumulation',
        'train_sampling':'same existing external17 2Hz rows only; no public GT in fit','baseline':'same Logistic C.03 balanced and RF160 depth10 leaf2 max_features.4 balanced seed42 from canonical source',
        'selection':'single last epoch; no early stopping/model selection on heldout; no hyperparameter search',
        'time_limit_training_seconds':900,'cpu_threads':2,'gpu':False,
        'protocol_amendment_before_fit':'This is an added bounded feasibility experiment, not the initial RAV4-to-Civic plus publicOOF adoption experiment. Single 17/6 fit requested under CPU 15min budget. Original adoption criteria remain unmet even if this feasibility gate passes. V7 Stage3 adoption forbidden from this fit alone.',
        'feasibility_gate_only':'both heldout head macroF1 nondecrease; stage3 strictly increase; >=4 of6 route stage3 strict improvements; local sample total inference ratio<=1.25; not adoption gate',
        'public_diagnostic':'OPEN001 10 known matching targetGT, all accel CONSTANT; excluded entire matching external route; not representative gate',
        'bindings':binding,'cache_inventory':inventory}
    write('frozen_plan.json',plan);print('Plan frozen; auditing recovered raw144',flush=True)
    audits=[]
    for inv in inventory:
        x=np.load(ROOT/inv['path']);raw,c=recover(x);z=reconstruct(raw,c,len(x));eq=np.array_equal(x.view(np.uint32),z.view(np.uint32))
        audits.append({'cache':inv['path'],'reconstruct864_bit_exact':eq,'max_abs':float(np.max(np.abs(x-z)))})
    direct=[]
    for p,x in [(public_path,np.load(ROOT/'research/stage3_cache/OPEN_001.npy')),(ROOT/sources[rows[0]['segment']]['local']/'video.hevc',rows[0]['full'])]:
        t=time.perf_counter();raw,n=direct_raw(p);expected,_=recover(x)
        direct.append({'path':str(p.relative_to(ROOT)),'decoded':n,'seconds':time.perf_counter()-t,'bit_exact':bool(np.array_equal(raw.view(np.uint32),expected.view(np.uint32))),'max_abs':float(np.max(np.abs(raw-expected)))})
    write('raw_recovery_audit.json',{'all_cache_reconstruction':audits,'direct_decode':direct})
    assert all(a['reconstruct864_bit_exact'] for a in audits) and all(a['bit_exact'] for a in direct),'Raw recovery mismatch; no training'
    model={};training_start=time.perf_counter()
    for task in ('accel','steer'):
        x,y=old.external_xy(train,task);model[task]=old.make_model(task).fit(x,y)
    base=[{task:old.probability(model[task],r['x_dense'],n) for task,n in [('accel',4),('steer',3)]} for r in val]
    proof=[]
    for i,p in enumerate(base):
        saved=np.load(ROOT/f'research/stage3_temporal_v1/external_validation_probabilities_{i:02}.npz')
        for task in ('accel','steer'):
            error=float(np.max(np.abs(p[task]-saved[task])))
            proof.append({'route':val[i]['route'],'task':task,'max_probability_abs_error':error,'all_predictions_equal':bool(np.array_equal(p[task].argmax(1),saved[task].argmax(1)))})
            assert error<1e-12 and proof[-1]['all_predictions_equal']
    write('baseline_reproduction.json',proof)
    rawtrain=np.concatenate([r['raw'] for r in train]);mean=rawtrain.mean(0);std=rawtrain.std(0);std[std<1e-6]=1.
    head=Residual();params=sum(p.numel() for p in head.parameters());optimizer=torch.optim.AdamW(head.parameters(),lr=.001,weight_decay=.0001)
    classweight={};denom={}
    for task,n in [('accel',4),('steer',3)]:
        _,y=old.external_xy(train,task);count=np.bincount(y,minlength=n);assert np.all(count>0)
        classweight[task]=torch.tensor(len(y)/(n*count),dtype=torch.float32);denom[task]=len(y)
    def tensor(r):return torch.from_numpy(((r['raw']-mean)/std).T.copy()).unsqueeze(0)
    prepared=[]
    for r in train:
        logs=np.concatenate([np.log(np.maximum(old.probability(model[task],r['x_train'],n),1e-6)) for task,n in [('accel',4),('steer',3)]],1)
        prepared.append((r,tensor(r),torch.tensor(logs,dtype=torch.float32)))
    losses=[]
    for epoch in range(40):
        if time.perf_counter()-training_start>900:raise TimeoutError('Fixed 900 second training limit; no candidate evaluation')
        optimizer.zero_grad();loss_epoch=0.
        for r,x,log in prepared:
            correction=interpolate(head(x),r['centers'],np.arange(len(r['x_train']))*10);logits=log+correction
            a=torch.tensor(r['accel_train'],dtype=torch.long);s=torch.tensor(r['steer_train'],dtype=torch.long);mask=a!=3
            loss=.7*F.cross_entropy(logits[:,:4],a,weight=classweight['accel'],reduction='sum')/denom['accel']
            loss+=.3*F.cross_entropy(logits[mask,4:],s[mask],weight=classweight['steer'],reduction='sum')/denom['steer']
            loss.backward();loss_epoch+=loss.item()
        optimizer.step();losses.append(loss_epoch)
        if epoch%10==9:print('Epoch',epoch+1,'train loss',loss_epoch,flush=True)
    training_seconds=time.perf_counter()-training_start
    assert training_seconds<=900,'Training exceeded fixed 900-second budget; no candidate evaluation'
    head.eval()
    def corrected(r,p,indices):
        with torch.no_grad():delta=interpolate(head(tensor(r)),r['centers'],indices).numpy()
        return {task:np.log(np.maximum(p[task],1e-6))+delta[:,sl] for task,sl in [('accel',slice(0,4)),('steer',slice(4,7))]}
    candidate=[corrected(r,p,np.arange(len(r['x_dense']))*2) for r,p in zip(val,base)]
    sb=score(val,base);sc=score(val,candidate)
    perroute=[{'route':r['route'],'vehicle':r['vehicle'],'baseline':score([r],[p]),'candidate':score([r],[q])} for r,p,q in zip(val,base,candidate)]
    public_x=np.load(ROOT/'research/stage3_cache/OPEN_001.npy');public_raw,public_c=recover(public_x)
    gt=pd.read_csv(ROOT/'Baseline/data/stage3/labels.csv');gt=gt[gt.ID=='OPEN_001'];idx=gt.frame_index.to_numpy()
    pr={'raw':public_raw,'centers':public_c,'accel':np.array([old.ACCEL.index(v) for v in gt.accel_label]),'steer':np.array([old.STEER.index(v) for v in gt.steer_label])}
    pp={task:old.probability(model[task],public_x[idx],n) for task,n in [('accel',4),('steer',3)]};pq=corrected(pr,pp,idx)
    public={'baseline':score([pr],[pp]),'candidate':score([pr],[pq]),'representative_adoption_evidence':False,'missing_accel_classes':['ACCELERATING','DECELERATING','STOPPED'],'selection_history':'previously repeatedly inspected public target labels; no GT used for this fit'}
    # Paired cached inference: total includes measured decoder/flow time separately, not a server guarantee.
    timings=[]
    with torch.no_grad():
        for r in [dict(pr,full=public_x),rows[0]]:
            pairs=[]
            for repeat in range(3):
                values={}
                for kind in (('baseline','candidate') if repeat%2==0 else ('candidate','baseline')):
                    t=time.perf_counter();p={task:old.probability(model[task],r['full'],n) for task,n in [('accel',4),('steer',3)]}
                    if kind=='candidate':corrected(r,p,np.arange(len(r['full'])))
                    values[kind]=time.perf_counter()-t
                pairs.append(values)
            timings.append({'rows':len(r['full']),'pairs':pairs})
    ratios=[]
    for timing,decode in zip(timings,direct):
        b=float(np.median([p['baseline'] for p in timing['pairs']]));c=float(np.median([p['candidate'] for p in timing['pairs']]))
        timing.update(baseline_cached_seconds=b,candidate_cached_seconds=c,decoder_and_flow_sample_seconds=decode['seconds'],estimated_total_ratio=(c+decode['seconds'])/(b+decode['seconds']))
        ratios.append(timing['estimated_total_ratio'])
    # Independent file state and chunk equivalence use one fixed sequence; no model updates.
    with torch.no_grad():
        x=tensor(rows[0]);a=head(x);head(tensor(pr));b=head(x)
        chunks=[]
        for lo in range(0,x.shape[-1],128):
            hi=min(lo+128,x.shape[-1]);l=max(lo-15,0);h=min(hi+15,x.shape[-1])
            chunks.append(head(x[:,:,l:h])[lo-l:hi-l])
        chunk=torch.cat(chunks);chunkerr=float(torch.max(torch.abs(chunk-a)));stateeq=bool(torch.equal(a,b))
    gate=all(sc[t]['macro_f1']>=sb[t]['macro_f1'] for t in ('accel','steer')) and sc['stage3']>sb['stage3'] and sum(r['candidate']['stage3']>r['baseline']['stage3'] for r in perroute)>=4 and max(ratios)<=1.25
    report={'status':'feasibility_passed_adoption_forbidden' if gate else 'feasibility_gate_failed','baseline':sb,'candidate':sc,'per_route':perroute,'public_OPEN001_gt_only_diagnostic':public,
        'train_seconds_including_baseline':training_seconds,'epochs_completed':40,'training_loss':losses,'parameters':params,'float32_parameter_bytes':params*4,
        'memory_estimate':'FP32 parameters 4P; Adam parameters+grad+2moments about16P excluding activations. Full route input 144*T*4 bytes; hidden layer about32*T*4 bytes each; no GPU allocated. Not measured server peak.',
        'timing':timings,'timing_limitations':'paired CPU cached heads; total ratios add one measured decoder/flow pass per sample; not full paired end-to-end or server60min guarantee',
        'contracts':{'file_order_no_state_bit_exact':stateeq,'chunk_halo15_max_logit_error':chunkerr,'chunk_float32_allclose_1e6':bool(torch.allclose(a,chunk,atol=1e-6,rtol=1e-6))},
        'feasibility_gate_passed':gate,'adoption_gate_passed':False,'no_production_model_written':True,'RAV4_to_Civic_not_executed':'single17route fit includes both vehicles; not cross-vehicle generalization proof',
        'protected_unchanged':all(sha(ROOT/p)==v for p,v in binding.items()),'elapsed_seconds':time.perf_counter()-start}
    write('report.json',report)
    # No checkpoint is emitted without root adoption review.
    print(json.dumps({'status':report['status'],'baseline':sb,'candidate':sc,'seconds':training_seconds},ensure_ascii=False),flush=True)

if __name__=='__main__':
    with threadpool_limits(limits=2):main()
