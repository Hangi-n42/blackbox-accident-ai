"""Isolated Stage3 experiment. Cached features, frozen folds, no production writes."""
from pathlib import Path
import argparse, hashlib, json, time, warnings
import numpy as np
import pandas as pd
import torch
from torch import nn
import joblib
from sklearn.base import clone
from sklearn.metrics import f1_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

O=Path(__file__).resolve().parent; R=O.parents[1]
B=R/'artifacts/stage3_training_basis_20260917'
D=R/'artifacts/stage3_mixed_datecheck_20260917'
F=R/'artifacts/stage3_factor_comparison_20260918/dis'
SEEDS=[17,42,73]; EPOCHS=120
ACC=['ACCELERATING','DECELERATING','CONSTANT','STOPPED']
STEER=['LEFT','STRAIGHT','RIGHT']
read=lambda p:json.loads(p.read_text())
write=lambda p,v:p.write_text(json.dumps(v,ensure_ascii=False,indent=2))
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def data(engine):
    cs=read(B/'cases.json'); out={}
    for c in cs:
        z=dict(np.load(B/c['labels_npz'])); c=dict(c)
        c.update(y=z['accel_candidate'].astype(int),sensor=z['acceleration_proxy'].astype(np.float32),
                 steer=z['steer_candidate'].astype(int),speed=z['speed_smoothed'],public=False)
        c['n']=len(c['y']);out[c['id']]=c
    pub=pd.read_csv(R/'Baseline/data/stage3/labels.csv')
    for id,g in pub.groupby('ID',sort=False):
        n=len(np.load(F/(id+'.npz'))['base']);y=np.full(n,-1);s=y.copy()
        y[g.sample_index]=g.accel_label.map(dict(zip(ACC,range(4))))
        s[g.sample_index]=g.steer_label.map(dict(zip(STEER,range(3))))
        out[id]={'id':id,'n':n,'y':y,'steer':s,'sensor':np.full(n,np.nan),
                 'public':True,'training_indices':g.sample_index.tolist(),
                 'evaluation_indices':g.sample_index.tolist(),'vehicle':'public'}
    for id,c in out.items():
        p=F/(id+'.npz') if engine=='dis' else O/'vjepa_features'/(id+'.npz')
        c['x']=np.load(p)['base'].astype(np.float32)
        assert len(c['x'])==c['n'] and np.isfinite(c['x']).all()
        if c['x'].shape[1]<864:c['x']=np.pad(c['x'],((0,0),(0,864-c['x'].shape[1])))
        assert c['x'].shape[1]==864
        ix=np.arange(c['n'])[:,None]+np.array([-5,0,5])[None,:]
        c['context']=c['x'][np.clip(ix,0,c['n']-1)]
    return out,pub

def make_splits(cs):
    splits=[]
    for k in range(1,4):
        rows=read(D/f'fold_{k}/split_manifest.json')
        train=[r['id'] for r in rows if r['experiment_role']=='train']
        held=[r['id'] for r in rows if r['experiment_role']!='train']
        assert set(train).isdisjoint(held)
        sel=read(D/f'fold_{k}/training_selection.json')['mixed_budget_rav4']
        assert set(r['id'] for r in sel)<=set(train)
        splits.append({'name':f'date_{k}','train':train,'held':held,
                       'selection':[(r['id'],r['sample_index']) for r in sel]})
    for held in [id for id,c in cs.items() if c['public']]:
        train=[id for id in cs if id!=held]
        sel=[(id,i) for id in train for i in cs[id]['training_indices']]
        splits.append({'name':held,'train':train,'held':[held],'selection':sel})
    return splits

def freeze(cs,splits):
    p=O/'freeze.json'
    if p.exists():return
    protected=[R/'artifacts/submissions/verify_v6/model/stage3/motion_model.joblib',
               R/'artifacts/submissions/verify_v6/model/stage2/code/solution/stage3.py',
               R/'releases/v7/source/model/stage3/motion_model.joblib']
    inputs=[B/'cases.json',R/'Baseline/data/stage3/labels.csv',R/'research/stage3_oof_external.csv']
    inputs += [B/c['labels_npz'] for c in cs.values() if not c['public']]
    inputs += [F/(id+'.npz') for id in cs]
    write(p,{'source_sha256':sha(Path(__file__)),'seeds':SEEDS,'epochs':EPOCHS,
        'architecture':'864->32 shared GELU at t-0.5,t,t+0.5 seconds; concatenated96->32 GELU;4way classifier and1 acceleration regressor always instantiated',
        'objectives':'class-balanced CE only vs identical CE +0.2 SmoothL1 of train-standardized continuous acceleration; regression uses finite existing11point1second CAN slope at2Hz; missing remains masked',
        'training':'AdamW lr0.001 weight_decay0.01, full-batch120epochs,clip_grad1; no early stopping or heldout tuning; scaler fit only selected training classification centers, clip zscore+-8; identical initialization per seed; CPU2threads',
        'feature_control':'DIS864 vs frozen VJEPA768 padded96zeros; identical input dimension and head parameter count; VJEPA temporal tokens retained, no whole-video pooling',
        'public_score':'50 official sparse labels LOVO; fixed historical external23 steer_forest OOF47nonSTOPPED; S3=.7accelMacroF1+.3steerMacroF1; steering identical for every candidate',
        'external':'existing3date split and exact385classification rows; no new public supervision in these fits; strict heldout5450-subset; proxy only; repeated dates cannot be pooled as independent',
        'gates':'report S3 increase and inversion risk separately. Promotion requires public S3 mean gain>=0.01, gain>=0 in each of3seeds, no new public inversions; external meanF1 no decrease in eachfold, mean opposite-rate increase <=0.01 in eachfold and vehicle. Passed gate yields development candidate only, never automatic production replacement.',
        'exposure':'all data previously used in development; public other4 labels train eachLOVO, heldvideo never fit; one known aligned external duplicate already excluded, fullsource overlap unproven; no independent/private claim',
        'vjepa_plan':'official frozen ViT-B/16 384, RGB10Hz16frame disjoint windows, spatial mean per tubelet8time tokens and timestamp interpolation to10Hz; replicate final padding; short-side384 then center-crop384 per official eval; CPU/MPS smoke before full28shortclips; no encoder finetuning',
        'protected':{str(p.relative_to(R)):sha(p) for p in protected if p.exists()},
        'inputs':{str(p.relative_to(R)):sha(p) for p in inputs},'splits':splits})
    write(O/'evaluation_manifest.json',[
        {'id':id,'source':c.get('raw_path',f'artifacts/public_eval_10hz/stage3/videos/{id}.mp4'),
         'labels':c.get('truth_source','Baseline/data/stage3/labels.csv'),
         'truth_kind':'official' if c['public'] else 'sensor_proxy',
         'development_exposed':True,'evaluation_indices':c['evaluation_indices'],
         'license':'competition rules' if c['public'] else 'MIT comma.ai2018'} for id,c in cs.items()])

class Head(nn.Module):
    def __init__(self):
        super().__init__();self.proj=nn.Sequential(nn.Linear(864,32),nn.GELU())
        self.temporal=nn.Sequential(nn.Linear(96,32),nn.GELU())
        self.cls=nn.Linear(32,4);self.reg=nn.Linear(32,1)
    def forward(self,x):
        h=self.temporal(self.proj(x).flatten(1));return self.cls(h),self.reg(h).flatten()

def metrics(y,p):
    cm=confusion_matrix(y,p,labels=range(4));n=int(np.isin(y,[0,1]).sum())
    return {'n':len(y),'macro_f1':float(f1_score(y,p,labels=range(4),average='macro',zero_division=0)),
            'class_f1':f1_score(y,p,labels=range(4),average=None,zero_division=0).tolist(),
            'confusion':cm.tolist(),'opposite':int(cm[0,1]+cm[1,0]),'opposite_denominator':n,
            'opposite_rate':float((cm[0,1]+cm[1,0])/n) if n else 0.,
            'accel_to_decel':int(cm[0,1]),'decel_to_accel':int(cm[1,0])}

def fit(cs,sp,engine):
    dest=O/engine/sp['name'];dest.mkdir(parents=True,exist_ok=True)
    sel=sp['selection'];pool=sorted(set(sel)|{(id,i) for id in sp['train'] if not cs[id]['public']
                           for i in range(0,cs[id]['n'],5) if np.isfinite(cs[id]['sensor'][i])})
    selected=set(sel);x=np.stack([cs[id]['context'][i] for id,i in pool])
    y=np.array([cs[id]['y'][i] if (id,i) in selected else -1 for id,i in pool]); mask=y>=0
    assert mask.sum()==len(sel)
    sensor=np.array([cs[id]['sensor'][i] for id,i in pool]);smask=np.isfinite(sensor)
    sc=StandardScaler().fit(x[mask,1]);x=np.clip(sc.transform(x.reshape(-1,864)),-8,8).reshape(x.shape).astype(np.float32)
    sm=float(sensor[smask].mean());ss=max(float(sensor[smask].std()),1e-5)
    st=np.zeros(len(sensor),np.float32);st[smask]=(sensor[smask]-sm)/ss
    tx=torch.from_numpy(x);ty=torch.from_numpy(y);ts=torch.from_numpy(st)
    cw=len(y[mask])/(4*np.bincount(y[mask],minlength=4).clip(1))
    weight=torch.tensor(cw,dtype=torch.float32)
    inputs={id:torch.from_numpy(np.clip(sc.transform(cs[id]['context'].reshape(-1,864)),-8,8).reshape(cs[id]['context'].shape).astype(np.float32)) for id in sp['held']}
    write(dest/'train_manifest.json',{'classification':sel,'sensor_pool':pool,'n_class':int(mask.sum()),'n_sensor':int(smask.sum()),'sensor_mean':sm,'sensor_std':ss,'class_counts':np.bincount(y[mask],minlength=4).tolist()})
    template=joblib.load(B/'expanded_rav4.joblib')['accel']
    start=time.perf_counter();linear=clone(template);linear.fit(np.stack([cs[id]['x'][i] for id,i in sel]),y[mask] if pool==sel else np.array([cs[id]['y'][i] for id,i in sel]))
    joblib.dump(linear,dest/'linear.joblib')
    for id in sp['held']:
        prob=linear.predict_proba(cs[id]['x']);np.savez_compressed(dest/f'linear_{id}.npz',prob=prob)
    log={'linear_seconds':time.perf_counter()-start,'fits':[]}
    for seed in SEEDS:
        for aux in [False,True]:
            torch.manual_seed(seed);m=Head();opt=torch.optim.AdamW(m.parameters(),lr=.001,weight_decay=.01)
            start=time.perf_counter();losses=[]
            for ep in range(EPOCHS):
                m.train();opt.zero_grad();pred,reg=m(tx)
                ce=nn.functional.cross_entropy(pred[mask],ty[mask],weight=weight)
                loss=ce+(0.2*nn.functional.smooth_l1_loss(reg[smask],ts[smask]) if aux else 0.)
                assert torch.isfinite(loss);loss.backward();nn.utils.clip_grad_norm_(m.parameters(),1.);opt.step()
                if ep in [0,EPOCHS-1]:losses.append(float(loss.detach()))
            name=('aux' if aux else 'cls')+f'_{seed}';m.eval();infer=0.
            for id,xx in inputs.items():
                started=time.perf_counter()
                with torch.inference_mode():p,a=m(xx)
                infer+=time.perf_counter()-started
                np.savez_compressed(dest/f'{name}_{id}.npz',prob=p.softmax(1).numpy(),acceleration_prediction=a.numpy()*ss+sm)
            torch.save({'state_dict':m.state_dict(),'scaler_mean':sc.mean_,'scaler_scale':sc.scale_,
                        'sensor_mean':sm,'sensor_std':ss,'experimental_only':True},dest/f'{name}.pt')
            log['fits'].append({'variant':name,'seconds':time.perf_counter()-start,'inference_seconds':infer,'loss_first_last':losses,'parameters':sum(p.numel() for p in m.parameters())})
    write(dest/'execution.json',log);print(engine,sp['name'],len(sel),'class',int(smask.sum()),'sensor; completed',flush=True)

def evaluate(cs,splits,engine):
    variants=['linear']+[f'{h}_{s}' for h in ['cls','aux'] for s in SEEDS];allrows=[];scores=[]
    old=pd.read_csv(R/'research/stage3_oof_external.csv')
    steer={(r.ID,r.sample_index):STEER.index(r.steer_forest) for r in old.itertuples()}
    for v in variants:
        pubrows=[]
        for sp in splits:
            rec=[]
            for id in sp['held']:
                c=cs[id];z=np.load(O/engine/sp['name']/f'{v}_{id}.npz');p=z['prob'].argmax(1)
                for i in c['evaluation_indices']:
                    row={'variant':v,'fold':sp['name'],'id':id,'sample_index':i,'vehicle':c['vehicle'],'truth':int(c['y'][i]),'prediction':int(p[i]),**{f'p{j}':float(z['prob'][i,j]) for j in range(4)}}
                    if c['public']:row.update(steer_truth=int(c['steer'][i]),steer_prediction=steer[id,i])
                    rec.append(row)
            frame=pd.DataFrame(rec);allrows.extend(rec)
            if sp['name'].startswith('OPEN'):pubrows.extend(rec)
            else:
                for group,g in [('all',frame),*list(frame.groupby('vehicle'))]:
                    scores.append({'engine':engine,'variant':v,'scope':sp['name'],'group':group,**metrics(g.truth,g.prediction)})
        frame=pd.DataFrame(pubrows);mask=frame.truth!=3
        sf=float(f1_score(frame.loc[mask,'steer_truth'],frame.loc[mask,'steer_prediction'],labels=range(3),average='macro',zero_division=0))
        acc=metrics(frame.truth,frame.prediction)
        scores.append({'engine':engine,'variant':v,'scope':'public_oof','group':'all',**acc,'steer_macro_f1':sf,'S3':.7*acc['macro_f1']+.3*sf})
    pd.DataFrame(allrows).to_csv(O/f'{engine}_predictions.csv',index=False)
    pd.DataFrame(scores).to_csv(O/f'{engine}_metrics.csv',index=False);write(O/f'{engine}_metrics.json',scores)
    f=read(O/'freeze.json');assert all(sha(R/p)==h for p,h in {**f['inputs'],**f['protected']}.items())
    write(O/f'{engine}_checks.json',{'completed':True,'inputs_and_production_unchanged':True,'features_finite':True,'folds':len(splits),'variants':len(variants)})
    print(pd.DataFrame(scores).query("group=='all'")[['scope','variant','macro_f1','opposite','S3']].to_string(index=False),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('engine',choices=['dis','vjepa']);p.add_argument('--evaluate-only',action='store_true');a=p.parse_args()
    torch.set_num_threads(2);torch.set_num_interop_threads(2)
    with threadpool_limits(limits=2):
        cs,_=data(a.engine);splits=make_splits(cs);freeze(cs,splits)
        if not a.evaluate_only:
            for sp in splits:
                if not (O/a.engine/sp['name']/'execution.json').exists():fit(cs,sp,a.engine)
        evaluate(cs,splits,a.engine)
