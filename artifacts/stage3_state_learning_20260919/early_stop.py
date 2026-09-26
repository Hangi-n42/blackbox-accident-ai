"""Prospective follow-up: choose epochs on training routes only, then refit.

Motivated by already observed 1.0 train F1, never an independent-test claim.
The original fixed120 experiment and all its outputs are preserved.
"""
from pathlib import Path
import argparse,hashlib,time,shutil
import numpy as np
import torch
from torch import nn
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
import run as base
O=base.O;DEST=O/'route_early_stop'

def internal_split(cs,sp):
    sel=sp['selection'];unique={id for id,i in sel if not cs[id]['public']}
    ordered=sorted(unique,key=lambda id:hashlib.sha256(('stage3-internal-stop:'+cs[id]['route']).encode()).hexdigest())
    held=[];target=max(1,int(len(sel)*.2));present={int(cs[id]['y'][i]) for id,i in sel}
    # Select whole sources and retain every currently trainable class in inner training.
    for vehicle in sorted({cs[id]['vehicle'] for id in unique}):
        for id in ordered:
            if cs[id]['vehicle']!=vehicle:continue
            ids=set(held+[id]);remain=[(j,i) for j,i in sel if j not in ids]
            n=sum(j in ids for j,i in sel)
            if {int(cs[j]['y'][i]) for j,i in remain}==present and n<=len(sel)*.35:
                held.append(id);break
    for id in ordered:
        if sum(j in held for j,i in sel)>=target:break
        if id in held:continue
        ids=set(held+[id]);remain=[(j,i) for j,i in sel if j not in ids]
        if {int(cs[j]['y'][i]) for j,i in remain}==present and sum(j in ids for j,i in sel)<=len(sel)*.35:held.append(id)
    train=[(id,i) for id,i in sel if id not in held];val=[(id,i) for id,i in sel if id in held]
    assert train and val and not set(held)&set(sp['held'])
    return [id for id in sp['train'] if id not in held],train,val,held

def prepare(cs,ids,sel):
    pool=sorted(set(sel)|{(id,i) for id in ids if not cs[id]['public'] for i in range(0,cs[id]['n'],5) if np.isfinite(cs[id]['sensor'][i])})
    selected=set(sel);x=np.stack([cs[id]['context'][i] for id,i in pool]);y=np.array([cs[id]['y'][i] if (id,i) in selected else -1 for id,i in pool]);mask=y>=0
    sensor=np.array([cs[id]['sensor'][i] for id,i in pool]);smask=np.isfinite(sensor)
    sc=StandardScaler().fit(x[mask,1]);x=np.clip(sc.transform(x.reshape(-1,864)),-8,8).reshape(x.shape).astype(np.float32)
    mean=float(sensor[smask].mean());std=max(float(sensor[smask].std()),1e-5);st=np.zeros(len(pool),np.float32);st[smask]=(sensor[smask]-mean)/std
    cw=mask.sum()/(4*np.bincount(y[mask],minlength=4).clip(1))
    return dict(x=torch.from_numpy(x),y=torch.from_numpy(y),mask=mask,sensor=torch.from_numpy(st),smask=smask,
                weight=torch.tensor(cw,dtype=torch.float32),scaler=sc,mean=mean,std=std)

def transform(x,sc):return torch.from_numpy(np.clip(sc.transform(x.reshape(-1,864)),-8,8).reshape(x.shape).astype(np.float32))

def train(a,seed,aux,epochs,val=None):
    torch.manual_seed(seed);m=base.Head();opt=torch.optim.AdamW(m.parameters(),lr=.001,weight_decay=.01);curve=[]
    for ep in range(epochs):
        m.train();opt.zero_grad();p,r=m(a['x'])
        loss=nn.functional.cross_entropy(p[a['mask']],a['y'][a['mask']],weight=a['weight'])
        if aux:loss=loss+.2*nn.functional.smooth_l1_loss(r[a['smask']],a['sensor'][a['smask']])
        loss.backward();nn.utils.clip_grad_norm_(m.parameters(),1.);opt.step()
        if val is not None:
            m.eval()
            with torch.inference_mode():p,_=m(val[0]);loss=nn.functional.cross_entropy(p,val[1],weight=a['weight'],reduction='none')
            curve.append(float(torch.stack([loss[g].mean() for g in val[2]]).mean()))
    return m.eval(),curve

def fit(cs,sp,engine):
    dst=DEST/engine/sp['name'];dst.mkdir(parents=True,exist_ok=True);src=O/engine/sp['name']
    for p in src.glob('linear*'):shutil.copyfile(p,dst/p.name)
    shutil.copyfile(src/'train_manifest.json',dst/'train_manifest.json')
    ids,sel,val,held=internal_split(cs,sp);a=prepare(cs,ids,sel);full=prepare(cs,sp['train'],sp['selection'])
    vx=transform(np.stack([cs[id]['context'][i] for id,i in val]),a['scaler']);vy=torch.tensor([cs[id]['y'][i] for id,i in val])
    groups=[np.array([id==h for id,i in val]) for h in held];groups=[g for g in groups if g.any()]
    logs=[]
    for seed in base.SEEDS:
        for aux in [False,True]:
            start=time.perf_counter();_,curve=train(a,seed,aux,base.EPOCHS,(vx,vy,groups));epochs=int(np.argmin(curve))+1
            m,_=train(full,seed,aux,epochs);name=('aux' if aux else 'cls')+f'_{seed}'
            for id in sp['held']:
                xx=transform(cs[id]['context'],full['scaler'])
                with torch.inference_mode():p,r=m(xx)
                np.savez_compressed(dst/f'{name}_{id}.npz',prob=p.softmax(1).numpy(),acceleration_prediction=r.numpy()*full['std']+full['mean'])
            torch.save({'state_dict':m.state_dict(),'scaler_mean':full['scaler'].mean_,'scaler_scale':full['scaler'].scale_,
                        'sensor_mean':full['mean'],'sensor_std':full['std'],'experimental_only':True},dst/f'{name}.pt')
            logs.append({'variant':name,'selected_epoch':epochs,'inner_group_balanced_ce_curve':curve,'seconds':time.perf_counter()-start})
    base.write(dst/'execution.json',{'internal_validation_sources':held,'internal_training_rows':sel,'internal_validation_rows':val,'fits':logs})
    print(engine,sp['name'],[(r['variant'],r['selected_epoch']) for r in logs],flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('engine',choices=['dis','vjepa']);args=p.parse_args()
    torch.set_num_threads(2);torch.set_num_interop_threads(2);cs,_=base.data(args.engine);splits=base.make_splits(cs)
    DEST.mkdir(exist_ok=True)
    if not (DEST/'freeze.json').exists():
        f=base.read(O/'freeze.json');f['followup_source_sha256']=base.sha(Path(__file__))
        f['amendment']='After fixed120 DIS results showed trainF1=1.0, before early-stop results: hash-selected20percent whole training sources, max35percent, at least one per vehicle when possible, retain all classes. Same selection for features/seeds/objectives. Inner group-balanced weightedCE selects epoch1..120; reinitialize and refit all outertrain for selectedepoch. All scaler and sensor statistics recomputed from innertrain only. No public heldout/date heldout tuning; same adoption gates, no claim of independent development.'
        f['internal_splits']={sp['name']:internal_split(cs,sp)[3] for sp in splits};base.write(DEST/'freeze.json',f)
    with threadpool_limits(limits=2):
        for sp in splits:
            if not (DEST/args.engine/sp['name']/'execution.json').exists():fit(cs,sp,args.engine)
        base.O=DEST;base.evaluate(cs,splits,args.engine)
