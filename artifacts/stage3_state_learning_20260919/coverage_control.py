"""One bounded data-coverage control, same DIS and linear recipe.

Only existing curated usable proxy labels are included; ambiguous/missing stay out.
"""
from pathlib import Path
import time
import numpy as np,pandas as pd,joblib
from sklearn.base import clone
from sklearn.metrics import f1_score
from threadpoolctl import threadpool_limits
from run import O,R,B,data,make_splits,metrics,write,read,sha,STEER

def main():
    dest=O/'coverage_control';dest.mkdir(exist_ok=True);cs,_=data('dis');splits=make_splits(cs)
    p=dest/'freeze.json'
    if p.exists():raise FileExistsError('Preserve completed control; do not overwrite')
    write(p,{'source_sha256':sha(Path(__file__)),'reason':'Post-main-experiment control: isolate current strict sample selection from feature/head improvements; not a pre-planned independent test.',
        'variants':'strict_full: all existing strict training indices; curated_usable: same curated label values, all label>=0 at2Hz in outertrain; curated_mass_matched: same curated_usable rows, uniform sample_weight=original_train_count/new_count to control total loss/regularization strength.',
        'fixed':'existing23sources, all heldout splits/evaluation rows/10HzDIS864 unchanged, LogisticRegression C=.03 balanced, original un-clipped standardization; no ambiguous/missing label assignment, no new data.',
        'gates':'same original meanS3>=+.01, no new public inversion, each externalfoldF1 nondecrease, eachfold/vehicle opposite-rate increase<=.01; deterministic linear singlefit, seed variance not claimed',
        'base_freeze_sha256':sha(O/'freeze.json')})
    old=pd.read_csv(R/'research/stage3_oof_external.csv');steer={(r.ID,r.sample_index):STEER.index(r.steer_forest) for r in old.itertuples()}
    predrows=[];scores=[];log=[]
    template=joblib.load(B/'expanded_rav4.joblib')['accel']
    for variant in ['strict_full','curated_usable','curated_mass_matched']:
        pub=[]
        for sp in splits:
            sel=[]
            for id in sp['train']:
                c=cs[id]
                ix=c['training_indices'] if variant=='strict_full' or c['public'] else [i for i in range(0,c['n'],5) if c['y'][i]>=0]
                sel.extend((id,i) for i in ix)
            original=set(map(tuple,sp['selection']));assert original<=set(sel)
            x=np.stack([cs[id]['x'][i] for id,i in sel]);y=np.array([cs[id]['y'][i] for id,i in sel]);assert (y>=0).all()
            model=clone(template);w=len(sp['selection'])/len(sel) if variant=='curated_mass_matched' else 1.
            start=time.perf_counter();model.fit(x,y,logisticregression__sample_weight=np.full(len(sel),w))
            folder=dest/variant/sp['name'];folder.mkdir(parents=True,exist_ok=True);joblib.dump(model,folder/'model.joblib')
            write(folder/'training_manifest.json',{'selection':sel,'class_counts':np.bincount(y,minlength=4).tolist(),'uniform_weight':w,'original_n':len(sp['selection'])})
            log.append({'variant':variant,'fold':sp['name'],'train_n':len(sel),'seconds':time.perf_counter()-start})
            rows=[]
            for id in sp['held']:
                c=cs[id];prob=model.predict_proba(c['x']);np.savez_compressed(folder/f'{id}.npz',prob=prob)
                for i in c['evaluation_indices']:
                    r={'variant':variant,'fold':sp['name'],'id':id,'vehicle':c['vehicle'],'sample_index':i,'truth':int(c['y'][i]),'prediction':int(prob[i].argmax())}
                    if c['public']:r.update(steer_truth=int(c['steer'][i]),steer_prediction=steer[id,i])
                    rows.append(r)
            predrows.extend(rows);df=pd.DataFrame(rows)
            if sp['name'].startswith('OPEN'):pub.extend(rows)
            else:
                for group,g in [('all',df),*list(df.groupby('vehicle'))]:scores.append({'variant':variant,'scope':sp['name'],'group':group,**metrics(g.truth,g.prediction)})
        df=pd.DataFrame(pub);keep=df.truth!=3;sf=f1_score(df.loc[keep,'steer_truth'],df.loc[keep,'steer_prediction'],labels=range(3),average='macro',zero_division=0);a=metrics(df.truth,df.prediction)
        scores.append({'variant':variant,'scope':'public_oof','group':'all',**a,'S3':.7*a['macro_f1']+.3*sf})
    frame=pd.DataFrame(predrows);base=pd.read_csv(O/'dis_predictions.csv');base=base[base.variant=='linear'][['fold','id','sample_index','prediction']].rename(columns={'prediction':'reference'})
    frame=frame.merge(base,on=['fold','id','sample_index'],validate='many_to_one')
    new=((frame.truth==0)&(frame.prediction==1))|((frame.truth==1)&(frame.prediction==0));old=((frame.truth==0)&(frame.reference==1))|((frame.truth==1)&(frame.reference==0));frame['new_opposite_vs_dis']=new&~old;frame['fixed_opposite_vs_dis']=old&~new
    assert (frame.loc[(frame.variant=='strict_full')&frame.fold.str.startswith('OPEN'),'prediction']==frame.loc[(frame.variant=='strict_full')&frame.fold.str.startswith('OPEN'),'reference']).all()
    frame.to_csv(dest/'predictions.csv',index=False);pd.DataFrame(scores).to_csv(dest/'metrics.csv',index=False);write(dest/'execution.json',log)
    print(pd.DataFrame(scores).query("group=='all'")[['variant','scope','n','macro_f1','opposite','S3']].to_string(index=False),flush=True)

if __name__=='__main__':
    with threadpool_limits(limits=2):main()
