"""Post-fit diagnostics only. Does not select/tune settings or change models."""
from pathlib import Path
import argparse,json,time
import numpy as np,pandas as pd,torch,joblib
from run import O,R,B,data,make_splits,Head,SEEDS,metrics,read,write,sha

def main(engine,inner_stop=False):
    global O
    cs,_=data(engine);splits=make_splits(cs);rows=[];continuous=[];sequence=[]
    if inner_stop:O=O/'route_early_stop'
    for sp in splits:
        d=O/engine/sp['name'];sel=sp['selection'];manifest=read(d/'train_manifest.json')
        xx=np.stack([cs[id]['context'][i] for id,i in sel]);yy=np.array([cs[id]['y'][i] for id,i in sel])
        lin=joblib.load(d/'linear.joblib');pr=lin.predict(xx[:,1]);rows.append({'engine':engine,'fold':sp['name'],'variant':'linear',**metrics(yy,pr)})
        for seed in SEEDS:
            for family in ['cls','aux']:
                name=f'{family}_{seed}';z=torch.load(d/f'{name}.pt',map_location='cpu',weights_only=False)
                m=Head();m.load_state_dict(z['state_dict']);m.eval()
                x=np.clip((xx-z['scaler_mean'])/z['scaler_scale'],-8,8).astype(np.float32)
                with torch.inference_mode():p,_=m(torch.from_numpy(x))
                rows.append({'engine':engine,'fold':sp['name'],'variant':name,**metrics(yy,p.argmax(1).numpy())})
                for id in sp['held']:
                    c=cs[id];pred=np.load(d/f'{name}_{id}.npz');p=pred['prob'].argmax(1)
                    if not c['public']:
                        valid=np.isfinite(c['sensor']);sensor=c['sensor'][valid];est=pred['acceleration_prediction'][valid]
                        continuous.append({'engine':engine,'fold':sp['name'],'variant':name,'id':id,'n':len(sensor),
                            'mae_m_s2':float(np.mean(abs(est-sensor))),
                            'constant_train_mean_mae':float(np.mean(abs(manifest['sensor_mean']-sensor))),
                            'correlation':float(np.corrcoef(sensor,est)[0,1]) if np.std(sensor)>0 and np.std(est)>0 else None})
                    mask=np.zeros(c['n'],bool);mask[c['evaluation_indices']]=True
                    wrong=mask&(p!=c['y']);opp=mask&(((c['y']==0)&(p==1))|((c['y']==1)&(p==0)))
                    def longest(v):
                        edges=np.flatnonzero(np.diff(np.r_[False,v,False]));return int(max(edges[1::2]-edges[::2],default=0))
                    sequence.append({'engine':engine,'fold':sp['name'],'variant':name,'id':id,
                        'wrong_max_adjacent_samples':longest(wrong),'opposite_max_adjacent_samples':longest(opp),
                        'note':'only strict labeled samples; gaps break runs. Sparse public points cannot evaluate transition timing.'})
    pd.DataFrame(rows).to_csv(O/f'{engine}_training_fit_diagnostics.csv',index=False)
    pd.DataFrame(continuous).to_csv(O/f'{engine}_sensor_diagnostics.csv',index=False)
    pd.DataFrame(sequence).to_csv(O/f'{engine}_sequence_diagnostics.csv',index=False)
    df=pd.read_csv(O/f'{engine}_predictions.csv');base=df[df.variant=='linear'][['fold','id','sample_index','prediction']].rename(columns={'prediction':'linear_prediction'})
    df=df.merge(base,on=['fold','id','sample_index'],validate='many_to_one')
    old=((df.truth==0)&(df.linear_prediction==1))|((df.truth==1)&(df.linear_prediction==0))
    new=((df.truth==0)&(df.prediction==1))|((df.truth==1)&(df.prediction==0))
    df['new_opposite']=new&~old;df['fixed_opposite']=old&~new
    df[df.prediction!=df.linear_prediction].to_csv(O/f'{engine}_changed_predictions.csv',index=False)
    write(O/f'{engine}_diagnostic_checks.json',{'post_fit_only':True,'train_evaluation_complete':True,'no_official_dense_transition_truth':True})
    print(engine,'diagnostics complete',flush=True)

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('engine');a.add_argument('--inner-stop',action='store_true');args=a.parse_args();torch.set_num_threads(2);main(args.engine,args.inner_stop)
