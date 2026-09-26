"""Check whether head input clipping confounds the linear/head comparison."""
import time
import numpy as np,pandas as pd,joblib
from sklearn.base import clone
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score
from threadpoolctl import threadpool_limits
from run import O,R,B,data,make_splits,metrics,write,STEER,sha

write(O/'clipping_control_freeze.json',{'scope':'post-fit diagnostic, no tuning/promotion; fixed zscore+-8 on both linear and neural inputs. Original linear remains reference. Same rows, labels, C=.03 and heldout folds. Quantify this preprocessing difference rather than attributing it to architecture.','source_sha256':sha(__import__('pathlib').Path(__file__))})
old=pd.read_csv(R/'research/stage3_oof_external.csv');steer={(r.ID,r.sample_index):STEER.index(r.steer_forest) for r in old.itertuples()}
rows=[];scores=[]
with threadpool_limits(limits=2):
    for engine in ['dis','vjepa']:
        cs,_=data(engine);pub=[]
        for sp in make_splits(cs):
            sel=sp['selection'];x=np.stack([cs[id]['x'][i] for id,i in sel]);y=np.array([cs[id]['y'][i] for id,i in sel])
            sc=StandardScaler().fit(x);model=clone(joblib.load(B/'expanded_rav4.joblib')['accel'].named_steps['logisticregression'])
            model.fit(np.clip(sc.transform(x),-8,8),y);records=[]
            for id in sp['held']:
                c=cs[id];pred=model.predict(np.clip(sc.transform(c['x']),-8,8))
                for i in c['evaluation_indices']:
                    row={'engine':engine,'fold':sp['name'],'id':id,'sample_index':i,'truth':int(c['y'][i]),'prediction':int(pred[i])}
                    if c['public']:row.update(steer_truth=c['steer'][i],steer_prediction=steer[id,i])
                    records.append(row)
            rows.extend(records);frame=pd.DataFrame(records)
            if sp['name'].startswith('OPEN'):pub.extend(records)
            else:scores.append({'engine':engine,'scope':sp['name'],**metrics(frame.truth,frame.prediction)})
        frame=pd.DataFrame(pub);keep=frame.truth!=3;sf=f1_score(frame.loc[keep,'steer_truth'],frame.loc[keep,'steer_prediction'],labels=range(3),average='macro',zero_division=0);a=metrics(frame.truth,frame.prediction)
        scores.append({'engine':engine,'scope':'public_oof',**a,'S3':.7*a['macro_f1']+.3*sf})
pd.DataFrame(rows).to_csv(O/'clipping_control_predictions.csv',index=False)
pd.DataFrame(scores).to_csv(O/'clipping_control_metrics.csv',index=False)
print(pd.DataFrame(scores)[['engine','scope','macro_f1','opposite','S3']].to_string(index=False))
