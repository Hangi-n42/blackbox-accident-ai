"""Continuous CAN transfer avoids assuming DACON's category thresholds."""
import sys,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from train_stage3 import ROOT,public_data,external_data,candidate,mirror_features
from solution.stage3 import transfer_features,ACCEL,STEER
import cv2,joblib,numpy as np,pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import f1_score,classification_report

cv2.setNumThreads(2)
x,ya,ys,g,df=public_data();ext=external_data()
assert ext is not None
continuous=ext[4].copy();mirror_y=continuous.copy();mirror_y[:,2]*=-1
reg=ExtraTreesRegressor(n_estimators=160,max_depth=16,min_samples_leaf=4,max_features=.5,n_jobs=2,random_state=42)
reg.fit(np.concatenate([ext[0],mirror_features(ext[0])]),np.concatenate([continuous,mirror_y]))
z=transfer_features(x,reg);zm=transfer_features(mirror_features(x),reg)
report={'method':'External CAN continuous regression, then public-label classifier. External thresholds not used. Public training fold includes reflection augmentation.','external_rows':len(ext[0]),'external_routes':len(set(ext[3])),'results':{}}
models={'motion_regressor':reg,'feature_version':'dis256_roi144_temporal6_v1'}
out=df.copy()
for task,y,labels in [('accel',ya,ACCEL),('steer',ys,STEER)]:
    records=[]
    for name in ('linear','extra','forest'):
        oof=np.empty_like(y)
        for held in np.unique(g):
            test=g==held;train=~test
            if task=='steer':train&=ya!=3
            mirror_y=2-y[train] if task=='steer' else y[train]
            m=candidate(name);m.fit(np.concatenate([z[train],zm[train]]),np.concatenate([y[train],mirror_y]));oof[test]=m.predict(z[test])
        mask=ya!=3 if task=='steer' else np.ones(len(y),bool)
        score=f1_score(y[mask],oof[mask],labels=np.arange(len(labels)),average='macro',zero_division=0)
        records.append({'name':name,'macro_f1':score,'report':classification_report(y[mask],oof[mask],labels=np.arange(len(labels)),target_names=list(labels),zero_division=0,output_dict=True)})
        out[f'{task}_{name}']=labels[oof];print(task,name,score,flush=True)
    winner=max(records,key=lambda r:r['macro_f1']);m=candidate(winner['name']);keep=ya!=3 if task=='steer' else np.ones(len(y),bool)
    m.fit(np.concatenate([z[keep],zm[keep]]),np.concatenate([y[keep],2-y[keep] if task=='steer' else y[keep]]))
    models[task]=m;report['results'][task]=records;report[task+'_selected']=winner['name']
report['selected_oof_stage3']=.7*max(r['macro_f1'] for r in report['results']['accel'])+.3*max(r['macro_f1'] for r in report['results']['steer'])
report['caveat']='Model selection uses same small public OOF set. External aligned-video duplicates excluded; route overlap not fully ruled out.'
joblib.dump(models,ROOT/'solution/model/stage3/motion_model_transfer.joblib',compress=3)
(ROOT/'research/stage3_validation_transfer.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
out.to_csv(ROOT/'research/stage3_oof_transfer.csv',index=False)
print(json.dumps({'stage3_oof':report['selected_oof_stage3']}),flush=True)
