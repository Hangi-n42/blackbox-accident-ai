"""Compare current Mac and comma-only models on exposed public development labels."""
import json,sys,time
from pathlib import Path
import cv2,joblib,numpy as np,pandas as pd
from sklearn.metrics import f1_score,confusion_matrix
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'artifacts/submissions/verify_v6/model/stage2/code'))
from solution.stage3_v5_compatible import extract_motion,ACCEL,STEER
OUT=ROOT/'artifacts/data_pilot_20260916/comma_experiment'

def main():
    cv2.setNumThreads(2);cache=OUT/'mac_public_features';cache.mkdir(exist_ok=True)
    gt=pd.read_csv(ROOT/'Baseline/data/stage3/labels.csv')
    models={'current_mac':joblib.load(ROOT/'artifacts/submissions/verify_v6/model/stage3/motion_model.joblib')}
    models.update({name:joblib.load(OUT/(name+'.joblib')) for name in ['old','clean']})
    frames={};start=time.monotonic()
    for id in gt.ID.unique():
        path=cache/(id+'.npy')
        if not path.exists():np.save(path,extract_motion(ROOT/'artifacts/public_eval_10hz/stage3/videos'/(id+'.mp4')))
        frames[id]=np.load(path);assert frames[id].shape[1]==864
        print('features',id,frames[id].shape,flush=True)
    report={'scope':'Exposed public development labels; current model training differs from comma-only models. Old versus clean isolates cleaning; current versus clean is adoption diagnostic, not causal data-only comparison.','results':{}}
    for name,model in models.items():
        rows=[]
        for r in gt.itertuples():
            x=frames[r.ID][[r.sample_index]]
            rows.append({'ID':r.ID,'sample_index':r.sample_index,'accel_label':str(ACCEL[int(model['accel'].predict(x)[0])]),'steer_label':str(STEER[int(model['steer'].predict(x)[0])])})
        pred=pd.DataFrame(rows);joined=gt.merge(pred,on=['ID','sample_index'],suffixes=('_gt','_pred'),validate='one_to_one');res={}
        for task,names in [('accel',list(ACCEL)),('steer',list(STEER))]:
            subset=joined if task=='accel' else joined[joined.accel_label_gt!='STOPPED']
            a=subset[task+'_label_gt'];b=subset[task+'_label_pred']
            res[task]={'macro_f1':float(f1_score(a,b,labels=names,average='macro',zero_division=0)),'n':len(a),'correct':int((a==b).sum()),'classes':names,'confusion':confusion_matrix(a,b,labels=names).tolist()}
        res['weighted']=.7*res['accel']['macro_f1']+.3*res['steer']['macro_f1'];report['results'][name]=res
        pred.to_csv(OUT/(name+'_public.csv'),index=False);print(name,res['weighted'],flush=True)
    previous=pd.read_csv(ROOT/'artifacts/mac_experiments/baseline_20260916/stage3.csv')
    current=pd.read_csv(OUT/'current_mac_public.csv');x=current.merge(previous,on=['ID','sample_index'],suffixes=('_new','_prior'),validate='one_to_one')
    report['current_matches_prior_labeled_predictions']={t:int((x[t+'_label_new']!=x[t+'_label_prior']).sum()) for t in ['accel','steer']}
    report['seconds']=time.monotonic()-start;report['production_changed']=False
    (OUT/'public_comparison.json').write_text(json.dumps(report,indent=2))
if __name__=='__main__':main()
