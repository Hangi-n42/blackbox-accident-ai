"""Fixed 17/6-route development comparison; sensor proxies are not competition GT."""
import json,sys
from pathlib import Path
import joblib,numpy as np
from scipy.ndimage import uniform_filter1d
from sklearn.metrics import f1_score
from threadpoolctl import threadpool_limits
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from research.stage3_temporal_experiment import make_model
from scripts.data.clean_comma_pilot import align
OUT=ROOT/'artifacts/data_pilot_20260916/comma_experiment'

def labels(t,speed,angle):
    speed=uniform_filter1d(speed,size=21)
    acceleration=np.gradient(speed,t)
    angle=uniform_filter1d(angle,size=11)
    return np.where(speed<.3,3,np.where(acceleration>.25,0,np.where(acceleration<-.25,1,2))),np.where(angle>2,0,np.where(angle<-2,2,1))

def main():
    OUT.mkdir(exist_ok=True)
    split=json.loads((ROOT/'research/stage3_scale_augmentation/external_split.json').read_text())
    manifest=json.loads((ROOT/'artifacts/data_pilot_20260916/comma/manifest.json').read_text())['records']
    rows=[]
    for item in manifest:
        segment=item['segment'];chunk,route,seg=segment.split('/')
        base=ROOT/'external_data/comma2k19'/chunk/route.replace('|','_')/seg
        x=np.load(ROOT/'research/stage3_cache'/('ext_'+segment.replace('/','_').replace('|','_')+'.npy'))
        t=np.load(base/'global_pose/frame_times').ravel()
        assert len(x)==len(t)
        original=[];clean=[];valid=np.ones(len(t),bool)
        for name in ['speed','steering_angle']:
            p=base/'processed_log/CAN'/name;st=np.load(p/'t').ravel();v=np.load(p/'value').ravel()
            original.append(np.interp(t,st,v))
            value,mask,_=align(st,v,t)
            # Supply edge values only for smoothing; expanded edge mask excludes their influence.
            clean.append(np.interp(t,st[np.r_[np.diff(st)!=0,True]],v[np.r_[np.diff(st)!=0,True]]))
            valid &= mask
        valid &= uniform_filter1d(valid.astype(float),size=23,mode='constant') > .999
        rows.append({'route':item['source_group'],'x':x,'old':labels(t,*original),'clean':labels(t,*clean),'valid':valid})
    report={'scope':'Previously exposed comma development routes only; not independent target-domain accuracy.',
            'policy':'Same native-frame features, 17/6 source split, 2Hz training and 10Hz evaluation. Old proxy thresholds and smoothing unchanged; cleaned CAN duplicate handling and expanded invalid-time masks.',
            'split':split,'results':{}}
    predictions={}
    for variant in ['old','clean']:
        models={};predictions[variant]={}
        for k,task in enumerate(['accel','steer']):
            xx=[];yy=[]
            for row in rows:
                if row['route'] not in split['train']:continue
                idx=np.arange(0,len(row['x']),10)
                if variant=='clean':idx=idx[row['valid'][idx]]
                if task=='steer':idx=idx[row[variant][0][idx]!=3]
                xx.append(row['x'][idx]);yy.append(row[variant][k][idx])
            model=make_model(task)
            with threadpool_limits(limits=2):model.fit(np.concatenate(xx),np.concatenate(yy))
            models[task]=model;truth=[];pred=[]
            for row in rows:
                if row['route'] not in split['validation']:continue
                idx=np.arange(0,len(row['x']),2);idx=idx[row['valid'][idx]]
                if task=='steer':idx=idx[row['clean'][0][idx]!=3]
                truth.extend(row['clean'][k][idx]);pred.extend(model.predict(row['x'][idx]))
            score=float(f1_score(truth,pred,labels=range(4 if k==0 else 3),average='macro',zero_division=0))
            report['results'].setdefault(variant,{})[task]={'macro_f1':score,'train_samples':sum(map(len,yy)),'evaluation_samples':len(truth)}
            print(variant,task,score,flush=True)
        joblib.dump(models,OUT/(variant+'.joblib'),compress=3)
        r=report['results'][variant];r['weighted']=.7*r['accel']['macro_f1']+.3*r['steer']['macro_f1']
    report['delta']=report['results']['clean']['weighted']-report['results']['old']['weighted']
    report['decision']='Diagnostic only; do not replace submitted model from this exposed sensor-proxy comparison.'
    (OUT/'report.json').write_text(json.dumps(report,indent=2))
    print('weighted_delta',report['delta'],flush=True)
if __name__=='__main__':main()
