import sys,json,time,argparse
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import cv2,joblib,numpy as np,pandas as pd
from scipy.ndimage import uniform_filter1d
from sklearn.ensemble import ExtraTreesClassifier,RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score,classification_report,confusion_matrix
from solution.stage3 import extract_motion,ACCEL,STEER

ROOT=Path(__file__).resolve().parents[1]
CACHE=ROOT/'research/stage3_cache'

def feat(path,tag):
    CACHE.mkdir(parents=True,exist_ok=True);p=CACHE/(tag+'.npy')
    if p.exists():return np.load(p)
    t=time.perf_counter();x=extract_motion(path,source_fps=20.);np.save(p,x)
    print('features',tag,x.shape,round(time.perf_counter()-t,2),flush=True)
    return x

def public_data():
    df=pd.read_csv(ROOT/'Baseline/data/stage3/labels.csv');xx=[]
    for video,rows in df.groupby('ID',sort=False):
        x=feat(ROOT/'Baseline/data/stage3/videos'/f'{video}.mp4',video)
        xx.append(x[rows.frame_index.to_numpy()])
    return np.concatenate(xx),np.array([list(ACCEL).index(v) for v in df.accel_label]),np.array([list(STEER).index(v) for v in df.steer_label]),df.ID.to_numpy(),df

def candidate(name):
    if name=='linear':return make_pipeline(StandardScaler(),LogisticRegression(C=.03,class_weight='balanced',max_iter=2000,random_state=42))
    if name=='extra':return ExtraTreesClassifier(n_estimators=160,max_depth=10,min_samples_leaf=2,max_features=.4,class_weight='balanced',n_jobs=2,random_state=42)
    return RandomForestClassifier(n_estimators=160,max_depth=10,min_samples_leaf=2,max_features=.4,class_weight='balanced',n_jobs=2,random_state=42)

def mirror_features(x):
    y=x.reshape(-1,6,3,4,4,3)[:,:,:,::-1,:,:].copy()
    y[:,:,:,:,0,:]=-y[:,:,:,:,0,:][...,[0,2,1]]
    return y.reshape(x.shape)

def augment(x,y,task,enabled):
    if not enabled:return x,y
    mirrored=2-y if task=='steer' else y
    return np.concatenate([x,mirror_features(x)]),np.concatenate([y,mirrored])

def external_data():
    allx=[];alla=[];alls=[];groups=[];continuous=[]
    def fingerprint(video):
        cap=cv2.VideoCapture(str(video));views=[];i=0
        while i<=90:
            ok,img=cap.read()
            if not ok:break
            if i in (0,30,60,90):views.append(cv2.resize(cv2.cvtColor(img,cv2.COLOR_BGR2GRAY),(32,24)).astype(float)/255)
            i+=1
        cap.release();return np.array(views)
    public_fingerprints=[fingerprint(p) for p in sorted((ROOT/'Baseline/data/stage3/videos').glob('*.mp4'))]
    overlap=[]
    for manifest in sorted((ROOT/'external_data/comma2k19').glob('*_manifest.json')):
        for row in json.loads(manifest.read_text()):
            folder=ROOT/row['local'];v=folder/'video.hevc'
            if not v.exists():continue
            f=fingerprint(v)
            distance=min(np.sqrt(np.mean((f[:min(len(f),len(p))]-p[:min(len(f),len(p))])**2)) for p in public_fingerprints)
            if distance<.025:
                overlap.append({'segment':row['segment'],'rmse':float(distance)});continue
            x=feat(v,'ext_'+row['segment'].replace('/','_').replace('|','_'))
            ft=np.load(folder/'global_pose/frame_times').ravel()
            speed_t=np.load(folder/'processed_log/CAN/speed/t').ravel();speed_v=np.load(folder/'processed_log/CAN/speed/value').ravel()
            steer_t=np.load(folder/'processed_log/CAN/steering_angle/t').ravel();steer_v=np.load(folder/'processed_log/CAN/steering_angle/value').ravel()
            n=min(len(ft),len(x));ft=ft[:n]
            speed=uniform_filter1d(np.interp(ft,speed_t,speed_v),size=21)
            accel=np.gradient(speed,ft)
            steer=uniform_filter1d(np.interp(ft,steer_t,steer_v),size=11)
            # Explicit external proxy labels; these are not official DACON thresholds.
            a=np.where(speed<.3,3,np.where(accel>.25,0,np.where(accel<-.25,1,2)))
            s=np.where(steer>2.,0,np.where(steer<-2.,2,1))
            ix=np.arange(0,n,10)
            allx.append(x[ix]);alla.append(a[ix]);alls.append(s[ix]);groups.extend([row['route']]*len(ix))
            continuous.append(np.column_stack([speed[ix],accel[ix],steer[ix]]))
    (ROOT/'research/stage3_external_overlap.json').write_text(json.dumps({'excluded':overlap,'method':'4 aligned decoded-frame grayscale thumbnails RMSE < .025. Detects aligned video duplicates; cannot rule out all route/source overlap.'},indent=2),encoding='utf-8')
    if not allx:return None
    return np.concatenate(allx),np.concatenate(alla),np.concatenate(alls),np.array(groups),np.concatenate(continuous)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--external',action='store_true');parser.add_argument('--mirror',action='store_true');args=parser.parse_args()
    cv2.setNumThreads(2);x,ya,ys,g,df=public_data();ext=external_data() if args.external else None
    report={'scope':'5 public videos, leave-one-video-out; no private labels','external':ext is not None,'mirror':args.mirror,'results':{}}
    if ext is not None:report['external_counts']={'rows':len(ext[0]),'routes':len(set(ext[3])),'accel':np.bincount(ext[1],minlength=4).tolist(),'steer':np.bincount(ext[2],minlength=3).tolist()}
    best={};predtable=df.copy()
    for task,y,labels in [('accel',ya,ACCEL),('steer',ys,STEER)]:
        records=[]
        for name in ('linear','extra','forest'):
            oof=np.empty_like(y)
            for held in np.unique(g):
                test=g==held;train=~test
                if task=='steer':train=train&(ya!=3)
                xx=x[train];yy=y[train]
                if ext is not None:
                    # Full external provenance/overlap exclusion must precede use.
                    target=ext[1] if task=='accel' else ext[2]
                    keep=ext[1]!=3 if task=='steer' else np.ones(len(target),bool)
                    xx=np.concatenate([xx,ext[0][keep]]);yy=np.concatenate([yy,target[keep]])
                xx,yy=augment(xx,yy,task,args.mirror)
                m=candidate(name);m.fit(xx,yy);oof[test]=m.predict(x[test])
            mask=ya!=3 if task=='steer' else np.ones(len(y),bool)
            score=f1_score(y[mask],oof[mask],labels=np.arange(len(labels)),average='macro',zero_division=0)
            record={'name':name,'macro_f1':score,'confusion':confusion_matrix(y[mask],oof[mask],labels=np.arange(len(labels))).tolist(),'class_report':classification_report(y[mask],oof[mask],labels=np.arange(len(labels)),target_names=list(labels),zero_division=0,output_dict=True)}
            records.append(record);predtable[f'{task}_{name}']=labels[oof]
            print(task,name,score,flush=True)
        winner=max(records,key=lambda r:r['macro_f1']);model=candidate(winner['name'])
        keep=ya!=3 if task=='steer' else np.ones(len(y),bool)
        xx=x[keep];yy=y[keep]
        if ext is not None:
            target=ext[1] if task=='accel' else ext[2]
            keep=ext[1]!=3 if task=='steer' else np.ones(len(target),bool)
            xx=np.concatenate([xx,ext[0][keep]]);yy=np.concatenate([yy,target[keep]])
        xx,yy=augment(xx,yy,task,args.mirror)
        model.fit(xx,yy);best[task]=model;report['results'][task]=records;report[task+'_selected']=winner['name']
    report['selected_oof_stage3']=.7*max(r['macro_f1'] for r in report['results']['accel'])+.3*max(r['macro_f1'] for r in report['results']['steer'])
    report['selection_caveat']='Model choice uses same 5-fold OOF estimates; optimistic selection bias. Not independent final performance.'
    out=ROOT/'solution/model/stage3';out.mkdir(parents=True,exist_ok=True)
    best['feature_version']='dis256_roi144_temporal6_v1';best['trained_with_external']=ext is not None
    suffix='_external' if ext is not None else '_public'
    if args.mirror:suffix+='_mirror'
    joblib.dump(best,out/('motion_model'+(suffix if ext is not None or args.mirror else '')+'.joblib'),compress=3)
    (ROOT/f'research/stage3_validation{suffix}.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    predtable.to_csv(ROOT/f'research/stage3_oof{suffix}.csv',index=False)
    print(json.dumps({'stage3_oof':report['selected_oof_stage3'],'external':ext is not None}),flush=True)
if __name__=='__main__':main()
