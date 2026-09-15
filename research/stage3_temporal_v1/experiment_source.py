"""One bounded experiment: external-route-selected fixed probability smoothing.

No production source/model/ZIP is modified. No public label selects the window.
"""
from pathlib import Path
import sys,json,time,hashlib,shutil
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
import joblib
from scipy.ndimage import uniform_filter1d
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score,classification_report
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'research/stage3_temporal_v1'
ACCEL=['ACCELERATING','DECELERATING','CONSTANT','STOPPED']
STEER=['LEFT','STRAIGHT','RIGHT']
WINDOWS=(1,3,5,11,21)  # Fixed before evaluating any held-out labels.

def digest(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def make_model(task):
    if task=='accel':return make_pipeline(StandardScaler(),LogisticRegression(C=.03,class_weight='balanced',max_iter=2000,random_state=42))
    return RandomForestClassifier(n_estimators=160,max_depth=10,min_samples_leaf=2,max_features=.4,class_weight='balanced',n_jobs=2,random_state=42)

def load_external():
    excluded={r['segment'] for r in json.loads((ROOT/'research/stage3_external_overlap.json').read_text())['excluded']}
    result=[]
    for p in sorted((ROOT/'external_data/comma2k19').glob('*_manifest.json')):
        for row in json.loads(p.read_text()):
            if row['segment'] in excluded:continue
            base=ROOT/row['local'];tag='ext_'+row['segment'].replace('/','_').replace('|','_')
            x=np.load(ROOT/'research/stage3_cache'/(tag+'.npy'))
            ft=np.load(base/'global_pose/frame_times').ravel();n=min(len(ft),len(x));ft=ft[:n]
            st=np.load(base/'processed_log/CAN/speed/t').ravel();sv=np.load(base/'processed_log/CAN/speed/value').ravel()
            at=np.load(base/'processed_log/CAN/steering_angle/t').ravel();av=np.load(base/'processed_log/CAN/steering_angle/value').ravel()
            speed=uniform_filter1d(np.interp(ft,st,sv),size=21)
            accel=np.gradient(speed,ft);angle=uniform_filter1d(np.interp(ft,at,av),size=11)
            ya=np.where(speed<.3,3,np.where(accel>.25,0,np.where(accel<-.25,1,2)))
            ys=np.where(angle>2.,0,np.where(angle<-2.,2,1))
            result.append({'route':row['route'],'segment':row['segment'],'vehicle':row['route'].split('/')[-1].split('|')[0],
                           'x_train':x[:n:10],'accel_train':ya[::10],'steer_train':ys[::10],
                           'x_dense':x[:n:2],'accel':ya[::2],'steer':ys[::2]})
    return result

def split_routes(rows):
    # Exactly three validation routes per vehicle, independent of their labels.
    validation=set()
    for vehicle in sorted({r['vehicle'] for r in rows}):
        routes={r['route'] for r in rows if r['vehicle']==vehicle}
        ordered=sorted(routes,key=lambda s:hashlib.sha256(('stage3-smooth-v1:'+s).encode()).hexdigest())
        validation.update(ordered[:3])
    return [r for r in rows if r['route'] not in validation],[r for r in rows if r['route'] in validation]

def external_xy(rows,task):
    x=np.concatenate([r['x_train'] for r in rows]);y=np.concatenate([r[task+'_train'] for r in rows])
    keep=np.concatenate([r['accel_train'] for r in rows])!=3 if task=='steer' else np.ones(len(y),bool)
    return x[keep],y[keep]

def probability(model,x,classes):
    result=np.zeros((len(x),classes));result[:,model.classes_.astype(int)]=model.predict_proba(x)
    return result

def smooth(p,window):
    return p if window==1 else uniform_filter1d(p,size=window,axis=0,mode='nearest')

def score_sequences(rows,probs,window):
    result={}
    for task,labels in [('accel',ACCEL),('steer',STEER)]:
        yy=[];pp=[]
        for r,prob in zip(rows,probs):
            keep=r['accel']!=3 if task=='steer' else np.ones(len(r['accel']),bool)
            yy.extend(r[task][keep]);pp.extend(smooth(prob[task],window).argmax(1)[keep])
        result[task]=float(f1_score(yy,pp,labels=range(len(labels)),average='macro',zero_division=0))
    result['stage3']=.7*result['accel']+.3*result['steer']
    return result

def write_json(path,value):path.write_text(json.dumps(value,indent=2),encoding='utf-8')

def main():
    if OUT.exists():raise FileExistsError(f'Preserve previous experiment: {OUT}')
    OUT.mkdir(parents=True)
    shutil.copyfile(__file__,OUT/'experiment_source.py')
    protected=[ROOT/'model/stage3/motion_model.joblib',ROOT/'solution/stage3.py']+sorted((ROOT/'artifacts/submissions').glob('*.zip'))
    # Large ZIP contents are not reread; modification evidence uses size/mtime.
    before={str(p.relative_to(ROOT)):{'size':p.stat().st_size,'mtime_ns':p.stat().st_mtime_ns,
             'sha256':digest(p) if p.suffix!='.zip' else None} for p in protected}
    t=time.perf_counter();external=load_external();train,val=split_routes(external)
    assert len(train)==17 and len(val)==6 and not({r['route'] for r in train}&{r['route'] for r in val})
    split={'train':[r['route'] for r in train],'validation':[r['route'] for r in val],
           'policy':'SHA256(stage3-smooth-v1:route); first three per vehicle held out; no label stratification',
           'validation_never_used_for_model_fitting':True,'validation_dense_counts':{task:np.bincount(np.concatenate([r[task] for r in val]),minlength=n).tolist() for task,n in [('accel',4),('steer',3)]}}
    write_json(OUT/'external_split.json',split)
    print('Split fixed: 17 train routes, 6 validation routes',flush=True)
    models={}
    for task in ('accel','steer'):
        x,y=external_xy(train,task);model=make_model(task);fit_start=time.perf_counter();model.fit(x,y);models[task]=model
        print('External-only fit',task,round(time.perf_counter()-fit_start,2),flush=True)
    probs=[{task:probability(models[task],r['x_dense'],n) for task,n in [('accel',4),('steer',3)]} for r in val]
    records=[{'window_frames':window,'seconds':0 if window==1 else window/10,**score_sequences(val,probs,window)} for window in WINDOWS]
    # Max stage score, break exact ties toward the shorter window.
    chosen=max(records,key=lambda r:(r['stage3'],-r['window_frames']))
    frozen={'window_frames':chosen['window_frames'],'sample_rate_hz':10,'padding':'nearest','operation':'arithmetic mean of class probabilities',
            'chosen_only_from_external_validation':True,'external_validation_scores':records,'selected_external_score':chosen,
            'frozen_before_public_oof':True,'selection_elapsed_seconds':time.perf_counter()-t}
    write_json(OUT/'frozen_smoothing.json',frozen)
    print('Window frozen from external validation:',json.dumps(chosen),flush=True)
    for i,r in enumerate(val):
        np.savez_compressed(OUT/f'external_validation_probabilities_{i:02}.npz',accel=probs[i]['accel'],steer=probs[i]['steer'],gt_accel=r['accel'],gt_steer=r['steer'])
    if chosen['window_frames']==1:
        outcome={'status':'no_change_selected','reason':'External validation selected identity. It produces the same predictions as raw by definition; no public-label window search or new submission candidate is justified.',
                 'public_oof_executed':False,'frozen_smoothing':frozen,'elapsed_seconds':time.perf_counter()-t}
    else:
        # Public labels are read only after the smoothing setting is frozen.
        frame=pd.read_csv(ROOT/'Baseline/data/stage3/labels.csv');px=[]
        public=[]
        for video,rows in frame.groupby('ID',sort=False):
            features=np.load(ROOT/'research/stage3_cache'/f'{video}.npy')
            public.append({'ID':video,'rows':rows,'x_dense':features[::2]})
            px.append(features[rows.frame_index.to_numpy()])
        px=np.concatenate(px);ya=np.array([ACCEL.index(v) for v in frame.accel_label]);ys=np.array([STEER.index(v) for v in frame.steer_label]);groups=frame.ID.to_numpy()
        public_results=[];fit_seconds=[];post_seconds=[]
        predtable=frame.copy()
        for task,labels,y in [('accel',ACCEL,ya),('steer',STEER,ys)]:
            oof_raw=np.empty_like(y);oof_smooth=np.empty_like(y);ex,ey=external_xy(train,task)
            for entry in public:
                held=entry['ID'];train_mask=groups!=held;test_mask=~train_mask
                if task=='steer':train_mask&=ya!=3
                m=make_model(task);started=time.perf_counter();m.fit(np.concatenate([px[train_mask],ex]),np.concatenate([y[train_mask],ey]));fit_seconds.append(time.perf_counter()-started)
                p=probability(m,entry['x_dense'],len(labels));indices=entry['rows'].frame_index.to_numpy()//2
                oof_raw[test_mask]=p.argmax(1)[indices]
                started=time.perf_counter();ps=smooth(p,chosen['window_frames']);post_seconds.append(time.perf_counter()-started)
                oof_smooth[test_mask]=ps.argmax(1)[indices]
                np.savez_compressed(OUT/f'public_oof_probabilities_{task}_{held}.npz',probability=p)
                print('Public held-out fit',task,held,round(fit_seconds[-1],2),flush=True)
            mask=ya!=3 if task=='steer' else np.ones(len(y),bool)
            result={'task':task,'raw_macro_f1':float(f1_score(y[mask],oof_raw[mask],labels=range(len(labels)),average='macro',zero_division=0)),
                    'smoothed_macro_f1':float(f1_score(y[mask],oof_smooth[mask],labels=range(len(labels)),average='macro',zero_division=0)),
                    'raw_report':classification_report(y[mask],oof_raw[mask],labels=range(len(labels)),target_names=labels,output_dict=True,zero_division=0),
                    'smoothed_report':classification_report(y[mask],oof_smooth[mask],labels=range(len(labels)),target_names=labels,output_dict=True,zero_division=0)}
            public_results.append(result);predtable[task+'_raw']=np.array(labels)[oof_raw];predtable[task+'_smooth']=np.array(labels)[oof_smooth]
        raw=.7*public_results[0]['raw_macro_f1']+.3*public_results[1]['raw_macro_f1']
        filtered=.7*public_results[0]['smoothed_macro_f1']+.3*public_results[1]['smoothed_macro_f1']
        existing=json.loads((ROOT/'research/stage3_selection.json').read_text())['stage3_oof']
        promote=filtered>raw and filtered>existing
        outcome={'status':'candidate_improved' if promote else 'do_not_adopt','public_oof_executed':True,'public_results':public_results,
                 'raw_stage3_same_external17':raw,'smoothed_stage3_same_external17':filtered,'existing_external23_stage3_reference':existing,
                 'delta_same_data':filtered-raw,'delta_existing_submission':filtered-existing,'adoption_criterion_met':promote,
                 'frozen_smoothing':frozen,'training_seconds_sum':sum(fit_seconds),'postprocessing_seconds_for_public5_both_heads':sum(post_seconds),'elapsed_seconds':time.perf_counter()-t}
        predtable.to_csv(OUT/'public_oof_comparison.csv',index=False)
        # Only train an additional full-data candidate when both acceptance criteria hold.
        if promote:
            final={}
            for task,y in [('accel',ya),('steer',ys)]:
                ex,ey=external_xy(train,task);keep=ya!=3 if task=='steer' else np.ones(len(y),bool)
                m=make_model(task);m.fit(np.concatenate([px[keep],ex]),np.concatenate([y[keep],ey]));final[task]=m
            final['smoothing']=frozen;final['feature_version']='dis256_roi144_temporal6_v1'
            joblib.dump(final,OUT/'candidate_model.joblib',compress=3)
            outcome['candidate_sha256']=digest(OUT/'candidate_model.joblib')
    after={str(p.relative_to(ROOT)):{'size':p.stat().st_size,'mtime_ns':p.stat().st_mtime_ns,
            'sha256':digest(p) if p.suffix!='.zip' else None} for p in protected}
    outcome['protected_files_unchanged']=before==after
    outcome['protected_files_before']=before;outcome['protected_files_after']=after
    outcome['caveats']=['CAN proxy category thresholds are not official DACON thresholds.','Only 50 public sparse labels; no claim about unseen transition-boundary timing between them.','Fixed symmetric smoothing can delay or suppress short true events.','External 6 validation routes remain excluded from every model fit.','Existing submission reference used 23 external routes, so its difference is not an isolated smoothing effect.']
    assert outcome['protected_files_unchanged']
    write_json(OUT/'report.json',outcome)
    print(json.dumps({k:v for k,v in outcome.items() if k not in ['frozen_smoothing','public_results','protected_files_before','protected_files_after']},indent=2),flush=True)

if __name__=='__main__':
    with threadpool_limits(limits=2):main()
