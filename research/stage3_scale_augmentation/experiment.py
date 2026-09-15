"""One fixed feature-scale augmentation test, gated by external validation."""
from pathlib import Path
import hashlib,json,time,sys,shutil
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import classification_report,f1_score,confusion_matrix
from threadpoolctl import threadpool_limits
from research.stage3_temporal_experiment import load_external,external_xy,make_model,ACCEL,STEER

OUT=Path(__file__).resolve().parent
SCALES=(.75,1.,1.25)

def sha(path):
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def save(name,value):
    (OUT/name).write_text(json.dumps(value,indent=2,default=str),encoding='utf8')

def protect():
    paths=[ROOT/'solution/stage3.py',ROOT/'model/stage3/motion_model.joblib']+sorted((ROOT/'artifacts/submissions').glob('*.zip'))
    return {str(p.relative_to(ROOT)):{'size':p.stat().st_size,'mtime_ns':p.stat().st_mtime_ns,'sha256':sha(p) if p.suffix!='.zip' else None} for p in paths}

def fit(task,x,y,augmented):
    original_n=len(y)
    if augmented:
        scales=(1.,1.,1.) if augmented=='replicated' else SCALES
        x=np.concatenate([x*np.float32(scale) for scale in scales]);y=np.tile(y,len(scales))
    model=make_model(task);start=time.perf_counter();model.fit(x,y)
    return model,{'seconds':time.perf_counter()-start,'original_rows':original_n,'fitted_rows':len(y)}

def metrics(task,truth,pred):
    names=ACCEL if task=='accel' else STEER
    return {'macro_f1':float(f1_score(truth,pred,labels=range(len(names)),average='macro',zero_division=0)),
            'classification':classification_report(truth,pred,labels=range(len(names)),target_names=names,output_dict=True,zero_division=0),
            'confusion_matrix':confusion_matrix(truth,pred,labels=range(len(names))).tolist()}

def score(ya,ys,pa,ps):
    keep=ya!=3
    a=metrics('accel',ya,pa);s=metrics('steer',ys[keep],ps[keep])
    return {'accel':a,'steer':s,'stage3':.7*a['macro_f1']+.3*s['macro_f1']}

def delta(base,aug):
    result={'stage3':aug['stage3']-base['stage3']}
    for task,names in [('accel',ACCEL),('steer',STEER)]:
        result[task]={'macro_f1':aug[task]['macro_f1']-base[task]['macro_f1'],
                      'class_f1':{name:aug[task]['classification'][name]['f1-score']-base[task]['classification'][name]['f1-score'] for name in names}}
    return result

def public_oof(train):
    frame=pd.read_csv(ROOT/'Baseline/data/stage3/labels.csv')
    features={video:np.load(ROOT/'research/stage3_cache'/f'{video}.npy') for video in frame.ID.unique()}
    px=np.concatenate([features[video][rows.frame_index.to_numpy()] for video,rows in frame.groupby('ID',sort=False)])
    ya=np.array([ACCEL.index(v) for v in frame.accel_label]);ys=np.array([STEER.index(v) for v in frame.steer_label]);groups=frame.ID.to_numpy()
    output={};table=frame.copy();times={}
    for name,aug in [('baseline',False),('replicated','replicated'),('augmented',True)]:
        predictions={};times[name]=[]
        for task,y in [('accel',ya),('steer',ys)]:
            ex,ey=external_xy(train,task);pred=np.empty_like(y)
            for held in frame.ID.unique():
                training=groups!=held;test=~training
                if task=='steer':training&=ya!=3
                x=np.concatenate([px[training],ex]);target=np.concatenate([y[training],ey])
                model,timing=fit(task,x,target,aug);times[name].append({'task':task,'held_out_video':held,**timing})
                # Exactly the preceding experiment's dense 10Hz/floor-index OOF convention.
                dense=features[held][::2];indices=frame.loc[test,'frame_index'].to_numpy()//2
                pred[test]=model.predict(dense)[indices].astype(int)
                print('public',name,task,held,round(timing['seconds'],2),flush=True)
            predictions[task]=pred;table[f'{name}_{task}']=np.array(ACCEL if task=='accel' else STEER)[pred]
        output[name]=score(ya,ys,predictions['accel'],predictions['steer'])
    output['delta']=delta(output['baseline'],output['augmented']);output['delta_vs_replicated']=delta(output['replicated'],output['augmented']);output['fit_times']=times
    output['existing_external23_reference']=json.loads((ROOT/'research/stage3_selection.json').read_text())['stage3_oof']
    output['delta_existing_reference']=output['augmented']['stage3']-output['existing_external23_reference']
    output['caveat']='Development diagnostic on the same 50 sparse public labels previously used for model selection; subject to selection bias. Not independent test accuracy. External six validation routes remain excluded.'
    table.to_csv(OUT/'public_oof_comparison.csv',index=False)
    save('public_oof.json',output)
    return output

def main():
    if (OUT/'report.json').exists():raise FileExistsError('Preserve completed single experiment')
    start=time.perf_counter();before=protect()
    splitpath=ROOT/'research/stage3_temporal_v1/external_split.json'
    split=json.loads(splitpath.read_text());shutil.copyfile(splitpath,OUT/'external_split.json')
    config={'hypothesis':'Training on fixed positive global feature scales may reduce sensitivity to flow magnitude. This is not a camera-FOV transformation.',
            'scales':SCALES,'feature_dimensions':864,'augmentation_training_only':True,'validation_features':'original only',
            'sample_weight':'All replicated samples have unit weight; threefold sample count can also affect effective regularization and RF bootstrap/min-leaf behavior.',
            'threads':2,'gpu':False,'random_state':42,'gate':'Public OOF runs only if augmented external stage3 score is strictly greater than baseline.',
            'split_source':str(splitpath.relative_to(ROOT)),'split_source_sha256':sha(splitpath),
            'model_parameters':{task:make_model(task).get_params() for task in ('accel','steer')},
            'no_smoothing':True,'protected_files_before':before}
    if not (OUT/'config_frozen_before_fit.json').exists():save('config_frozen_before_fit.json',config)
    source_paths=[ROOT/'research/stage3_temporal_experiment.py',ROOT/'research/stage3_external_overlap.json',
                  ROOT/'external_data/comma2k19/LICENSE']+sorted((ROOT/'external_data/comma2k19').glob('*_manifest.json'))
    save('sources.json',{'data':'comma2k19 local licensed subset and existing flow feature cache',
                         'official_repository':'https://github.com/commaai/comma2k19',
                         'license':'MIT; exact local text copied as comma2k19_LICENSE',
                         'files':[{'path':str(p.relative_to(ROOT)),'sha256':sha(p)} for p in source_paths]})
    shutil.copyfile(ROOT/'external_data/comma2k19/LICENSE',OUT/'comma2k19_LICENSE')
    rows=load_external();lookup={r['route']:r for r in rows}
    train=[lookup[r] for r in split['train']];val=[lookup[r] for r in split['validation']]
    assert len(train)==17 and len(val)==6 and set(split['train']).isdisjoint(split['validation'])
    xval=np.concatenate([r['x_dense'] for r in val]);ya=np.concatenate([r['accel'] for r in val]);ys=np.concatenate([r['steer'] for r in val])
    external={};allpred={}
    for name,aug in [('baseline',False),('replicated','replicated'),('augmented',True)]:
        models={};timings={};pred={}
        for task in ('accel','steer'):
            x,y=external_xy(train,task);models[task],timings[task]=fit(task,x,y,aug)
            pred[task]=models[task].predict(xval).astype(int)
            print('external',name,task,timings[task],flush=True)
        external[name]=score(ya,ys,pred['accel'],pred['steer']);external[name]['fit_times']=timings
        joblib.dump(models,OUT/f'external17_{name}.joblib',compress=3);allpred[name]=pred
    external['delta']=delta(external['baseline'],external['augmented'])
    external['delta_vs_replicated']=delta(external['replicated'],external['augmented'])
    external['per_route']={};offset=0
    for r in val:
        n=len(r['accel']);route={}
        for name in allpred:route[name]=score(r['accel'],r['steer'],allpred[name]['accel'][offset:offset+n],allpred[name]['steer'][offset:offset+n])
        route['delta']=delta(route['baseline'],route['augmented']);route['delta_vs_replicated']=delta(route['replicated'],route['augmented']);external['per_route'][r['route']]=route;offset+=n
    gate=external['delta']['stage3']>0 and external['delta_vs_replicated']['stage3']>0
    save('external_validation.json',external)
    save('gate_before_public.json',{'external_stage3_improved':gate,'delta':external['delta'],'public_oof_authorized_by_gate':gate,'elapsed_seconds':time.perf_counter()-start})
    np.savez_compressed(OUT/'external_validation_predictions.npz',gt_accel=ya,gt_steer=ys,
                        **{f'{name}_{task}':allpred[name][task] for name in allpred for task in ('accel','steer')})
    print('external gate',gate,'baseline',external['baseline']['stage3'],'augmented',external['augmented']['stage3'],flush=True)
    public=public_oof(train) if gate else None
    after=protect();assert before==after
    report={'status':'external_improvement_public_diagnostic_completed' if gate else 'do_not_adopt_external_not_improved',
            'external':external,'public_oof_executed':gate,'public':public,'protected_files_unchanged':before==after,
            'protected_files_after':after,'elapsed_seconds':time.perf_counter()-start,
            'caveats':['CAN proxy thresholds are not official competition thresholds.',
                       'External validation has already supported earlier experiments and is no longer a wholly untouched test set.',
                       'Positive global scaling is a feature-space stress augmentation, not a calibrated physical camera transformation.',
                       'Replicating samples three times also affects effective regularization/bootstrap; benefit cannot be attributed solely to scale invariance.',
                       'No production model, solution code, or ZIP was changed. No submission adoption performed.']}
    save('report.json',report)
    print(json.dumps({'status':report['status'],'external_delta':external['delta'],'public_delta':None if public is None else public['delta'],'elapsed_seconds':report['elapsed_seconds']},indent=2),flush=True)

if __name__=='__main__':
    with threadpool_limits(limits=2):main()
