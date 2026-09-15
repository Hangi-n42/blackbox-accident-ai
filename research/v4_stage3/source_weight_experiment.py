"""Predeclared single 1:1 effective-source-weight candidate, no production refit."""
from pathlib import Path
import sys,json,time,hashlib
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import numpy as np,pandas as pd,joblib
from sklearn.metrics import classification_report,confusion_matrix,f1_score
from threadpoolctl import threadpool_limits
from research.stage3_temporal_experiment import load_external,external_xy,make_model,ACCEL,STEER
from research.v4_stage3.audit import protection
OUT=Path(__file__).resolve().parent

def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def save(name,value):(OUT/name).write_text(json.dumps(value,indent=2,default=str),encoding='utf8')
def scores(ya,ys,pa,ps):
    result={}
    for task,y,p,names in [('accel',ya,pa,ACCEL),('steer',ys,ps,STEER)]:
        keep=ya!=3 if task=='steer' else np.ones(len(ya),bool)
        result[task]={'macro_f1':float(f1_score(y[keep],p[keep],labels=range(len(names)),average='macro',zero_division=0)),
          'report':classification_report(y[keep],p[keep],labels=range(len(names)),target_names=names,output_dict=True,zero_division=0),
          'confusion':confusion_matrix(y[keep],p[keep],labels=range(len(names))).tolist()}
    result['stage3']=.7*result['accel']['macro_f1']+.3*result['steer']['macro_f1']
    return result

def fit(task,px,py,ex,ey,weighted):
    x=np.concatenate([px,ex]);y=np.concatenate([py,ey]);n=len(y)
    classes,counts=np.unique(y,return_counts=True);c={int(k):n/(len(classes)*v) for k,v in zip(classes,counts)}
    class_weights=np.array([c[int(v)] for v in y]);source=np.r_[np.zeros(len(py),int),np.ones(len(ey),int)]
    w=np.ones(n)
    if weighted:
        for s in (0,1):w[source==s]=(n/2)/class_weights[source==s].sum()
    effective=w*class_weights
    assert np.isclose(effective.sum(),n)
    if weighted:assert all(np.isclose(effective[source==s].sum(),n/2) for s in (0,1))
    # Same unweighted StandardScaler in both arms; only classifier sample weight changes.
    m=make_model(task);started=time.perf_counter()
    if task=='accel':m.fit(x,y,logisticregression__sample_weight=w)
    else:m.fit(x,y,sample_weight=w)
    audit={'seconds':time.perf_counter()-started,'public_rows':len(py),'external_rows':len(ey),
       'class_weight_values':c,'effective_weight_sum':float(effective.sum()),
       'effective_source_sums':{str(s):float(effective[source==s].sum()) for s in (0,1)},
       'effective_class_sums':{str(k):float(effective[y==k].sum()) for k in classes},
       'public_classes':np.unique(py).tolist(),'feature_sha256':hashlib.sha256(x.tobytes()).hexdigest(),
       'target_sha256':hashlib.sha256(y.tobytes()).hexdigest()}
    return m,audit

def codec_diagnostic(frame,px,ya,ys):
    from solution.stage3_fast import extract_motion
    import cv2
    cv2.setNumThreads(2);m=joblib.load(ROOT/'model/stage3/motion_model.joblib')
    original={t:m[t].predict(px).astype(int) for t in ('accel','steer')};corrected={t:np.empty(len(frame),int) for t in ('accel','steer')}
    for video in frame.ID.unique():
        keep=(frame.ID==video).to_numpy();x=extract_motion(ROOT/'artifacts/public_eval_10hz/stage3/videos'/f'{video}.mp4',source_fps=10.)
        index=frame.loc[keep,'frame_index'].to_numpy()//2
        for t in corrected:corrected[t][keep]=m[t].predict(x)[index].astype(int)
    result={'scope':'Fixed production model already trained on these public labels: training-fit sensitivity diagnostic, NOT OOF accuracy or candidate selection.',
      'original_pixels':scores(ya,ys,original['accel'],original['steer']),
      'crf18_corrected_10hz':scores(ya,ys,corrected['accel'],corrected['steer']),
      'changed_sparse_rows':{t:int((original[t]!=corrected[t]).sum()) for t in original}}
    save('codec_sparse_diagnostic.json',result)
    table=frame.copy()
    for t,names in [('accel',ACCEL),('steer',STEER)]:
        table['original_'+t]=np.array(names)[original[t]];table['crf18_'+t]=np.array(names)[corrected[t]]
    table.to_csv(OUT/'codec_sparse_predictions.csv',index=False)
    print('codec sparse',result['changed_sparse_rows'],flush=True)

def main():
    if (OUT/'experiment_report.json').exists():raise FileExistsError('Preserve completed single experiment')
    audit=json.loads((OUT/'audit_report.json').read_text());can=audit['public_can']
    assert audit['same_pixel_time_gate_passed'] and can['full_decoded_pixel_sequence_matches_public'] and can['counts_match'] and can['timestamp_monotonic']
    assert not any(r['count_match'] is False or not r['positive_time_steps'] for r in audit['external_timestamp_integrity'])
    started=time.perf_counter();before=protection();external=load_external()
    rav=[r for r in external if r['vehicle']=='b0c9d2329ad1606b'];civic=[r for r in external if r['vehicle']=='99c94dc769b5d96e']
    assert len(rav)==11 and len(civic)==12
    plan={'timestamp_unix':time.time(),'candidate_count':1,'baseline':'same rows, class_weight balanced, unit sample weights',
      'candidate':'For each source s, sample_weight=N/(2*sum(class_weight_i for i in source s)); resulting combined class*sample weight sums are N/2 each source and total N.',
      'source_codes':{'0':'official public GT','1':'external CAN proxy'},
      'controls':'Same features, rows, y, class_weight numeric values, seed42, hyperparameters and unweighted scaler. No replication, no threshold search, no class-bias tuning.',
      'unavoidable_class_mass_change':'Source weighting changes effective class mixture. In constant-only public stress, preserving source1:1 AND baseline class totals is mathematically impossible. Every fold logs resulting class totals.',
      'public_oof':'5 public video groups, each trained on other4 official videos and all23 external segments; existing small-sample selection bias and unknown remaining public origins remain.',
      'external_stress':{'train_public':['OPEN_001'],'train_external':[r['route'] for r in rav],
        'validation_external':[r['route'] for r in civic],
        'validation_unit':'all10Hz proxy rows, stationary GT masks steering',
        'independence':'Validation vehicle absent from this model fit; only known-RAV4 public clip allowed. Unknown-source public4 clips excluded. Data previously used in other experiments, not a fresh untouched test set.',
        'limits':'One direction, two vehicles, one highway family. Public training10 labels are all CONSTANT. Stress model is not final full-public candidate.'},
      'existing_17_6':'Already used development split; no additional repeated route experiment since source stress provides the intended new separation.',
      'continuation_gate':'No final checkpoint/refit. If candidate improves neither same-condition public OOF nor source stress over baseline, stop with no candidate. Any improvement still requires root review; old public50 alone never auto-adopts.',
      'gpu':False,'threads':2,'audit_sha256':sha(OUT/'audit_report.json'),'script_sha256':sha(Path(__file__)),
      'model_params':{t:make_model(t).get_params() for t in ('accel','steer')},'protected_before':before}
    save('experiment_plan_frozen.json',plan)
    frame=pd.read_csv(ROOT/'Baseline/data/stage3/labels.csv');px=np.stack([np.load(ROOT/'research/stage3_cache'/f'{r.ID}.npy')[r.frame_index] for r in frame.itertuples()])
    ya=np.array([ACCEL.index(v) for v in frame.accel_label]);ys=np.array([STEER.index(v) for v in frame.steer_label]);g=frame.ID.to_numpy()
    codec_diagnostic(frame,px,ya,ys)
    public={};source_results={};fitting=[];predtable=frame.copy()
    for name,weighted in [('baseline',False),('source_equal',True)]:
        p={}
        for task,y in [('accel',ya),('steer',ys)]:
            pred=np.empty_like(y);ex,ey=external_xy(external,task)
            for held in frame.ID.unique():
                train=g!=held;test=~train
                if task=='steer':train&=ya!=3
                m,detail=fit(task,px[train],y[train],ex,ey,weighted);fitting.append({'scope':'public_oof','arm':name,'task':task,'held':held,**detail})
                pred[test]=m.predict(px[test]).astype(int)
                print('public',name,task,held,round(detail['seconds'],2),flush=True)
            p[task]=pred;predtable[name+'_'+task]=np.array(ACCEL if task=='accel' else STEER)[pred]
        public[name]=scores(ya,ys,p['accel'],p['steer']);save('public_oof_partial.json',public);predtable.to_csv(OUT/'public_oof_predictions.csv',index=False)
        sp={};vx=np.concatenate([r['x_dense'] for r in civic]);vya=np.concatenate([r['accel'] for r in civic]);vys=np.concatenate([r['steer'] for r in civic])
        for task,y in [('accel',ya),('steer',ys)]:
            train=g=='OPEN_001'
            if task=='steer':train&=ya!=3
            ex,ey=external_xy(rav,task);m,detail=fit(task,px[train],y[train],ex,ey,weighted)
            fitting.append({'scope':'source_stress','arm':name,'task':task,**detail});sp[task]=m.predict(vx).astype(int)
            print('source stress',name,task,round(detail['seconds'],2),flush=True)
        source_results[name]=scores(vya,vys,sp['accel'],sp['steer']);save('source_stress_partial.json',source_results)
        np.savez_compressed(OUT/f'source_stress_{name}_predictions.npz',gt_accel=vya,gt_steer=vys,accel=sp['accel'],steer=sp['steer'])
    after=protection();assert before==after
    dp=public['source_equal']['stage3']-public['baseline']['stage3'];ds=source_results['source_equal']['stage3']-source_results['baseline']['stage3']
    report={'public':public,'source_stress':source_results,'public_stage3_delta':dp,'source_stress_stage3_delta':ds,
       'status':'stop_no_gain_in_either_diagnostic' if dp<=0 and ds<=0 else 'root_review_required_no_auto_adoption',
       'protected_unchanged':before==after,'protected_after':after,'training_details':fitting,'seconds':time.perf_counter()-started,
       'final_checkpoint_created':False,'production_refit_performed':False}
    save('experiment_report.json',report)
    print(json.dumps({'public':{k:v['stage3'] for k,v in public.items()},'source_stress':{k:v['stage3'] for k,v in source_results.items()},'delta_public':dp,'delta_source':ds,'status':report['status'],'seconds':report['seconds']},indent=2),flush=True)

if __name__=='__main__':
    with threadpool_limits(limits=2):main()
