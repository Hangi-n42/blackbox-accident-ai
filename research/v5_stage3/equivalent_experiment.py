"""Accuracy experiment only after the separately frozen equivalent-optimization gates pass."""
from pathlib import Path
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='2'
import sys,json,time,hashlib
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import cv2,numpy as np,pandas as pd,joblib
from unittest.mock import patch
from threadpoolctl import threadpool_limits
from solution import stage3_v5_fast as candidate
from research.stage3_temporal_experiment import load_external,external_xy,make_model,ACCEL,STEER
from research.v4_stage3.source_weight_experiment import scores
from research.v4_stage3.audit import protection
OUT=Path(__file__).resolve().parent/'equivalent_optimization'

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(n,v):(OUT/n).write_text(json.dumps(v,indent=2,default=str),encoding='utf8')
def array_sha(a):return hashlib.sha256(a.tobytes()).hexdigest()

def prediction_contract():
    """Use in-memory prefix adapters around old heads, not candidate weights/performance."""
    original=joblib.load(ROOT/'model/stage3/motion_model.joblib')
    class PrefixHead:
        def __init__(self,head):self.head=head
        def predict(self,x):
            assert x.shape[1]==1128 and np.isfinite(x).all()
            return self.head.predict(x[:,:864])
    adapter={t:PrefixHead(original[t]) for t in ('accel','steer')}
    with patch.object(candidate.joblib,'load',return_value=adapter):
        predicted=candidate.predict_stage3(ROOT/'artifacts/public_eval_10hz/stage3',ROOT/'model/stage3')
    assert len(predicted)==2998
    for name,group in predicted.groupby('ID'):
        assert group.sample_index.to_list()==list(range(len(group)))
        x=np.load(OUT/'cache_v5_fast'/f'{name}_corrected10_v5_fast.npy')
        assert group.accel_label.to_list()==np.array(ACCEL)[original['accel'].predict(x[:,:864]).astype(int)].tolist()
        assert group.steer_label.to_list()==np.array(STEER)[original['steer'].predict(x[:,:864]).astype(int)].tolist()
    folder=OUT/'contract_inputs'/'videos';folder.mkdir(parents=True,exist_ok=False)
    os.link(ROOT/'artifacts/public_eval_10hz/stage3/videos/OPEN_001.mp4',folder/'RENAMED.mp4')
    with patch.object(candidate.joblib,'load',return_value=adapter):
        alone=candidate.predict_stage3(folder.parent,ROOT/'model/stage3')
        os.link(ROOT/'artifacts/public_eval_10hz/stage3/videos/OPEN_002.mp4',folder/'OTHER.mp4')
        together=candidate.predict_stage3(folder.parent,ROOT/'model/stage3')
    target=together[together.ID=='RENAMED'].reset_index(drop=True)
    assert alone.equals(target)
    reference=predicted[predicted.ID=='OPEN_001'].reset_index(drop=True).copy();reference.ID='RENAMED'
    assert alone.equals(reference)
    single=folder/'SINGLE.avi'
    writer=cv2.VideoWriter(str(single),cv2.VideoWriter_fourcc(*'FFV1'),10.,(256,144))
    assert writer.isOpened()
    writer.write(np.zeros((144,256,3),np.uint8));writer.release()
    feature=candidate.extract_motion(single,10.)
    assert feature.shape==(1,1128) and not np.any(feature)
    return {'rows':2998,'all_indices_labels_finite_passed':True,'renamed_and_other_file_invariance':True,
        'single_frame_zero_policy':True,'scope':'Contract only: in-memory existing864 heads with prefix adapter. No trained1128 head, no candidate performance claim or checkpoint.'}

def cached_external(original):
    manifests=[]
    for p in sorted((ROOT/'external_data/comma2k19').glob('*_manifest.json')):
        manifests.extend(json.loads(p.read_text()))
    metadata={r['segment']:r for r in manifests}
    result=[];audit=[]
    for row in original:
        record=metadata[row['segment']]
        tag='ext_'+row['segment'].replace('/','_').replace('|','_')
        dest=OUT/'cache_v5_fast'/(tag+'_v5_fast.npy')
        start=time.perf_counter()
        if dest.exists():
            tested=json.loads((OUT/'real_external_contract.json').read_text())
            assert tested['record']['segment']==row['segment'] and tested['full1128']['bit_equal']
            x=np.load(dest);detail={'rows':len(x),'flow_pairs':tested['flow_pairs'],'fit_failures':tested['fit_failures'],'reused_contract_verified_cache':True}
        else:
            x,detail=candidate.extract_motion(ROOT/record['local']/'video.hevc',20.,return_diagnostics=True)
        base=np.load(ROOT/'research/stage3_cache'/(tag+'.npy'))
        assert np.array_equal(x[:,:864].view(np.uint32),base.view(np.uint32))
        n=len(x)
        assert len(row['accel'])==len(x[::2]) and len(row['accel_train'])==len(x[::10])
        new=dict(row,x_train=x[::10],x_dense=x[::2])
        if not dest.exists():np.save(dest,x)
        audit.append({'segment':row['segment'],'route':row['route'],'cache':str(dest.relative_to(ROOT)),
            'prefix_bit_equal':True,'cache_sha256':sha(dest),'baseline_cache_sha256':sha(ROOT/'research/stage3_cache'/(tag+'.npy')),
            'seconds':time.perf_counter()-start,**detail})
        result.append(new)
        save('external_cache_partial.json',audit)
        print('external cache',len(result),'/23',detail['fit_failures'],round(audit[-1]['seconds'],2),flush=True)
    return result,audit

def verify_baseline_fit(prior,scope,task,x,y,held=None):
    records=[r for r in prior['training_details'] if r['scope']==scope and r['arm']=='baseline' and r['task']==task and r.get('held')==held]
    assert len(records)==1
    record=records[0]
    observed={'feature_sha256':array_sha(x[:,:864]),'target_sha256':array_sha(y)}
    assert all(observed[k]==record[k] for k in observed)
    n=len(y);classes,counts=np.unique(y,return_counts=True)
    values={str(int(k)):float(n/(len(classes)*v)) for k,v in zip(classes,counts)}
    assert values==record['class_weight_values']
    return {'scope':scope,'task':task,'held':held,**observed,'same_balanced_class_weights':True,
        'same_rows_order_targets_seed_parameters_unit_sample_weights':True}

def fit_candidate(task,x,y):
    start=time.perf_counter();m=make_model(task);w=np.ones(len(y))
    if task=='accel':m.fit(x,y,logisticregression__sample_weight=w)
    else:m.fit(x,y,sample_weight=w)
    return m,{'seconds':time.perf_counter()-start,'rows':len(y),'feature_sha256':array_sha(x),'target_sha256':array_sha(y)}

def source_gate(result,per_route):
    a=result['candidate']['accel']['macro_f1']-result['baseline']['accel']['macro_f1']
    s=result['candidate']['steer']['macro_f1']-result['baseline']['steer']['macro_f1']
    score=result['candidate']['stage3']-result['baseline']['stage3']
    positive=sum(r['stage3_delta']>1e-12 for r in per_route)
    return {'accel_delta':a,'steer_delta':s,'stage3_delta':score,'positive_routes':positive,
            'passed':a>=0 and s>=0 and score>1e-12 and positive>=7}

def main():
    if (OUT/'execution_frozen.json').exists():raise FileExistsError('Preserve single execution')
    plan=json.loads((ROOT/'research/v5_stage3/plan_frozen.json').read_text());audit=json.loads((OUT/'audit_report.json').read_text())
    assert audit['local_runtime_gate_passed']
    assert sha(ROOT/'solution/stage3_v5.py')==plan['feature_source_sha256']
    optimized_plan=json.loads((OUT/'optimization_plan_frozen.json').read_text())
    assert sha(ROOT/'solution/stage3_v5_fast.py')==optimized_plan['optimized_feature_source_sha256']
    prior=json.loads((ROOT/'research/v4_stage3/experiment_report.json').read_text())
    assert sha(ROOT/'research/v4_stage3/experiment_report.json')==plan['existing_v4_baseline_report_sha256']
    params={t:make_model(t).get_params() for t in ('accel','steer')}
    assert json.loads(json.dumps(params,default=str))==plan['model_params']
    save('execution_frozen.json',{'timestamp_unix':time.time(),'script_sha256':sha(Path(__file__)),
        'plan_sha256':sha(ROOT/'research/v5_stage3/plan_frozen.json'),'optimization_plan_sha256':sha(OUT/'optimization_plan_frozen.json'),'audit_sha256':sha(OUT/'audit_report.json'),
        'order':['prediction_contract','external_features','source_stress','only_if_source_passes_public_oof'],
        'baseline_reuse':'Requires per-fold exact existing864 input/target hashes, class weights, same current factory parameters/seed and old prediction GT arrays.',
        'no_checkpoint_or_full_refit':True})
    started=time.perf_counter();before=protection();cv2.setNumThreads(2)
    contract=prediction_contract();save('prediction_contract.json',contract)
    original=load_external()
    rav0=[r for r in original if r['vehicle']=='b0c9d2329ad1606b'];civic0=[r for r in original if r['vehicle']=='99c94dc769b5d96e']
    assert [r['route'] for r in rav0]==plan['external_stress']['train_external']
    assert [r['route'] for r in civic0]==plan['external_stress']['validation_external']
    rows,cache_audit=cached_external(original)
    rav=[r for r in rows if r['vehicle']=='b0c9d2329ad1606b'];civic=[r for r in rows if r['vehicle']=='99c94dc769b5d96e']
    frame=pd.read_csv(ROOT/'Baseline/data/stage3/labels.csv')
    px=np.stack([np.load(OUT/'cache_v5_fast'/f'{r.ID}_v5_fast.npy')[r.frame_index] for r in frame.itertuples()])
    ya=np.array([ACCEL.index(v) for v in frame.accel_label]);ys=np.array([STEER.index(v) for v in frame.steer_label]);g=frame.ID.to_numpy()
    vx=np.concatenate([r['x_dense'] for r in civic]);vya=np.concatenate([r['accel'] for r in civic]);vys=np.concatenate([r['steer'] for r in civic])
    old=np.load(ROOT/'research/v4_stage3/source_stress_baseline_predictions.npz')
    assert np.array_equal(old['gt_accel'],vya) and np.array_equal(old['gt_steer'],vys)
    sp={};fits=[];proof=[]
    for task,y in [('accel',ya),('steer',ys)]:
        train=g=='OPEN_001'
        if task=='steer':train&=ya!=3
        ex,ey=external_xy(rav,task);x=np.concatenate([px[train],ex]);labels=np.concatenate([y[train],ey])
        proof.append(verify_baseline_fit(prior,'source_stress',task,x,labels))
        m,detail=fit_candidate(task,x,labels);sp[task]=m.predict(vx).astype(int)
        fits.append({'scope':'source_stress','task':task,**detail})
        print('source candidate',task,round(detail['seconds'],2),flush=True)
    result={'baseline':scores(vya,vys,old['accel'],old['steer']),'candidate':scores(vya,vys,sp['accel'],sp['steer'])}
    assert result['baseline']==prior['source_stress']['baseline']
    per_route=[];offset=0
    for r in civic:
        n=len(r['accel']);ix=slice(offset,offset+n);offset+=n
        b=scores(vya[ix],vys[ix],old['accel'][ix],old['steer'][ix]);c=scores(vya[ix],vys[ix],sp['accel'][ix],sp['steer'][ix])
        per_route.append({'route':r['route'],'baseline':b,'candidate':c,'stage3_delta':c['stage3']-b['stage3']})
    gate=source_gate(result,per_route)
    np.savez_compressed(OUT/'source_predictions.npz',gt_accel=vya,gt_steer=vys,baseline_accel=old['accel'],baseline_steer=old['steer'],candidate_accel=sp['accel'],candidate_steer=sp['steer'])
    save('source_results.json',result);save('source_per_route.json',per_route);save('source_gate_before_public.json',gate)
    save('baseline_reuse_proof.json',proof)
    print('SOURCE GATE',json.dumps(gate),flush=True)
    public=None;pg=None
    if gate['passed']:
        predictions={};oldpub=pd.read_csv(ROOT/'research/v4_stage3/public_oof_predictions.csv')
        assert frame.equals(oldpub[frame.columns])
        for task,y in [('accel',ya),('steer',ys)]:
            pred=np.empty_like(y);ex,ey=external_xy(rows,task)
            for held in frame.ID.unique():
                train=g!=held;test=~train
                if task=='steer':train&=ya!=3
                x=np.concatenate([px[train],ex]);labels=np.concatenate([y[train],ey])
                proof.append(verify_baseline_fit(prior,'public_oof',task,x,labels,held))
                m,detail=fit_candidate(task,x,labels);pred[test]=m.predict(px[test]).astype(int)
                fits.append({'scope':'public_oof','task':task,'held':held,**detail})
                print('public candidate',task,held,round(detail['seconds'],2),flush=True)
            predictions[task]=pred
        bpa=np.array([ACCEL.index(v) for v in oldpub.baseline_accel]);bps=np.array([STEER.index(v) for v in oldpub.baseline_steer])
        public={'baseline':scores(ya,ys,bpa,bps),'candidate':scores(ya,ys,predictions['accel'],predictions['steer'])}
        assert public['baseline']==prior['public']['baseline']
        da=public['candidate']['accel']['macro_f1']-public['baseline']['accel']['macro_f1']
        ds=public['candidate']['steer']['macro_f1']-public['baseline']['steer']['macro_f1']
        dt=public['candidate']['stage3']-public['baseline']['stage3']
        pg={'accel_delta':da,'steer_delta':ds,'stage3_delta':dt,'passed':da>=0 and ds>=0 and dt>1e-12}
        table=oldpub[frame.columns.tolist()+['baseline_accel','baseline_steer']].copy()
        for t,names in [('accel',ACCEL),('steer',STEER)]:table['candidate_'+t]=np.array(names)[predictions[t]]
        table.to_csv(OUT/'public_predictions.csv',index=False)
        save('public_results.json',public);save('public_gate.json',pg)
    after=protection();unchanged=all(after.get(k)==v for k,v in before.items());assert unchanged
    status='rejected_source_gate_no_public_oof' if not gate['passed'] else ('root_review_required_no_refit' if pg['passed'] else 'rejected_public_gate_no_refit')
    report={'status':status,'source_stress':result,'source_gate':gate,'public':public,'public_gate':pg,
        'training_details':fits,'baseline_reuse_proof':proof,'protected_unchanged':unchanged,
        'seconds':time.perf_counter()-started,'final_checkpoint_created':False,'production_refit_performed':False}
    save('baseline_reuse_proof.json',proof);save('experiment_report.json',report)
    print('FINAL',status,flush=True)

if __name__=='__main__':
    with threadpool_limits(limits=2):main()
