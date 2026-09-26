"""Read-only checkpoint comparison on 18 previously reserved window centers. No fitting."""
from pathlib import Path
import sys, json, hashlib, importlib.util, time, subprocess
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import cv2, joblib
from sklearn.metrics import f1_score, precision_recall_fscore_support, confusion_matrix
from threadpoolctl import threadpool_limits

O=Path(__file__).resolve().parent; R=O.parents[1]
A=R/'artifacts/stage3_followthrough_20260920/data_sources/acquired'
P=R/'model/stage3/motion_model.joblib'
E=R/'artifacts/stage3_zod_20260920/models/comma_only.joblib'
F=R/'artifacts/stage3_factor_comparison_20260918/run.py'
read=lambda p:json.loads(p.read_text())
def write(p,v): p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n')
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def aligned(w):return A/'aligned'/(w['segment'].replace('/','_').replace('|','_')+'.npz')
def opposite(y,p):return ((y==0)&(p==1))|((y==1)&(p==0))
def stat(p):return {'bytes':p.stat().st_size,'mtime_ns':p.stat().st_mtime_ns}

def freeze():
    assert not (O/'freeze.json').exists()
    w=read(A/'reserved_windows.json');assert len(w)==18
    assert np.bincount([r['label'] for r in w],minlength=4).tolist()==[6,6,6,0]
    routes={r['route'] for r in w}
    excluded={r['segment'] for r in read(R/'research/stage3_external_overlap.json')['excluded']}
    manifests=[R/'external_data/comma2k19/chunk1_manifest.json',R/'external_data/comma2k19/chunk3_manifest.json']
    historical=[r for p in manifests for r in read(p) if r['segment'] not in excluded]
    assert len(historical)==23 and routes.isdisjoint({r['route'] for r in historical})
    cs={r['id']:r for r in read(R/'artifacts/stage3_training_basis_20260917/cases.json')}
    selected=read(R/'artifacts/stage3_zod_20260920/training_manifest.json')['comma_selection']
    assert len(selected)==2395 and all(s in cs for s,i in selected)
    assert routes.isdisjoint({cs[s]['route'] for s,i in selected})
    labels={s:np.load(R/'artifacts/stage3_training_basis_20260917'/c['labels_npz'])['accel_candidate'] for s,c in cs.items()}
    counts=np.bincount([int(labels[s][i]) for s,i in selected],minlength=4).tolist()
    prod=joblib.load(P); ext=joblib.load(E)
    assert set(prod['accel'].classes_)==set(ext.classes_)=={0,1,2,3}
    assert int(prod['accel'][0].n_samples_seen_)==2811 and int(ext[0].n_samples_seen_)==2395
    assert sha(P)==read(R/'research/stage3_selection.json')['model_sha256']
    source=R/'solution/model/stage3/motion_model_external.joblib'
    source_verified=None
    if source.exists():
        original=joblib.load(source)['accel']
        source_verified=all(np.array_equal(getattr(original[0],a),getattr(prod['accel'][0],a)) for a in ['mean_','scale_']) and all(np.array_equal(getattr(original[-1],a),getattr(prod['accel'][-1],a)) for a in ['coef_','intercept_','classes_'])
        assert source_verified
    cmd=['rg','-l','--no-ignore','reserved_windows|reserved_manifest|2018-08-10--22-42-26|2018-05-18--16-00-42','artifacts','docs','research','scripts',
         '--glob','*.py','--glob','*.json','--glob','*.md','--glob','*.csv',
         '-g','!**/vendor/**','-g','!**/vjepa2_source/**','-g','!**/raw/**','-g','!**/node_modules/**','-g','!**/.venv/**','-g','!**/records/**','-g','!**/aligned/**','-g','!**/stage3_reserved_baseline_20260921/**']
    search=subprocess.run(cmd,cwd=R,text=True,capture_output=True);assert search.returncode in [0,1]
    assert read(R/'artifacts/stage3_direction_20260921/reserved_status.json')['used'] is False
    write(O/'exposure_before.json',{'search_command':cmd,'matched_files':search.stdout.splitlines(),
        'recorded_prior_model_predictions':0,'recorded_prior_training':0,'model_outcome_candidate_selection':False,
        'prior_exposure':'Sensor-based selection and54frame AI visual QA; known to researchers. Previous model comparisons explicitly left reserved data unopened.',
        'scope':'Project text/code/results records searched; no prediction/training use found. Not proof of unknown unrecorded activity or geographic independence.',
        'new_vehicles':False,'road_independence_verified':False,'public_other4_overlap_verified':False})
    provenance={'production':{'path':str(P.relative_to(R)),'sha256':sha(P),'external_rows':2761,'public_rows':50,'public_labels_in_training':True,'total_rows':2811,
        'external_class_counts':[521,570,1393,277],'source_checkpoint':str(source.relative_to(R)), 'source_parameters_match':source_verified,
        'training_route_manifest':historical,'source_report':'research/stage3_validation_external.json','scope':'Historical sensor proxy labels plus official50; full saved production acceleration head'},
        'expanded':{'path':str(E.relative_to(R)),'sha256':sha(E),'external_rows':2395,'public_rows':0,'public_labels_in_training':False,'zod_training':False,
        'class_counts':counts,'training_selection':selected,'training_routes':sorted({cs[s]['route'] for s,i in selected}),
        'source_script':'artifacts/stage3_zod_20260920/compare.py:fit(ext_sel,False,comma_only)',
        'scope':'Full external2395 saved baseline, not a held-out fold. PublicLOVO baseline2395+40 models are different and not selected here.'}}
    write(O/'model_provenance.json',provenance)
    files=[P,E,F,Path(__file__),O/'PROTOCOL.md',O/'model_provenance.json',O/'exposure_before.json',A/'reserved_windows.json',A/'reserved_manifest.json',
        R/'research/stage3_selection.json',R/'research/stage3_validation_external.json',R/'research/train_stage3.py',
        R/'artifacts/stage3_zod_20260920/training_manifest.json',R/'artifacts/stage3_zod_20260920/compare.py',
        R/'artifacts/stage3_training_basis_20260917/cases.json',R/'artifacts/stage3_direction_20260921/decision.json',
        R/'artifacts/submissions/verify_v6/model/stage2/code/solution/stage3_v5_compatible.py']
    files+=list({aligned(r) for r in w})
    for r in w:
        z=np.load(aligned(r));i=r['center_index']
        assert int(z['accel_candidate'][i])==r['label'] and bool(z['accel_use_mask'][i])
    write(O/'freeze.json',{'utc':datetime.now(timezone.utc).isoformat(),'models':provenance,'windows':w,
        'evaluation':'Exactly18 original center timestamps, same DIS864 array for both stored models; 4class argmax, no STOP renormalization. Score only present truth classes0/1/2; no steering/S3.',
        'decision':'Expanded followup only if pooledF1 and correctcount improve, eachrouteF1 nondecreases, and zero new inversions. Production baseline retained if pooledF1 and correctcount at least as high and eachrouteF1 nondecreases. Otherwise insufficient evidence. No independent/general superiority claim from18centers/two routes.',
        'inputs':{str(p.relative_to(R)):sha(p) for p in files},'video_stats':{r['video_path']:stat(R/r['video_path']) for r in w}})
    print('Frozen exact full checkpoints; reserved18 centers; no predictions yet',flush=True)

def metrics(y,p):
    pr,re,f,su=precision_recall_fscore_support(y,p,labels=[0,1,2],zero_division=0)
    cm=confusion_matrix(y,p,labels=[0,1,2,3]);den=cm.sum(0)+cm.sum(1)
    calc=np.divide(2*cm.diagonal(),den,out=np.zeros(4,float),where=den>0)[:3]
    assert np.allclose(calc,f)
    return {'n':len(y),'correct':int(np.sum(y==p)),'macro_f1_AD_C':float(np.mean(f)),
        'class_f1':f.tolist(),'class_precision':pr.tolist(),'class_recall':re.tolist(),'class_support':su.tolist(),
        'confusion_including_predicted_STOP':cm.tolist(),'opposite':int(opposite(y,p).sum()),'predicted_STOP':int(np.sum(p==3))}

def run():
    assert not (O/'predictions.csv').exists(),'This reserved comparison must run once'
    f=read(O/'freeze.json');assert all(sha(R/p)==h for p,h in f['inputs'].items())
    assert all(stat(R/p)==s for p,s in f['video_stats'].items())
    started=time.perf_counter();models={};loadcost={}
    for name,path in [('production',P),('expanded',E)]:
        t=time.perf_counter();m=joblib.load(path);models[name]=m['accel'] if name=='production' else m;loadcost[name]=time.perf_counter()-t
    spec=importlib.util.spec_from_file_location('existing_flow',F);flow=importlib.util.module_from_spec(spec);spec.loader.exec_module(flow)
    timings=[];rows=[];(O/'features').mkdir()
    for path in dict.fromkeys(w['video_path'] for w in f['windows']):
        ws=[(i,w) for i,w in enumerate(f['windows']) if w['video_path']==path];w=ws[0][1]
        c={'public':False,'raw_path':path,'labels_npz':str(aligned(w))}
        t=time.perf_counter();prev=None;raw=[];n=0;ff=flow.FrameFeatureComputer();dis=cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_FAST)
        for im in flow.frames(c):
            if prev is not None:raw.append(ff(dis.calc(prev,im,None)*10))
            prev=im;n+=1
        x=flow.combine(raw,n);assert len(x)==len(np.load(aligned(w))['time']) and np.isfinite(x).all()
        key=w['segment'].replace('/','_').replace('|','_');np.savez_compressed(O/'features'/(key+'.npz'),base=x)
        extraction=time.perf_counter()-t;ix=np.array([v['center_index'] for i,v in ws]);xx=x[ix];pred={};cost={}
        for name,m in models.items():
            t=time.perf_counter();pred[name]=m.predict_proba(xx);cost[name]=time.perf_counter()-t
            assert np.isfinite(pred[name]).all() and np.allclose(pred[name].sum(1),1)
        for j,(index,w) in enumerate(ws):
            row={'window':index,'segment':w['segment'],'route':w['route'],'vehicle':w['vehicle_model'],'center_index':w['center_index'],'center_seconds':w['center_seconds'],
                 'truth':w['label'],'truth_kind':w['truth_type'],'sensor_speed':w['speed_m_s'],'sensor_acceleration':w['acceleration_proxy']}
            for name,pp in pred.items():
                row[name]=int(models[name].classes_[pp[j].argmax()])
                for k in range(4):row[name+'_p'+str(k)]=float(pp[j,k])
            rows.append(row)
        timings.append({'segment':w['segment'],'extraction_seconds':extraction,'decoded_selected_frames_for_features':n,'evaluated_windows':len(ws),'prediction_seconds':cost,'execution_error':None})
        print(key,'features and both fixed predictions complete',round(extraction,2),'seconds',flush=True)
    df=pd.DataFrame(rows).sort_values('window');assert len(df)==18 and df.window.nunique()==18
    b=opposite(df.truth,df.production);a=opposite(df.truth,df.expanded)
    df['new_opposite']=a&~b;df['opposite_to_correct']=b&(df.expanded==df.truth)
    df['opposite_to_constant_wrong']=b&(df.expanded==2)&(df.truth!=2)
    df['corrected']=(df.production!=df.truth)&(df.expanded==df.truth)
    df['new_wrong']=(df.production==df.truth)&(df.expanded!=df.truth)
    df['both_wrong']=(df.production!=df.truth)&(df.expanded!=df.truth)
    df.to_csv(O/'predictions.csv',index=False)
    summary={};changes={}
    for name,g in [('all',df),*list(df.groupby('route',sort=False))]:
        summary[name]={m:metrics(g.truth.to_numpy(),g[m].to_numpy()) for m in models}
        changes[name]={k:int(g[k].sum()) for k in ['new_opposite','opposite_to_correct','opposite_to_constant_wrong','corrected','new_wrong','both_wrong']}
    r,p=summary['all']['expanded'],summary['all']['production'];routes=[v for k,v in summary.items() if k!='all']
    expanded_pass=(r['macro_f1_AD_C']>p['macro_f1_AD_C'] and r['correct']>p['correct'] and changes['all']['new_opposite']==0
                   and all(v['expanded']['macro_f1_AD_C']>=v['production']['macro_f1_AD_C'] for v in routes))
    production_pass=(p['macro_f1_AD_C']>=r['macro_f1_AD_C'] and p['correct']>=r['correct']
                     and all(v['production']['macro_f1_AD_C']>=v['expanded']['macro_f1_AD_C'] for v in routes))
    decision='확대학습 모델을 후속 검증 대상으로 유지' if expanded_pass else '운영 모델을 다음 개선의 기준으로 유지' if production_pass else '현재 표본으로 우열 판단 불가'
    write(O/'metrics.json',summary);write(O/'changes.json',changes)
    write(O/'decision.json',{'choice':decision,'expanded_observed_rule_pass':bool(expanded_pass),'production_observed_rule_pass':bool(production_pass),'production_replaced':False,'official_S3_computed':False,'population_superiority_claim':False})
    write(O/'runtime.json',{'load_seconds':loadcost,'clips':timings,'total_seconds':time.perf_counter()-started,'fits':0,'prediction_windows_per_model':18,'execution_failures':0})
    assert all(sha(R/p)==h for p,h in f['inputs'].items())
    write(O/'checks.json',{'same_features_and_truth_both_models':True,'unchanged_protected_inputs':True,'new_training':False,'same18centers_only':True,'confusion_formula_recomputed':True,'reserved_models_run_once':True})
    exposure={'utc':datetime.now(timezone.utc).isoformat(),'status':'development_exposed_after_fixed_baseline_comparison',
        'supersedes_historical_reserved_status':True,'comparison':str(O.relative_to(R)),'models':{k:v['sha256'] for k,v in f['models'].items()},
        'window_indices':list(range(18)),'model_predictions':36,'training':0,'labels_changed':False,'repeat_tuning_allowed':False,
        'reason':'User-authorized one-time comparison; subsequent work must not call these18 windows fresh independent evaluation.'}
    write(O/'exposure_after.json',exposure)
    exposurefile=A/'EXPOSURE_20260921_BASELINE_COMPARISON.json';assert not exposurefile.exists();write(exposurefile,exposure)
    print(json.dumps({'metrics':summary,'changes':changes,'decision':decision},ensure_ascii=False,indent=2))

if __name__=='__main__':
    with threadpool_limits(limits=2):
        cv2.setNumThreads(2)
        {'freeze':freeze,'run':run}[sys.argv[1]]()
