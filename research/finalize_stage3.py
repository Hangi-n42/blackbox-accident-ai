"""Historical transfer-head experiment, superseded by select_stage3.py.

This script fits an earlier mixed transfer model and overwrites model/stage3;
it does not reproduce the checkpoint in submissions 87570 or 87584. Their
recorded selection uses both heads from motion_model_external.joblib through
research/select_stage3.py. Retained only to reproduce the historical experiment.
"""
import sys,json,time,hashlib
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from train_stage3 import ROOT,public_data,external_data,candidate
from solution.stage3 import ACCEL,predict_stage3
import cv2,joblib,numpy as np,pandas as pd
from sklearn.metrics import f1_score,classification_report
from threadpoolctl import threadpool_limits

cv2.setNumThreads(1)
with threadpool_limits(limits=1):
    x,ya,ys,g,df=public_data();ext=external_data();oof=np.empty_like(ya)
    for held in np.unique(g):
        train=g!=held;m=candidate('linear')
        m.fit(np.concatenate([x[train],ext[0]]),np.concatenate([ya[train],ext[1]]))
        oof[~train]=m.predict(x[~train]);print('OOF complete',held,flush=True)
    accel_score=f1_score(ya,oof,labels=np.arange(4),average='macro',zero_division=0)
    m=candidate('linear');m.fit(np.concatenate([x,ext[0]]),np.concatenate([ya,ext[1]]))
    transfer=joblib.load(ROOT/'solution/model/stage3/motion_model_transfer.joblib')
    model={'accel':m,'steer':transfer['steer'],'steer_motion_regressor':transfer['motion_regressor'],'feature_version':'dis256_roi144_temporal6_v1','external_dataset':'commaai/comma2k19; MIT; 23 selected routes; public aligned duplicate excluded'}
    transfer_report=json.loads((ROOT/'research/stage3_validation_transfer.json').read_text())
    steer_score=next(r['macro_f1'] for r in transfer_report['results']['steer'] if r['name']==transfer_report['steer_selected'])
    report={'accel':{'experiment':'external proxy CAN labels + public labels; logistic','macro_f1':float(accel_score),'class_report':classification_report(ya,oof,labels=np.arange(4),target_names=list(ACCEL),output_dict=True,zero_division=0)},'steer':{'experiment':'continuous CAN regression then public mirror classifier','macro_f1':steer_score,'classifier':transfer_report['steer_selected']},'stage3_oof':float(.7*accel_score+.3*steer_score),'external_train_rows':len(ext[0]),'external_routes':len(set(ext[3])),'external_proxy_label_rule':{'speed_stop_m_s':.3,'accel_m_s2':.25,'steering_degrees_unused_for_final_steer':2.},'caveat':'Model/head choice uses same 5-video 50-label public OOF set, so there is selection bias and large small-sample uncertainty. Not a private score. Aligned video duplicate removed, full source-route overlap cannot be ruled out. External proxy label thresholds are not official DACON thresholds.'}
    out=ROOT/'model/stage3';out.mkdir(parents=True,exist_ok=True)
    joblib.dump(model,out/'motion_model.joblib',compress=3)
    report['model_sha256']=hashlib.file_digest((out/'motion_model.joblib').open('rb'),'sha256').hexdigest()
    (ROOT/'solution/model/stage3/motion_model_selected.joblib').write_bytes((out/'motion_model.joblib').read_bytes())
    (ROOT/'research/stage3_selection.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    p=df.copy();p['accel_oof']=ACCEL[oof];p.to_csv(ROOT/'research/stage3_oof_selected_accel.csv',index=False)
    print(json.dumps({'oof':report['stage3_oof'],'sha256':report['model_sha256']}),flush=True)

    eval_dir=ROOT/'artifacts/public_eval_10hz/stage3'
    if eval_dir.exists():
        start=time.perf_counter();result=predict_stage3(eval_dir,out)
        result.to_csv(ROOT/'research/stage3_full10hz_predictions.csv',index=False)
        counts=result.groupby('ID').size().to_dict()
        for _,rows in result.groupby('ID'):assert rows.sample_index.tolist()==list(range(len(rows)))
        check={'counts':counts,'elapsed_seconds':time.perf_counter()-start,'all_indices_contiguous':True,'all_labels_valid':bool(result.accel_label.isin(['ACCELERATING','DECELERATING','CONSTANT','STOPPED']).all() and result.steer_label.isin(['LEFT','STRAIGHT','RIGHT']).all()),'model_sha256':report['model_sha256'],'scope':'Actual predict_stage3 on public 10Hz converted videos. Training-fit predictions are not a validation score.'}
        assert check['all_labels_valid']
        (ROOT/'research/stage3_full10hz_contract.json').write_text(json.dumps(check,indent=2),encoding='utf-8')
        print(json.dumps(check),flush=True)
