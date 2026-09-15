"""Select each output head using recorded public-video OOF experiments."""
import json,hashlib
from pathlib import Path
import joblib
root=Path(__file__).resolve().parents[1]
experiments={
 'public':('stage3_validation_public.json','motion_model.joblib'),
 'public_mirror':('stage3_validation_public_mirror.json','motion_model_public_mirror.joblib'),
 'external':('stage3_validation_external.json','motion_model_external.joblib'),
 'transfer':('stage3_validation_transfer.json','motion_model_transfer.joblib'),
}
selected={};report={}
for task in ('accel','steer'):
    choices=[]
    for name,(report_name,model_name) in experiments.items():
        p=root/'research'/report_name
        if not p.exists():continue
        r=json.loads(p.read_text());rec=next(v for v in r['results'][task] if v['name']==r[task+'_selected'])
        choices.append((rec['macro_f1'],name,model_name))
    score,name,filename=max(choices)
    m=joblib.load(root/'solution/model/stage3'/filename);selected[task]=m[task]
    if 'motion_regressor' in m:selected[task+'_motion_regressor']=m['motion_regressor']
    report[task]={'experiment':name,'macro_f1':score,'source_checkpoint':filename}
selected['feature_version']='dis256_roi144_temporal6_v1';selected['selection']=report
report['stage3_oof']=.7*report['accel']['macro_f1']+.3*report['steer']['macro_f1']
report['caveat']='Selection uses same 5-video public OOF labels. Selection bias and small sample uncertainty. Not a private score.'
joblib.dump(selected,root/'solution/model/stage3/motion_model_selected.joblib',compress=3)
published=root/'model/stage3';published.mkdir(parents=True,exist_ok=True)
(published/'motion_model.joblib').write_bytes((root/'solution/model/stage3/motion_model_selected.joblib').read_bytes())
report['model_sha256']=hashlib.file_digest((published/'motion_model.joblib').open('rb'),'sha256').hexdigest()
report['external_source']='https://huggingface.co/datasets/commaai/comma2k19 (MIT)'
report['proxy_thresholds_not_official']={'speed_stop_m_s':.3,'accel_m_s2':.25,'steer_degrees':2.}
report['overlap_caveat']='One aligned duplicate of a public video excluded. Complete route/source overlap cannot be ruled out.'
(root/'research/stage3_selection.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print(json.dumps(report,indent=2))
