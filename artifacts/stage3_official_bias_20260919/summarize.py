"""Replay saved predictions, audit nested exclusion, and apply frozen gates."""
from pathlib import Path
import json, hashlib
import numpy as np
import pandas as pd
from scipy.special import softmax
from sklearn.metrics import f1_score, confusion_matrix

O=Path(__file__).resolve().parent; R=O.parents[1]
read=lambda p:json.loads(p.read_text())
write=lambda p,x:p.write_text(json.dumps(x,ensure_ascii=False,indent=2))
df=pd.read_csv(O/'predictions.csv'); freeze=read(O/'freeze.json')
saved=read(O/'metrics.json'); audits=[]
for sp in freeze['splits']:
    folder=O/sp['name'];c=read(folder/'calibrator.json');b=np.array(c['bias'])
    assert abs(b.sum())<1e-12 and c['gradient_max']<=1e-7
    for ins in c['inner']:
        mf=read(O/'base_models'/ins['model']/'training_manifest.json')
        assert ins['train']==mf['ids']
        assert set(ins['train']).isdisjoint(sp['held']+[ins['held']])
        assert set(a[0] for a in mf['selection'])==set(ins['train'])
    cal=pd.read_csv(folder/'calibration_oof.csv')
    assert set(cal.id)==set(a['held'] for a in c['inner'])
    assert not cal.duplicated(['id','sample_index']).any()
    for id in sp['held']:
        z=np.load(folder/f'{id}.npz');g=df[(df.fold==sp['name'])&(df.id==id)]
        assert np.array_equal(z['sample_index'],g.sample_index)
        assert np.allclose(softmax(z['logits']+b,axis=1),z['calibrated'],atol=1e-12)
        assert np.array_equal(z['raw'].argmax(1),g.raw)
        assert np.array_equal(z['calibrated'].argmax(1),g.calibrated)
for m in saved:
    g=df[df.fold.str.startswith('OPEN')] if m['scope']=='public_oof' else df[df.fold==m['scope']]
    if m['group']!='all':g=g[g.vehicle==m['group']]
    score=f1_score(g.truth,g[m['variant']],labels=range(4),average='macro',zero_division=0)
    cm=confusion_matrix(g.truth,g[m['variant']],labels=range(4))
    assert np.isclose(score,m['macro_f1'],atol=1e-14)
    assert cm.tolist()==m['confusion'] and len(g)==m['n']
    assert int(cm[0,1]+cm[1,0])==m['opposite']
    if not m['opposite_denominator']:assert m['opposite_rate'] is None
    if m['scope']=='public_oof':assert np.isclose(m['S3'],.7*score+.3*m['steer_F1'])
score={(a['variant'],a['scope'],a['group']):a for a in saved}
pg=df[df.fold.str.startswith('OPEN')]
gain=score['calibrated','public_oof','all']['S3']-score['raw','public_oof','all']['S3']
video_gains=[k for k in pg.fold.unique() if score['calibrated',k,'all']['macro_f1']>score['raw',k,'all']['macro_f1']]
date_f1=all(a['macro_f1']>=score['raw',a['scope'],a['group']]['macro_f1'] for a in saved if a['variant']=='calibrated' and a['scope'].startswith('date') and a['group']=='all')
date_risk=all(a['opposite_rate']-score['raw',a['scope'],a['group']]['opposite_rate']<=.01 for a in saved if a['variant']=='calibrated' and a['scope'].startswith('date') and a['opposite_denominator'])
gates={'public_S3_improved':gain>0,'no_new_public_opposite':int(pg.new_opposite.sum())==0,
       'at_least_two_public_videos_improve':len(video_gains)>=2,'each_date_F1_nondecrease':date_f1,
       'each_date_vehicle_opposite_rate_increase_at_most_001':date_risk}
write(O/'decision.json',{'gates':gates,'candidate':all(gates.values()),'public_S3_delta':gain,
                       'improving_public_videos':video_gains,'action':'reject; no further bias/lambda sweep; production unchanged'})
for fold,g in df.groupby('fold'):
    for group,s in [('all',g),*list(g.groupby('vehicle'))]:
        audits.append({'fold':fold,'group':group,'n':len(s),'changed':int((s.raw!=s.calibrated).sum()),
                       **{k:int(s[k].sum()) for k in ['new_opposite','fixed_opposite','new_wrong','recovered_correct']}})
pd.DataFrame(audits).to_csv(O/'paired_changes.csv',index=False)
for path,h in freeze['paths'].items():
    with (R/path).open('rb') as f:assert hashlib.file_digest(f,'sha256').hexdigest()==h
write(O/'final_checks.json',{'completed':True,'final_execution_exit_status':0,'initial_precision_attempt_exit_status':1,
       'prediction_probability_replay':True,'all_metrics_independently_recomputed':True,
       'nested_video_exclusion_rechecked':True,'NA_for_zero_reversal_denominator':True,
       'inputs_and_production_hashes_unchanged':True,'candidate':all(gates.values()),
       'tuning_after_results':False})
print(json.dumps(gates,indent=2));print('audit complete')
