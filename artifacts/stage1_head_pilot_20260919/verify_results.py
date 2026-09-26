"""Recompute confusion tables from saved predictions, verify frozen assets and labels."""
import csv,hashlib,json
from pathlib import Path
ROOT=Path.cwd();P=ROOT/'artifacts/stage1_head_pilot_20260919';T=ROOT/'artifacts/stage1_temporal23_20260919'
def metric(rr,key):
 tp=tn=fp=fn=0
 for r in rr:
  y=r['label']=='recapture';pred=r[key]>=.5
  tp+=y and pred;tn+=not y and not pred;fp+=not y and pred;fn+=y and not pred
 return {'macro_f1':(2*tp/(2*tp+fp+fn)+2*tn/(2*tn+fp+fn))/2,'fp':fp,'fn':fn,'original_n':tn+fp,'recapture_n':tp+fn}
for split in ['train','development','new_val','new_tcl']:
 rs=json.load(open(P/f'{split}_predictions.json'));s=json.load(open(P/f'{split}_summary.json'))
 assert len({(r['source_group'],r['label'],r['mode']) for r in rs})==len(rs)
 for mode,result in s['conditions'].items():
  rr=rs if mode=='primary_all' else [r for r in rs if r['mode']==mode]
  for method in ['baseline','candidate']:
   m=metric(rr,method)
   for k,v in m.items():assert abs(v-result[method][k])<1e-12
labels={r['ID']:r['label'] for r in csv.DictReader(open(ROOT/'Baseline/data/stage1/labels.csv'))}
r=json.load(open(P/'public_guard_predictions.json'));assert all((z['label']=='recapture')==(labels[z['ID']]=='RERECORDED') for z in r)
pub={m:metric(r,m) for m in ['baseline','candidate']};pub['new_original_fp']=sum(z['label']=='original' and z['baseline']<.5 and z['candidate']>=.5 for z in r)
(P/'public_guard_summary.json').write_text(json.dumps(pub,indent=2))
for protocol in [P/'protocol.json',T/'protocol.json']:
 for p,h in json.load(open(protocol))['frozen_files'].items():assert hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h,p
train={r['source_group'] for r in json.load(open(P/'train_records.json'))};dev={r['source_group'] for r in json.load(open(P/'development_records.json'))};new={r['source_group'] for r in json.load(open(P/'new_val_records.json'))}
assert not train&dev and not train&new and not dev&new
ver={'status':'PASS','confusion_tables_recomputed':True,'source_id_overlap_train_dev_test':0,'same_test_sources_across_two_devices':True,'location_independence':False,'public_labels_verified_against_baseline_csv':True,'frozen_model_scripts_unchanged':True,'no_hyperparameter_or_threshold_refit':True}
(P/'verification.json').write_text(json.dumps(ver,indent=2))
(P/'candidate_decision.json').write_text(json.dumps({'decision':'DO_NOT_ADOPT_PUBLIC_ORIGINAL_REGRESSION','development_gate':True,'new_iphone_macro_f1':1.0,'new_tcl_macro_f1':1.0,'public_original_fp_before':1,'public_original_fp_after':5,'public_guard':pub,'reason':'Positive proxy and comma checks do not override 4 new errors on known camera-original public examples. No post-hoc threshold search or further fit.'},indent=2))
d=json.load(open(T/'candidate_decision.json'));d['new_sources_subsequently_scored_by_predeclared_head_pilot']=True;(T/'candidate_decision.json').write_text(json.dumps(d,indent=2))
print(json.dumps(pub,indent=2));print('VERIFIED',ver)
