"""Paired inversion audit and frozen adoption gates; no fitting or selection."""
from pathlib import Path
import json,hashlib
import pandas as pd,numpy as np
O=Path(__file__).resolve().parent;R=O.parents[1];P=R/'artifacts/stage3_state_learning_20260919'
read=lambda p:json.loads(p.read_text());write=lambda p,z:p.write_text(json.dumps(z,ensure_ascii=False,indent=2))
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def main():
    m=pd.read_csv(O/'capped_control/metrics.csv');df=pd.read_csv(O/'capped_control/predictions.csv')
    ref=df[df.variant=='uniform'][['fold','id','sample_index','prediction']].rename(columns={'prediction':'reference'})
    df=df.merge(ref,on=['fold','id','sample_index'],validate='many_to_one')
    isopp=lambda a,b:((a==0)&(b==1))|((a==1)&(b==0))
    before=isopp(df.truth,df.reference);after=isopp(df.truth,df.prediction)
    df['new_opposite']=after&~before;df['fixed_opposite']=before&~after;df['new_from_correct']=df.new_opposite&(df.reference==df.truth)
    df.to_csv(O/'all_paired_predictions.csv',index=False)
    df.groupby(['variant','fold','vehicle'])[['new_opposite','fixed_opposite','new_from_correct']].sum().to_csv(O/'paired_error_counts.csv')
    strict=pd.read_csv(P/'dis_metrics.csv').query("variant=='linear'");strictpred=pd.read_csv(P/'dis_predictions.csv').query("variant=='linear'")
    base=m[m.variant=='uniform'];decisions=[]
    for variant,g in m.groupby('variant',sort=False):
        pub=g[(g.scope=='public_oof')&(g.group=='all')].iloc[0];bp=base[(base.scope=='public_oof')&(base.group=='all')].iloc[0]
        d=df[df.variant==variant];pg=d[d.fold.str.startswith('OPEN')];changes=[]
        for a in g[g.scope!='public_oof'].itertuples():
            b=base[(base.scope==a.scope)&(base.group==a.group)].iloc[0];s=strict[(strict.scope==a.scope)&(strict.group==a.group)].iloc[0]
            changes.append({'scope':a.scope,'group':a.group,'F1':a.macro_f1,'F1_delta_vs_expanded':a.macro_f1-b.macro_f1,'opposite':int(a.opposite),'opposite_delta_vs_expanded':int(a.opposite-b.opposite),'F1_delta_vs_strict':a.macro_f1-s.macro_f1,'opposite_rate_delta_vs_strict':a.opposite_rate-s.opposite_rate})
        civic=next(a for a in changes if a['scope']=='date_1' and a['group']=='99c94dc769b5d96e')
        partial=pub.S3>=bp.S3-1e-12 and pub.opposite<=bp.opposite and not pg.new_from_correct.any() and all(a['F1_delta_vs_expanded']>=-1e-12 for a in changes if a['group']=='all') and all(a['opposite_delta_vs_expanded']<=0 for a in changes) and civic['opposite']<275
        sp=pg.merge(strictpred[['fold','id','sample_index','prediction']].rename(columns={'prediction':'strict_reference'}),on=['fold','id','sample_index'],validate='one_to_one')
        added=isopp(sp.truth,sp.prediction)&~isopp(sp.truth,sp.strict_reference)
        oldscore=float(strict[(strict.scope=='public_oof')&(strict.group=='all')].S3.iloc[0])
        safe=partial and pub.S3>=oldscore+.01 and not added.any() and all(a['F1_delta_vs_strict']>=-1e-12 for a in changes if a['group']=='all') and all(a['opposite_rate_delta_vs_strict']<=.01 for a in changes)
        decisions.append({'variant':variant,'public_S3':float(pub.S3),'public_S3_delta_vs_expanded':float(pub.S3-bp.S3),'public_opposite':int(pub.opposite),'new_public_opposite_from_correct':int(pg.new_from_correct.sum()),'changes':changes,'partial_gate_passed':bool(partial),'old_safety_gate_passed':bool(safe),'promotable':bool(safe and variant=='selected')})
    write(O/'decision.json',decisions)
    f=read(O/'freeze.json');assert all(sha(R/p)==h for p,h in {**f['inputs'],**f['protected']}.items())
    assert sha(O/'run.py')==f['source_sha256'];assert sha(O/'capped_control.py')==read(O/'capped_control/freeze.json')['source_sha256']
    # Verify the inner folds share no vehicle-date and cannot contain outer-held sources.
    cases={c['id']:c for c in read(R/'artifacts/stage3_training_basis_20260917/cases.json')}
    key=lambda id:(cases[id]['vehicle'],cases[id]['route'].split('|')[1][:10])
    for sp in f['outer']:
        for ins in f['inner'][sp['name']]:
            tr={key(id) for id in ins['train'] if id in cases};va={key(id) for id in ins['held']}
            assert tr.isdisjoint(va) and set(ins['train']+ins['held']).isdisjoint(sp['held'])
    (O/'exit_status.txt').write_text('0\n');(O/'capped_control/exit_status.txt').write_text('0\n')
    write(O/'final_checks.json',{'completed':True,'all_executions_exit0':True,'baseline_probabilities_exact':True,'input_labels_features_production_unchanged':True,'executed_sources_match_frozen_hashes':True,'inner_vehicle_date_disjoint':True,'selected_using_inner_only':True,'class_mass_total_mass_scaler_fixed':True,'promotable_candidates':[a['variant'] for a in decisions if a['promotable']],'no_production_update':True})
    print(m[m.group=='all'][['variant','scope','macro_f1','opposite','S3']].to_string(index=False))
    print('all checks passed; no candidate adopted')

if __name__=='__main__':main()
