"""Aggregate every seed and predetermined risk gate, no best-seed selection."""
import json
from pathlib import Path
import pandas as pd,numpy as np
from run import O,R,read,write,sha

def main():
    sets=[('fixed120',O),('inner_stop',O/'route_early_stop')];tables=[];decisions=[];effects=[]
    baseline=pd.read_csv(O/'dis_predictions.csv');baseline=baseline[baseline.variant=='linear'][['fold','id','sample_index','prediction']].rename(columns={'prediction':'reference'})
    bm=pd.read_csv(O/'dis_metrics.csv');bm=bm[bm.variant=='linear']
    for protocol,path in sets:
        for engine in ['dis','vjepa']:
            m=pd.read_csv(path/f'{engine}_metrics.csv');m['family']=m.variant.str.split('_').str[0]
            for (scope,group,family),g in m.groupby(['scope','group','family']):
                tables.append({'protocol':protocol,'engine':engine,'family':family,'scope':scope,'group':group,'n':int(g.n.iloc[0]),
                    'F1_mean':g.macro_f1.mean(),'F1_min':g.macro_f1.min(),'F1_max':g.macro_f1.max(),
                    'S3_mean':g.S3.mean(),'S3_min':g.S3.min(),'S3_max':g.S3.max(),
                    'opposite_mean':g.opposite.mean(),'opposite_min':g.opposite.min(),'opposite_max':g.opposite.max()})
            p=pd.read_csv(path/f'{engine}_predictions.csv').merge(baseline,on=['fold','id','sample_index'],validate='many_to_one')
            old=((p.truth==0)&(p.reference==1))|((p.truth==1)&(p.reference==0))
            new=((p.truth==0)&(p.prediction==1))|((p.truth==1)&(p.prediction==0))
            p['new_opposite_vs_dis']=new&~old;p['fixed_opposite_vs_dis']=old&~new
            p[(p.prediction!=p.reference)].to_csv(path/f'{engine}_changes_vs_dis.csv',index=False)
            for family in ['linear','cls','aux']:
                if engine=='dis' and family=='linear':continue
                q=m[m.family==family];pub=q[(q.scope=='public_oof')&(q.group=='all')]
                basepub=bm[(bm.scope=='public_oof')&(bm.group=='all')].iloc[0]
                gains=pub.S3-basepub.S3
                pp=p[p.variant.str.split('_').str[0]==family];npub=pp[pp.fold.str.startswith('OPEN')].new_opposite_vs_dis
                risk=[]
                for (scope,group),g in q[q.scope!='public_oof'].groupby(['scope','group']):
                    b=bm[(bm.scope==scope)&(bm.group==group)].iloc[0]
                    risk.append({'scope':scope,'group':group,'F1_delta':g.macro_f1.mean()-b.macro_f1,'opposite_rate_delta':g.opposite_rate.mean()-b.opposite_rate})
                good=bool(gains.mean()>=.01 and gains.min()>=0 and not npub.any() and
                          all(x['F1_delta']>=0 for x in risk if x['group']=='all') and all(x['opposite_rate_delta']<=.01 for x in risk))
                decisions.append({'protocol':protocol,'engine':engine,'family':family,'public_S3_mean_gain':float(gains.mean()),
                    'public_S3_worst_seed_gain':float(gains.min()),'new_public_opposites_across_seeds':int(npub.sum()),
                    'external_changes':risk,'gate_passed':good})
    tab=pd.DataFrame(tables);tab.to_csv(O/'comparison.csv',index=False);write(O/'decisions.json',decisions)
    for protocol in ['fixed120','inner_stop']:
        a=tab[(tab.protocol==protocol)&(tab.group=='all')]
        for scope in a.scope.unique():
            for label,l,r in [('DIS model change',('dis','cls'),('dis','linear')),('DIS auxiliary objective',('dis','aux'),('dis','cls')),('encoder linear',('vjepa','linear'),('dis','linear')),('encoder same classifier',('vjepa','cls'),('dis','cls')),('encoder same auxiliary',('vjepa','aux'),('dis','aux')),('VJEPA auxiliary objective',('vjepa','aux'),('vjepa','cls'))]:
                left=a[(a.scope==scope)&(a.engine==l[0])&(a.family==l[1])].iloc[0];right=a[(a.scope==scope)&(a.engine==r[0])&(a.family==r[1])].iloc[0]
                effects.append({'protocol':protocol,'scope':scope,'effect':label,'F1_delta':left.F1_mean-right.F1_mean,'S3_delta':left.S3_mean-right.S3_mean,'opposite_delta':left.opposite_mean-right.opposite_mean})
    pd.DataFrame(effects).to_csv(O/'factor_effects.csv',index=False)
    f=read(O/'freeze.json');assert all(sha(R/p)==h for p,h in {**f['inputs'],**f['protected']}.items())
    write(O/'final_checks.json',{'completed':True,'production_and_inputs_unchanged':True,'all_engines_protocols_scored':True,
        'passed_candidates':[d for d in decisions if d['gate_passed']],
        'steering':'fixed historical external23 LOVO predictions, no steer training change; all candidates identical 0.8503042850868937 on47nonSTOPPED',
        'production_transfer_correction':'Current production checkpoint keys accel/steer/feature_version/selection; accel logistic, steer random forest. CAN transfer was historical only.',
        'scope':'development diagnosis; public50 labels, external date groups1612/1167/561 strict proxy points, dates repeat; not private or independent score',
        'baseline_warning':'DIS linear refit here uses curated strict1083external+40public rows perpublicfold, not original uncurated external23 recipe with roughly2760external rows. Historical0.798298 and current-fit0.849724 are separate references, not this baseline.',
        'conditional_geometry':'Not run: no additional geometry ground truth or calibrated consistency evidence emerged from these classification/regression experiments. Mean pooling/short windows are representation limits; classifier failure does not prove camera pose/depth is causal.'})
    print(tab[(tab.scope=='public_oof')&(tab.group=='all')][['protocol','engine','family','F1_mean','S3_mean','S3_min','S3_max','opposite_min','opposite_max']].to_string(index=False))
    print('passed',sum(d['gate_passed'] for d in decisions))

if __name__=='__main__':main()
