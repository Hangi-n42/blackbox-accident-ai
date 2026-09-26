"""Fixed-weight directional derivatives only; no fit, ablation or CCD access."""
import importlib.util
import json
import platform
from datetime import datetime, timezone
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
BASE=ROOT/'artifacts/stage2_entry_l2_01_20260921'
spec=importlib.util.spec_from_file_location('lambda01',BASE/'run.py')
base=importlib.util.module_from_spec(spec);spec.loader.exec_module(base)
sha=base.helper.sha

def write(name,obj):
    (HERE/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n')

def main():
    assert platform.system()=='Darwin' and platform.machine()=='arm64'
    assert not (HERE/'protocol.json').exists(),'Preserve completed diagnostic'
    protocol=json.loads((BASE/'protocol.json').read_text())
    inherited=json.loads((BASE/'freeze.json').read_text())['sha256']
    for p,h in inherited.items():assert sha(ROOT/p)==h,p
    hashes=json.loads((BASE/'STATUS.json').read_text())['artifact_sha256']
    for name in ['model.npz','protocol.json','result.json','run.py']:
        assert sha(BASE/name)==hashes[str((BASE/name).relative_to(ROOT))]
    state=dict(np.load(BASE/'model.npz',allow_pickle=False));w=state['weights']
    x=np.load(base.helper.PRIOR/'design.npy',allow_pickle=False)
    cases=protocol['cases'];pairs=[np.array(c['top_pairs'],dtype=int) for c in cases]
    assert [len(p) for p in pairs]==[11,11,11,20]
    assert cases[0]['ID']=='00000' and cases[0]['frames'][6]==561 and cases[0]['frames'][0]==0
    delta=x[0,6]-x[0,0];length=float(np.linalg.norm(delta));assert length>0
    d=delta/length
    np.testing.assert_allclose(d@d,1,rtol=0,atol=1e-14)
    write('protocol.json',{'created_utc':datetime.now(timezone.utc).isoformat(),'baseline':str(BASE.relative_to(ROOT)),
                          'scope':'Fixed lambda.1 weights, standardized14features,train4,53pairs; no fit, lambda change, model forward, CCD or feature ablation.',
                          'direction':'d=(x_00000_f561-x_00000_f0)/L2norm. Increasing weights along d improves this pair margin.',
                          'pair_gradient':'-D*sigmoid(-D@w)/(4*n_incident_pairs)',
                          'regularization_gradient':'0.1*w',
                          'sign':'negative projected loss derivative supports +d locally; positive opposes; zero neutral. Not causal/global proof.',
                          'checks':'pair sums to incident sums to total; total matches original objective gradient; explicit f561 vs f0 pair.',
                          'readout':'Every pair, incident signed/support/opposition totals, L2 and total gradient projection; per-pair score sensitivity without changing weights.'})
    paths=[ROOT/p for p in inherited]+[BASE/'model.npz',BASE/'result.json',BASE/'freeze.json',BASE/'STATUS.json',Path(__file__),HERE/'protocol.json']
    frozen={str(p.relative_to(ROOT)):sha(p) for p in paths}
    write('freeze.json',{'created_utc':datetime.now(timezone.utc).isoformat(),'sha256':frozen})
    groups={'appearance_pca':slice(0,4),'previous_difference':slice(4,8),'following_difference':slice(8,12),
            'relative_time':slice(12,13),'first_indicator':slice(13,14)}
    incidents=[];rows=[];diffs=[];data_grad=np.zeros(14)
    for z,c,p in zip(x,cases,pairs):
        D=z[p[:,0]]-z[p[:,1]];diffs.append(D);margins=D@w
        gradients=-D*np.exp(-np.logaddexp(0,margins))[:,None]/(4*len(p))
        incident_grad=gradients.sum(axis=0);data_grad+=incident_grad
        projections=gradients@d
        incidents.append({'ID':c['ID'],'pairs':len(p),'gradient':incident_grad.tolist(),
                          'projection':float(incident_grad@d),'support_sum':float(projections[projections<0].sum()),
                          'oppose_sum':float(projections[projections>0].sum()),
                          'support_count':int(np.sum(projections<0)),'oppose_count':int(np.sum(projections>0))})
        for (a,b),margin,g,proj,sensitivity in zip(p,margins,gradients,projections,D@d):
            rows.append({'ID':c['ID'],'winner_index':int(a),'loser_index':int(b),'winner_frame':c['frames'][a],
                         'loser_frame':c['frames'][b],'current_margin':float(margin),'margin_directional_derivative':float(sensitivity),
                         'gradient':g.tolist(),'loss_directional_derivative':float(proj),
                         'projection_by_feature_group':{name:float(g[s]@d[s]) for name,s in groups.items()}})
    reg_grad=.1*w;total_grad=data_grad+reg_grad
    loss,reference_grad=base.objective(w,diffs,.1)
    np.testing.assert_allclose(total_grad,reference_grad,rtol=0,atol=1e-15)
    np.testing.assert_allclose(sum(r['loss_directional_derivative'] for r in rows)+reg_grad@d,total_grad@d,rtol=0,atol=1e-15)
    direct=[r for r in rows if r['ID']=='00000' and r['winner_index']==6 and r['loser_index']==0]
    assert len(direct)==1 and direct[0]['loss_directional_derivative']<0
    assert all(sha(ROOT/p)==h for p,h in frozen.items())
    result={'completed_utc':datetime.now(timezone.utc).isoformat(),'platform':platform.platform(),'lambda':.1,
            'direction':d.tolist(),'direction_by_feature_group':{name:float(np.linalg.norm(d[s])) for name,s in groups.items()},
            'target_current_margin':float(delta@w),'target_margin_directional_derivative':float(delta@d),
            'incidents':incidents,'pairs':rows,'direct_target_pair':direct[0],
            'data_projection':float(data_grad@d),'regularization_projection':float(reg_grad@d),
            'total_projection':float(total_grad@d),'total_gradient_l2':float(np.linalg.norm(total_grad)),
            'objective':loss,'frozen_files_preserved':len(frozen),'training_runs':0,'encoder_forwards':0,'ccd_accesses':0,
            'weight_updates':0,'submission_changes':0,
            'limits':'One local direction in frozen standardized feature coordinates. A helpful pairwise margin direction need not improve full argmax, other incidents, ±.3 accuracy or generalization.'}
    write('result.json',result)
    print(json.dumps({k:result[k] for k in ['target_current_margin','target_margin_directional_derivative','incidents','data_projection','regularization_projection','total_projection','total_gradient_l2']},indent=2))
    print('direct pair',direct[0]['loss_directional_derivative'])
    for r in sorted(rows,key=lambda r:r['loss_directional_derivative'],reverse=True)[:6]:print('opposing',r['ID'],r['winner_frame'],r['loser_frame'],r['loss_directional_derivative'])

if __name__=='__main__':main()
