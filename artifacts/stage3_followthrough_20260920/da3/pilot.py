"""Bounded pose-only validity diagnostic; not a classifier or score experiment."""
from smoke import O,R,load,InputProcessor
import json,time,hashlib,traceback
import numpy as np,pandas as pd,torch
P=R/'artifacts/stage3_point_motion_20260919'
def write(p,x):p.write_text(json.dumps(x,indent=2,ensure_ascii=False))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def feature(c,t):
    tt=t-t[len(t)//2];design=np.stack([np.ones(len(t)),tt,tt**2],axis=1)
    coef=np.linalg.lstsq(design,c,rcond=None)[0];v=coef[1];a=2*coef[2];vv=float(v@v)
    span=float(np.linalg.norm(c[-1]-c[0]));path=float(np.linalg.norm(np.diff(c,axis=0),axis=1).sum())
    return {'q':float(v@a/vv) if vv>1e-12 else None,'speed_relative':float(np.sqrt(vv)),'accel_relative':float(np.linalg.norm(a)),'span':span,'path':path,'quad_residual_over_span':float(np.sqrt(np.mean(np.sum((c-design@coef)**2,axis=1)))/max(span,1e-12))}
def align_error(a,b):
    # Align row-vector b to a by one positive global scale and proper rotation.
    ac=a-a.mean(0);bc=b-b.mean(0);u,s,vt=np.linalg.svd(bc.T@ac);d=np.ones(3);d[-1]=np.linalg.det(u@vt);rot=u@np.diag(d)@vt
    scale=float((s*d).sum()/max((bc*bc).sum(),1e-12));fit=scale*bc@rot
    return float(np.sqrt(np.mean(np.sum((ac-fit)**2,axis=1)))/max(np.linalg.norm(a[-1]-a[0]),1e-12))
def main():
    if (O/'pilot_freeze.json').exists():raise FileExistsError('Frozen pilot exists')
    windows=json.loads((P/'windows.json').read_text());assert len(windows)==28
    write(O/'pilot_freeze.json',{'source_sha256':sha(__import__('pathlib').Path(__file__)),'weights':json.loads((O/'provenance.json').read_text()),'windows_sha256':sha(P/'windows.json'),'frame_hashes':{w['key']:sha(P/(w['key']+'_frames.npz')) for w in windows},'window_count':28,'input':'existing21RGBframes/2s at10Hz; same504 upper_bound_resize allwindows; automatic whole image, no hand masks; MPSfloat32; official fullnet camera decoder, not ray pose; no sensors as input','variants':['chronological_first','reverse_first_then_restore_time_order','chronological_middle'],'q':'quadratic camera-center C(t) fit, at midpoint q=(v dot a)/(v dot v); world-to-camera centers=-R.T@t. Not metric acceleration.','frozen_diagnostic_thresholds':'q difference <=.02/s and globalSim3 RMS/span<=.10 across reference/order; A/D sign agreement; CONSTANT abs(q)<=.02/s. Provisional engineering tolerances, not official label thresholds. No tuning.','expansion_gate':'At least3 stable held A/D/C each; all three q variants sign-consistent where |sensor q|>.01, held MAE below zero-q baseline, and no systematic constant false acceleration. ExistingheldA2 cannot establish this gate, so no classifier fit without new reserved evidence.','runtime_cap_seconds':600,'stop_rule':'stop at walltime cap or unsupported operation/OOM, record incomplete; do not change model/resolution/threshold after seeing outputs'})
    torch.set_num_threads(2);torch.manual_seed(42);start=time.perf_counter();model=load('mps');processor=InputProcessor();rows=[];errors=[]
    folder=O/'poses';folder.mkdir();stop=False
    for w in windows:
        z=np.load(P/(w['key']+'_frames.npz'));rgb=z['rgb'];t=z['time'];assert len(rgb)==21
        x,_,_=processor(list(rgb),process_res=504,num_workers=1)
        if x.ndim==4:x=x.unsqueeze(0)
        centers={};measurements={}
        for variant,reverse,reference in [('first',False,'first'),('reverse',True,'first'),('middle',False,'middle')]:
            if time.perf_counter()-start>600:stop=True;errors.append({'key':w['key'],'variant':variant,'reason':'runtime cap600s'});break
            began=time.perf_counter()
            try:
                inp=x.flip(1) if reverse else x
                with torch.inference_mode():out=model(inp.to('mps'),ref_view_strategy=reference,use_ray_pose=False,infer_gs=False)
                ext=out['extrinsics'][0].cpu().numpy();intr=out['intrinsics'][0].cpu().numpy()
                if reverse:ext=ext[::-1].copy();intr=intr[::-1].copy()
                assert ext.shape==(21,3,4) and np.isfinite(ext).all()
                rot=ext[:,:3,:3];c=-np.einsum('nij,nj->ni',rot.transpose(0,2,1),ext[:,:3,3])
                assert np.max(abs(np.linalg.det(rot)-1))<.01
                np.savez_compressed(folder/f'{w["key"]}_{variant}.npz',extrinsics=ext,intrinsics=intr,centers=c,time=t)
                centers[variant]=c;measurements[variant]=feature(c,t)
                rows.append({'key':w['key'],'id':w['id'],'role':w['role'],'label':w['label'],'variant':variant,'sensor_q':w['sensor_ratio_a_v'],**measurements[variant],'seconds':time.perf_counter()-began})
                print(w['key'],variant,'q',measurements[variant]['q'],'seconds',round(time.perf_counter()-began,2),flush=True)
                del out;torch.mps.empty_cache()
            except Exception as e:
                errors.append({'key':w['key'],'variant':variant,'error':repr(e),'traceback':traceback.format_exc()});stop=True;break
        if len(centers)==3:
            base=measurements['first']['q'];checks=[]
            for variant in ['reverse','middle']:
                delta=abs(measurements[variant]['q']-base) if base is not None and measurements[variant]['q'] is not None else float('inf')
                err=align_error(centers['first'],centers[variant]);checks.append(delta<=.02 and err<=.1)
                for r in rows:
                    if r['key']==w['key'] and r['variant']==variant:r.update(q_difference=delta,trajectory_sim3_error=err)
            for r in rows:
                if r['key']==w['key']:r['stable']=all(checks)
        pd.DataFrame(rows).to_csv(O/'pilot_results.csv',index=False);write(O/'pilot_errors.json',errors)
        if stop:break
    df=pd.DataFrame(rows);summaries=[]
    for (role,variant,label),g in df.groupby(['role','variant','label']):
        valid=g.q.notna();q=g.loc[valid,'q'];truth=g.loc[valid,'sensor_q']
        summaries.append({'role':role,'variant':variant,'label':int(label),'n':len(g),'finite':int(valid.sum()),'stable':int(g.get('stable',pd.Series(False,index=g.index)).fillna(False).sum()),'MAE':float(np.mean(abs(q-truth))),'zero_q_MAE':float(np.mean(abs(truth))),'sign_agreement':int((q*truth>0).sum()) if label in [0,1] else None,'constant_abs_q_le002':int((abs(q)<=.02).sum()) if label==2 else None})
    write(O/'pilot_summary.json',{'completed_windows':int(sum(df.groupby('key').size()==3)),'planned_windows':28,'seconds':time.perf_counter()-start,'errors':errors,'groups':summaries,'feature_adopted':False,'classifier_fit':False,'official_score_measured':False,'limitations':'Development-selected windows, sensor proxy, no metric-pose truth. Reference/order tests detect instability but cannot establish absolute pose accuracy or absence of temporal scale drift.'})
    print('complete',time.perf_counter()-start,'seconds',flush=True)
if __name__=='__main__':main()
