"""Exact linear margin accounting on three existing error/control pairs, no model edits."""
import json,sys,time,hashlib
from pathlib import Path
import numpy as np,pandas as pd,cv2,joblib
from PIL import Image,ImageDraw
R=Path(__file__).resolve().parents[2];O=Path(__file__).resolve().parent;D=R/'artifacts/pipeline_diagnosis_20260917';S=R/'artifacts/stage3_smoothing_check_20260917'
sys.path.insert(0,str(R/'artifacts/submissions/verify_v6/model/stage2/code'))
from solution.stage3_v5_compatible import extract_motion,ACCEL

def read(p):return json.loads(p.read_text())
def write(p,d):p.write_text(json.dumps(d,ensure_ascii=False,indent=2))
def sha(p):return hashlib.file_digest(p.open('rb'),'sha256').hexdigest()
assert not (O/'results.json').exists()
modelpath=R/'artifacts/submissions/verify_v6/model/stage3/motion_model.joblib';model=joblib.load(modelpath);pipeline=model['accel'];scaler=pipeline.named_steps['standardscaler'];lr=pipeline.named_steps['logisticregression'];assert lr.coef_.shape==(4,864)
# These three exposed cases show constant errors in two vehicles. Select longest continuous same-error run, then nearest correct constant control by time.
chosen=[]
for id,base,scorefile in [('comma_04',D,R/'artifacts/stage3_smoothing_20260917/scored_rows.csv'),('extra_00',S,S/'scored_rows.csv'),('extra_02',S,S/'scored_rows.csv')]:
 df=pd.read_csv(scorefile);key='ID' if 'ID' in df else 'id';df=df[(df[key]==id)&(df.truth=='CONSTANT')].sort_values('sample_index');runs=[]
 for pred,g in df[df.baseline!='CONSTANT'].groupby('baseline'):
  arr=g.sample_index.to_numpy();cuts=np.r_[0,np.flatnonzero(np.diff(arr)>1)+1,len(arr)]
  for a,b in zip(cuts[:-1],cuts[1:]):runs.append((int(b-a),int(arr[a]),int(arr[b-1]),pred))
 n,start,end,pred=max(runs,key=lambda r:(r[0],-r[1]));idx=(start+end)//2;good=df[df.baseline=='CONSTANT'];ctrl=int(good.iloc[np.argmin(abs(good.sample_index.to_numpy()-idx))].sample_index)
 row=next(r for r in read(base/'comma_cases.json') if r['id']==id)
 chosen.append(dict(id=id,base=str(base.relative_to(R)),input_path=row['input_path'],labels_path=str((base/row['labels_npz']).relative_to(R)),vehicle=row.get('vehicle'),segment=row['segment'],error_run=[start,end],error_index=idx,control_index=ctrl,prediction=pred,selection='longest same-class constant error run midpoint; nearest correct constant time, not randomized or independent'))
write(O/'selection.json',chosen);write(O/'freeze.json',{'model_sha256':sha(modelpath),'script_sha256':sha(Path(__file__)),'production_feature_sha256':sha(R/'artifacts/submissions/verify_v6/model/stage2/code/solution/stage3_v5_compatible.py'),'scope':'diagnosis only; no feature ablation, model change or training'})
cv2.setNumThreads(2);results=[];times=['raw','mean_5','mean_15','mean_31','difference_lag5','difference_lag15'];channels=['horizontal','vertical','magnitude','radial'];classes=lr.classes_.astype(int)
for r in chosen:
 start=time.perf_counter();x=extract_motion(R/r['input_path'],source_fps=10.);z=scaler.transform(x);pred=pipeline.predict(x);p=pipeline.predict_proba(x);d=np.load(R/r['labels_path']);e=r['error_index'];c=r['control_index'];target=int(np.flatnonzero(ACCEL==r['prediction'])[0]);ti=int(np.flatnonzero(classes==target)[0]);ci=int(np.flatnonzero(classes==2)[0]);assert str(ACCEL[pred[e]])==r['prediction'] and pred[c]==2
 priorroot=R/'artifacts/stage3_smoothing_20260917' if r['id']=='comma_04' else S;oldp=np.load(priorroot/(r['id']+'_probabilities.npz'))['probabilities'];assert np.allclose(p,oldp,rtol=1e-8,atol=1e-10)
 dw=lr.coef_[ti]-lr.coef_[ci];bias=float(lr.intercept_[ti]-lr.intercept_[ci]);contrib=(z*dw).reshape(-1,6,12,4,3);margin=contrib.sum(axis=(1,2,3,4))+bias;direct=pipeline.decision_function(x)[:,ti]-pipeline.decision_function(x)[:,ci];assert np.allclose(margin,direct,atol=1e-10)
 delta=contrib[e]-contrib[c];roi_delta=delta.sum(axis=(0,2,3));top=np.argsort(roi_delta)[::-1][:3]
 detail=dict(r,probability_error=dict(zip(map(str,ACCEL[classes]),map(float,p[e]))),probability_control=dict(zip(map(str,ACCEL[classes]),map(float,p[c]))),wrong_vs_constant_margin_error=float(margin[e]),wrong_vs_constant_margin_control=float(margin[c]),margin_difference=float(margin[e]-margin[c]),margin_bias=bias,delta_by_time=dict(zip(times,map(float,delta.sum(axis=(1,2,3))))),delta_by_channel=dict(zip(channels,map(float,delta.sum(axis=(0,1,3))))),delta_by_roi={str(i):float(v) for i,v in enumerate(roi_delta)},top_positive_roi=top.tolist(),error_contribution_by_time=dict(zip(times,map(float,contrib[e].sum(axis=(1,2,3))))),control_distance_s=abs(e-c)/10,sensor={name:{'speed_m_s':float(d['speed_smoothed'][i]),'acceleration_m_s2':float(d['acceleration_proxy'][i]),'steering_deg':float(d['steering_smoothed'][i]),'index':int(i),'native_frame':int(d['frame_index'][i]),'boot_s':float(d['time'][i])} for name,i in [('error',e),('control',c)]},baseline_reproduced=True,margin_accounting_verified=True,seconds=time.perf_counter()-start)
 np.savez_compressed(O/(r['id']+'_features.npz'),features=x,standardized=z,margin=margin,roi_delta=roi_delta)
 # Full-frame views: top contributing ROI rectangles, same geometry as current features.
 cap=cv2.VideoCapture(str(R/r['input_path']));sheet=Image.new('RGB',(1200,760),'white');draw=ImageDraw.Draw(sheet)
 for col,(name,i) in enumerate([('error',e),('control',c)]):
  cap.set(cv2.CAP_PROP_POS_FRAMES,i);ok,f=cap.read();assert ok;im=Image.fromarray(cv2.cvtColor(f,cv2.COLOR_BGR2RGB)).resize((600,330));dr=ImageDraw.Draw(im)
  for k in top:
   row,colroi=divmod(int(k),4);ya,yb=[(.15,.50),(.50,.72),(.72,.93)][row];xa,xb=[(.03,.27),(.27,.5),(.5,.73),(.73,.97)][colroi];dr.rectangle((xa*600,ya*330,xb*600,yb*330),outline='red',width=2);dr.text((xa*600+3,ya*330+3),f'ROI {k}',fill='yellow')
  sheet.paste(im,(col*600,40));draw.text((col*600+5,5),f'{r["id"]} {name} {i/10:.1f}s / '+str(detail['sensor'][name]),fill='black')
 # Error context +/-1sec with sensor values, no overlays.
 for col,i in enumerate([max(0,e-10),e,min(len(x)-1,e+10)]):
  cap.set(cv2.CAP_PROP_POS_FRAMES,i);ok,f=cap.read();assert ok;sheet.paste(Image.fromarray(cv2.cvtColor(f,cv2.COLOR_BGR2RGB)).resize((400,220)),(col*400,430));draw.text((col*400+4,395),f'{i/10:.1f}s speed {d["speed_smoothed"][i]:.2f} accel {d["acceleration_proxy"][i]:.3f}',fill='black')
 cap.release();sheet.save(O/(r['id']+'_review.jpg'));results.append(detail);print(r['id'],detail['wrong_vs_constant_margin_error'],detail['top_positive_roi'],flush=True)
write(O/'results.json',results);assert sha(modelpath)==read(O/'freeze.json')['model_sha256'];print('PASS: exact probability replay, linear logit accounting, model unchanged')
