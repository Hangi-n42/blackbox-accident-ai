"""Fixed-model diagnostic of coherent synthetic image-translation flow; not a deployable correction."""
from pathlib import Path
import json,sys,hashlib
import numpy as np,pandas as pd,cv2,av,joblib
from scipy.ndimage import uniform_filter1d
O=Path(__file__).resolve().parent;R=O.parents[1];B=R/'artifacts/stage3_training_basis_20260917';F=R/'artifacts/stage3_mixed_datecheck_20260917/fold_1'
sys.path.insert(0,str(R/'artifacts/submissions/verify_v6/model/stage2/code'));from solution.stage3_v5_compatible import FrameFeatureComputer
write=lambda n,x:(O/n).write_text(json.dumps(x,ensure_ascii=False,indent=2));sha=lambda p:hashlib.file_digest(p.open('rb'),'sha256').hexdigest()
cv2.setNumThreads(2)
def combine(raw,n):
 parts=[raw]+[uniform_filter1d(raw,size=w,axis=0,mode='nearest') for w in [5,15,31]];ix=np.arange(len(raw));sm=parts[2]
 for lag in [5,15]:parts.append((sm[np.minimum(ix+lag,len(ix)-1)]-sm[np.maximum(ix-lag,0)])/(2*lag/10))
 a=np.concatenate(parts,1);return np.array([np.interp(np.arange(n),ix+.5,a[:,j]) for j in range(864)],dtype=np.float32).T

def main():
 assert not (O/'results.json').exists()
 cases=json.load(open(B/'cases.json'));model=joblib.load(F/'mixed_budget_rav4.joblib')['accel'];sel=pd.read_csv(R/'artifacts/stage3_reversal_diagnosis_20260917/selected_74.csv');prev=pd.read_csv(R/'artifacts/stage3_reversal_diagnosis_20260917/full_context.csv')
 variants={'base':None,'x_phase0':(0,0),'x_phase90':(0,np.pi/2),'y_phase0':(1,0),'y_phase90':(1,np.pi/2)}
 write('freeze.json',{'model_sha256':sha(F/'mixed_budget_rav4.joblib'),'source_script_sha256':sha(Path(__file__)),'probe':'uniform image translation offset .25px*sin(2*pi*2Hz*t+phase) at width256; flow receives discrete offset difference per second','amplitude_pixels':.25,'frequency_hz':2,'variants':list(variants),'scope':'2 existing diagnostic clips, all previously selected74 reversals and101 both-correct acceleration controls','no_real_video_augmentation':'ideal field perturbation; does not reproduce resampling or DIS reestimation; not proof of actual camera vibration','no_tuning_or_training':True})
 results={};allrows=[]
 for id in ['expanded_14','extra_01']:
  c=next(c for c in cases if c['id']==id);d=np.load(B/c['labels_npz']);indices={int(k):i for i,k in enumerate(d['frame_index'])};native_max=max(indices);dis=cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_FAST);raw={v:[] for v in variants};ffs={v:FrameFeatureComputer() for v in variants};last=None
  with av.open(str(R/c['raw_path'])) as con:
   for native,f in enumerate(con.decode(video=0)):
    if native in indices:
     i=indices[native];im=f.to_ndarray(format='bgr24');h=max(96,int(round(im.shape[0]*256/im.shape[1])));gray=cv2.cvtColor(cv2.resize(im,(256,h)),cv2.COLOR_BGR2GRAY)
     if last is not None:
      flow=dis.calc(last,gray,None)*10
      for name,setting in variants.items():
       mod=flow
       if setting is not None:
        axis,phase=setting;shift=.25*(np.sin(2*np.pi*2*i/10+phase)-np.sin(2*np.pi*2*(i-1)/10+phase))*10;mod=flow.copy();mod[...,axis]+=shift
       raw[name].append(ffs[name](mod))
     last=gray
    if native>=native_max:break
  n=len(d['time']);features={name:combine(np.array(a),n) for name,a in raw.items()};p=R/c['feature_cache'] if c['feature_cache'] else B/(id+'_features.npy');a=np.load(p);cache=a['features'] if isinstance(a,np.lib.npyio.NpzFile) else a;assert np.array_equal(features['base'],cache)
  proba={name:model.predict_proba(x) for name,x in features.items()};pred={name:model.predict(x) for name,x in features.items()};g=prev[prev.id==id].sort_values('index');assert np.array_equal(pred['base'],g.mixed_budget_rav4_prediction)
  target=sel[sel.id==id]['index'].to_numpy();control=g[g.strict_mask&(g.proxy_truth==0)&(g.rav4_prediction==0)&(g.mixed_budget_rav4_prediction==0)]['index'].to_numpy();results[id]={}
  for scope,ix in [('new_reversals',target),('both_correct_acceleration',control)]:
   results[id][scope]={name:{'n':len(ix),'accelerating':int((p[ix]==0).sum()),'decelerating':int((p[ix]==1).sum()),'constant':int((p[ix]==2).sum()),'stopped':int((p[ix]==3).sum()),'changed_from_base':int((p[ix]!=pred['base'][ix]).sum()),'max_abs_decel_probability_change':float(np.max(abs(proba[name][ix,1]-proba['base'][ix,1])))} for name,p in pred.items()}
   for i in ix:
    for name in variants:allrows.append({'id':id,'scope':scope,'index':int(i),'variant':name,'pred':int(pred[name][i]),'p_accel':float(proba[name][i,0]),'p_decel':float(proba[name][i,1])})
  print(id,json.dumps(results[id]),flush=True)
 pd.DataFrame(allrows).to_csv(O/'probe_predictions.csv',index=False);write('results.json',results);assert sha(F/'mixed_budget_rav4.joblib')==json.load(open(O/'freeze.json'))['model_sha256']
if __name__=='__main__':main()
