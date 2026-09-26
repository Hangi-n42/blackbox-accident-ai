"""Use the unchanged DIS feature extractor on nearest native ZOD frames."""
from qa import O,R,read,write
import sys,time,importlib.util
import numpy as np,cv2,joblib
spec=importlib.util.spec_from_file_location('factor',R/'artifacts/stage3_factor_comparison_20260918/run.py');factor=importlib.util.module_from_spec(spec);spec.loader.exec_module(factor)
def extract(sid):
 out=O/'features';out.mkdir(exist_ok=True);target=out/(sid+'.npz')
 if target.exists():return
 start=time.monotonic();d=np.load(O/'labels'/(sid+'.npz'));frames=read(O/'raw/sequences'/sid/'info.json')['camera_frames']['front_blur'];ff=factor.FrameFeatureComputer();dis=cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_FAST)
 raw=[];prev=None;files=[]
 for k in d['frame_index']:
  p=O/'raw'/frames[int(k)]['filepath'];im=cv2.imread(str(p));assert im is not None and im.shape[:2]==(2168,3848),p
  gray=cv2.cvtColor(cv2.resize(im,(256,max(96,round(im.shape[0]*256/im.shape[1])))),cv2.COLOR_BGR2GRAY)
  if prev is not None:raw.append(ff(dis.calc(prev,gray,None)*10))
  prev=gray;files.append(str(p.relative_to(R)))
 x=factor.combine(raw,len(files));assert x.shape==(len(files),864) and np.isfinite(x).all()
 np.savez_compressed(target,base=x);write(out/(sid+'.json'),{'id':sid,'frames':len(files),'seconds':time.monotonic()-start,'sources':files,'images_decode_pass':True,'sensor_input_to_model':False})
 print('extracted',sid,len(files),round(time.monotonic()-start,2),flush=True)
def pilot():
 import sklearn.metrics as sm
 prod=joblib.load(R/'releases/v7/source/model/stage3/motion_model.joblib')['accel'];rows=[];truth=[];pred=[]
 for sid in read(O/'freeze.json')['pilot_ids']:
  extract(sid);x=np.load(O/'features'/(sid+'.npz'))['base'];d=np.load(O/'labels'/(sid+'.npz'));p=prod.predict_proba(x);ix=np.flatnonzero(d['diagnostic_accel_mask']);y=d['accel_candidate'][ix];pp=p[ix].argmax(1);truth.extend(y.tolist());pred.extend(pp.tolist())
  rows.append({'id':sid,'n':len(ix),'confusion':sm.confusion_matrix(y,pp,labels=range(4)).tolist(),'class_support':np.bincount(y,minlength=4).tolist()});np.savez_compressed(O/'features'/(sid+'_production_prediction.npz'),prob=p,eval_indices=ix)
 write(O/'pilot_baseline.json',{'production_sha256':read(O/'freeze.json')['protected']['releases/v7/source/model/stage3/motion_model.joblib'],'videos':rows,'n':len(truth),'confusion':sm.confusion_matrix(truth,pred,labels=range(4)).tolist(),'proxy_macro_f1_4class':float(sm.f1_score(truth,pred,labels=range(4),average='macro',zero_division=0)),'note':'Pilot development diagnostic; no STOPPED or strict DECELERATING support; not S3 or representative accuracy.'})
if __name__=='__main__':
 if sys.argv[1]=='pilot':pilot()
 else:
  for row in read(O/'split_manifest.json'):
   if row['role']!='excluded':extract(row['id'])
