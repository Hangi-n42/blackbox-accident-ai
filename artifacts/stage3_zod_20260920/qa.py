"""Bounded ZOD QA and sensor-proxy labels; no model predictions used here."""
from pathlib import Path
import sys,json,hashlib
from datetime import datetime
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
O=Path(__file__).resolve().parent;R=O.parents[1]
sys.path.insert(0,str(O/'deps'))
import h5py
read=lambda p:json.loads(p.read_text())
write=lambda p,x:p.write_text(json.dumps(x,indent=2,ensure_ascii=False))
stamp=lambda s:datetime.fromisoformat(s.replace('Z','+00:00')).timestamp()
def align(t,v,q,gap):
 assert np.all(np.diff(t)>0)
 j=np.clip(np.searchsorted(t,q),1,len(t)-1)
 mask=(q>=t[0])&(q<=t[-1])&(t[j]-t[j-1]<=gap)&np.isfinite(v[j])&np.isfinite(v[j-1])
 out=np.interp(q,t,v);out[~mask]=np.nan
 return out,mask,j
def labels(v):
 a=np.full(len(v),np.nan);vs=a.copy();xx=np.arange(-5,6)*.1
 a[5:-5]=(sliding_window_view(v,11)*xx).sum(1)/(xx@xx);vs[5:-5]=sliding_window_view(v,11).mean(1)
 y=np.full(len(v),-1,np.int8);y[(vs<.2)&np.isfinite(a)]=3
 y[(vs>.4)&(a>.3)]=0;y[(vs>.4)&(a<-.3)]=1;y[(vs>.4)&(abs(a)<.2)]=2
 changes=np.flatnonzero((y[1:]!=y[:-1])&(y[1:]>=0)&(y[:-1]>=0))+1
 for k in changes:y[max(0,k-3):k+4]=-1
 strict=((y==3)&(vs<.1))|((vs>1)&(((y==0)&(a>.5))|((y==1)&(a<-.5))|((y==2)&(abs(a)<.1))))
 mask=np.zeros(len(v),bool)
 for k in range(5,len(v)-5):mask[k]=strict[k-5:k+6].all() and (y[k-5:k+6]==y[k]).all()
 return y,a,vs,mask
def one(sid,require_raw=True):
 p=O/'raw/sequences'/sid;info=read(p/'info.json');meta=read(p/'metadata.json');e=read(p/'ego_motion.json')
 frames=info['camera_frames']['front_blur'];ft=np.array([stamp(f['time']) for f in frames]);assert np.all(np.diff(ft)>0)
 q=ft[0]+np.arange(int(np.floor((ft[-1]-ft[0])*10))+1)/10
 right=np.clip(np.searchsorted(ft,q),1,len(ft)-1);fi=np.where(abs(ft[right]-q)<abs(ft[right-1]-q),right,right-1)
 t=np.array(e['timestamps']);vv=np.array(e['velocities'])[:,0];v,valid,j=align(t,vv,q,.15)
 report={'id':sid,'vehicle':meta['collection_car'],'group':meta['collection_car']+'_'+meta['start_time'][:10],
 'start':meta['start_time'],'end':meta['end_time'],'longitude':meta['longitude'],'latitude':meta['latitude'],
 'native_frames':len(ft),'samples10Hz':len(q),'max_frame_error_s':float(abs(ft[fi]-q).max()),'frame_duplicate_count':int(len(fi)-len(np.unique(fi))),
 'metadata_max_gap_s':float(np.diff(t).max()),'raw_oxts_available':(p/'oxts.hdf5').exists(), 'truth_kind':'sensor_proxy_trial_thresholds_not_official',
 'source':'Zenseact Open Dataset V2 sequences','license':'CC BY-SA 4.0','source_url':'https://zod.zenseact.com/sequences/',
 'source_paths':[str((p/x).relative_to(R)) for x in ['info.json','ego_motion.json','metadata.json','calibration.json']]}
 raw_ok=not require_raw
 if (p/'oxts.hdf5').exists():
  with h5py.File(p/'oxts.hdf5') as h:
   rt=315964800+h['timestamp'][:]+h['leapSeconds'][:];rv=h['velForward'][:]
   rv_q,raw_valid,rj=align(rt,rv,q,.03)
   flags=(h['isValidNorthVelocity'][:]==1)&(h['isValidEastVelocity'][:]==1)&(h['isValidXY'][:]==1)
   raw_valid &= flags[rj]&flags[rj-1]
   # Published generic isValid has no documented semantics in devkit; do not silently call it passed.
   std=np.maximum(h['stdDevVelNorth'][:],h['stdDevVelEast'][:])
   raw_valid &= np.isfinite(std[rj])&(std[rj]<=.5)&np.isfinite(std[rj-1])&(std[rj-1]<=.5)
   # Compare the published interpolation at its native sample times, not a second interpolation grid.
   raw_error=float(abs(vv-np.interp(t,rt,rv)).max())
   valid &= raw_valid
   report.update(raw_vs_metadata_max_m_s=raw_error,raw_velocity_invalid=int((~raw_valid).sum()),
       generic_isValid_values=np.unique(h['isValid'][:]).tolist(),generic_isValid_interpretation='undocumented; explicit North/East/XY flags and velocity std gates used; not a generic validity pass',
       raw_time_utc_formula='GPS epoch315964800 + timestamp + leapSeconds',raw_dt_range_s=[float(np.diff(rt).min()),float(np.diff(rt).max())],
       velocity_std_p50_p99_m_s=np.quantile(std[rj],[.5,.99]).tolist(),double_interpolation_max_difference_m_s=float(np.nanmax(abs(v-rv_q))))
   raw_ok=raw_error<.001
  report['source_paths'].append(str((p/'oxts.hdf5').relative_to(R)))
 v[~valid]=np.nan;y,a,vs,strict=labels(v)
 pos=np.array(e['poses'])[:,:3,3];pv=np.linalg.norm(np.diff(pos,axis=0),axis=1)/np.diff(t)
 pe=float(np.quantile(abs(pv-(vv[1:]+vv[:-1])/2),.99))
 report['pose_velocity_abs_error_p99_m_s']=pe
 if (p/'vehicle_data.hdf5').exists():
  with h5py.File(p/'vehicle_data.hdf5') as h:
   k='ego_vehicle_data/';ct=h[k+'timestamp/nanoseconds/value'][:]*1e-9;cv=h[k+'lon_vel_data/velocity/meters_per_second/value'][:]
   can,cm,cj=align(ct,cv,q,.05);cm &= (h[k+'valid/unitless/value'][:][cj]==1)&(h[k+'valid/unitless/value'][:][cj-1]==1)
   can[~cm]=np.nan;cy,ca,_,_=labels(can);sel=(y>=0)&(cy>=0)
   report.update(can_valid_points=int(cm.sum()),can_vs_oxts_speed_max_m_s=float(np.nanmax(abs(can-v))) if np.isfinite(can-v).any() else None,can_label_agreement=float(np.mean(y[sel]==cy[sel])) if sel.any() else None,
       can_accel_slope_rmse=float(np.sqrt(np.nanmean((a-ca)**2))) if np.isfinite(a-ca).any() else None,can_comparison_n=int(sel.sum()))
   report['source_paths'].append(str((p/'vehicle_data.hdf5').relative_to(R)))
 report.update(usable_counts=np.bincount(y[y>=0],minlength=4).tolist(),strict_counts=np.bincount(y[strict],minlength=4).tolist(),
       training2Hz_counts=np.bincount(y[(np.arange(len(y))%5==0)&(y>=0)],minlength=4).tolist(),invalid_speed=int((~valid).sum()),
       structural_qa_pass=bool(raw_ok and report['max_frame_error_s']<=.056 and pe<.5 and (y>=0).sum()>=20))
 return report,dict(time=q,frame_index=fi,frame_time=ft[fi],speed=v,speed_valid=valid,speed_smoothed=vs,acceleration_proxy=a,accel_candidate=y,diagnostic_accel_mask=strict)
def main(end,raw):
 dest=O/('labels' if raw else 'metadata_preview');dest.mkdir(exist_ok=True);reports=[]
 for i in range(end+1):
  sid=f'{i:06}';r,d=one(sid,raw);reports.append(r)
  np.savez_compressed(dest/(sid+'.npz'),**d)
 write(dest/'qa.json',reports)
 print(json.dumps([{'id':x['id'],'group':x['group'],'qa':x['structural_qa_pass'],'train':x['training2Hz_counts'],'strict':x['strict_counts']} for x in reports],indent=2))
if __name__=='__main__':main(int(sys.argv[1]),'--raw' in sys.argv)
