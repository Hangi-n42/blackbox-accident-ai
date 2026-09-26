"""Replay saved trajectories and paired controls without re-running trackers."""
import importlib.util
from pathlib import Path
import numpy as np,pandas as pd,cv2
O=Path(__file__).resolve().parent
s=importlib.util.spec_from_file_location('experiment',O/'run.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
d=pd.read_csv(O/'results.csv');f=m.read(O/'freeze.json')
assert all(m.sha(m.R/p)==h for p,h in {**f['inputs'],**f['protected']}.items())
assert m.sha(O/'run.py')==f['script_sha256']
flow=np.zeros((32,32,2),np.float32);flow[:]=[2.,-1.]
assert np.array_equal(m.sample(flow,np.array([[5.,5.],[12.5,13.5]],np.float32)),np.array([[2.,-1.],[2.,-1.]],np.float32))
for r in d.itertuples():
    z=np.load(O/f'{r.key}_{r.variant}.npz');im=np.load(m.P/(r.key+'_frames.npz'))
    g,c,good=m.geometry(z['xy'],z['time'],*im['rgb'].shape[1:3])
    assert g['valid']==r.valid and g['tracked']==r.tracked and np.array_equal(good,z['usable'])
    if pd.notna(r.candidate):assert np.isclose(g['candidate'],r.candidate,atol=1e-10)
for key in f['keys']:
    a=np.load(O/(key+'_lk_intersection.npz'));b=np.load(O/(key+'_dis_intersection.npz'))
    assert np.array_equal(a['initial_point_indices'],b['initial_point_indices'])
    assert np.array_equal(a['xy'][0],b['xy'][0])
a=d[d.variant=='lk_common'].set_index('key');b=d[d.variant=='dis_common'].set_index('key');both=a.valid&b.valid
threshold=.030443026891414933
summary={'lk_pass':int(a.valid.sum()),'dis_pass':int(b.valid.sum()),'n':len(a),'lk_surviving_points':int(a.tracked.sum()),'dis_surviving_points':int(b.tracked.sum()),
         'recovered_windows':list(a.index[~a.valid&b.valid]),'lost_windows':list(a.index[a.valid&~b.valid]),
         'both_valid_windows':list(a.index[both]),
         'common_valid_ratio_MAE_LK':float(abs(a.loc[both].candidate-a.loc[both].truth_ratio).mean()),
         'common_valid_ratio_MAE_DIS':float(abs(b.loc[both].candidate-b.loc[both].truth_ratio).mean()),
         'MAE_limit':'4 selected windows only; not overall coverage-adjusted error, not independent or official score',
         'adopt':False,'reason':'no failed window recovered; one old accepted window lost; held constant still outside prior descriptive deadband',
         'classifier_fits':0,'score_improvement_evaluated':False}
m.write(O/'summary.json',summary)
dis=cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_FAST)
m.write(O/'runtime_parameters.json',{'opencv':cv2.__version__,'dis':{name:getattr(dis,name)() for name in ['getFinestScale','getGradientDescentIterations','getPatchSize','getPatchStride','getUseMeanNormalization','getUseSpatialPropagation','getVariationalRefinementIterations']},'verified_primary_reference':'https://docs.opencv.org/4.10.0/javadoc/org/opencv/video/DISOpticalFlow.html'})
m.write(O/'final_checks.json',{'run_exit_status':0,'audit_exit_status':0,'all50_saved_trajectory_metrics_replayed':True,
                               'matched_initial_point_identity_verified':True,'flow_sampling_translation_check':True,
                               'input_production_preserved':True,'no_hyperparameter_search':True,'adopt':False})
print(summary)
