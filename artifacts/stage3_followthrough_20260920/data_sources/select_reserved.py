"""Select nine sensor-defined windows per new route, minimizing clip count."""
from pathlib import Path
import json,itertools
import numpy as np
from scipy.optimize import milp, Bounds, LinearConstraint
from scipy.sparse import lil_matrix
O=Path(__file__).resolve().parent;A=O/'acquired'
records=[json.loads(p.read_text()) for p in sorted((A/'records').glob('*.json'))]
allmeta=json.loads((O/'remote_segment_sizes.json').read_text());meta={s['segment']:dict(s,source_url=r['source_url'],route=r['route'],vehicle=r['vehicle']) for r in allmeta for s in r['segments']}
results=[]
for route in sorted({r['route'] for r in records}):
 rr=[r for r in records if r['route']==route];wins=[w for r in rr for w in r.get('windows',[])];segs=sorted({w['segment'] for w in wins});N=len(wins);S=len(segs)
 counts={str(k):sum(w['label']==k for w in wins) for k in range(3)};rec={'route':route,'vehicle':rr[0]['vehicle'],'candidate_windows':counts,'source_segments_inspected':len(rr)}
 if min(counts.values())<3:rec.update(status='insufficient_class_candidates');results.append(rec);continue
 low=min(w['speed_m_s'] for w in wins);high=max(w['speed_m_s'] for w in wins);M=high-low+3
 constraints=[]
 for k in range(3):constraints.append(({i:1 for i,w in enumerate(wins) if w['label']==k},3,3))
 for i,w in enumerate(wins):
  j=N+segs.index(w['segment']);constraints.append(({i:1,j:-1},-np.inf,0))
  # If x_i=1, v_i-2 <= L <= v_i. L is shared lower end of the 2m/s band.
  constraints.append(({i:M,N+S:1},-np.inf,w['speed_m_s']+M))
  constraints.append(({i:M,N+S:-1},-np.inf,2-w['speed_m_s']+M))
 for i,a in enumerate(wins):
  for j in range(i+1,N):
   b=wins[j]
   if abs((a['start_boot_s']+a['end_boot_s']-b['start_boot_s']-b['end_boot_s'])/2)<3-1e-7:constraints.append(({i:1,j:1},-np.inf,1))
 mat=lil_matrix((len(constraints),N+S+1));lb=[];ub=[]
 for n,(co,l,u) in enumerate(constraints):
  for j,v in co.items():mat[n,j]=v
  lb.append(l);ub.append(u)
 c=np.zeros(N+S+1);c[N:N+S]=1+np.arange(S)*1e-5
 bounds=Bounds(np.r_[np.zeros(N+S),low-2],np.r_[np.ones(N+S),high])
 ans=milp(c,integrality=np.r_[np.ones(N+S),0],bounds=bounds,constraints=LinearConstraint(mat.tocsr(),lb,ub),options={'time_limit':20,'mip_rel_gap':0})
 rec.update(solver_status=int(ans.status),solver_message=ans.message)
 if ans.x is None:rec.update(status='no_solution_returned');results.append(rec);continue
 selected=[w for i,w in enumerate(wins) if ans.x[i]>.5];selectedsegs=sorted({w['segment'] for w in selected})
 assert len(selected)==9 and all(sum(w['label']==k for w in selected)==3 for k in range(3))
 assert max(w['speed_m_s'] for w in selected)-min(w['speed_m_s'] for w in selected)<=2+1e-6
 assert all(abs((a['start_boot_s']+a['end_boot_s']-b['start_boot_s']-b['end_boot_s'])/2)>=3-1e-7 for a,b in itertools.combinations(selected,2))
 rec.update(status='feasible_sensor_only',optimal_minimum_segments=ans.status==0,selected_windows=selected,selected_segments=selectedsegs,number_of_segments=len(selectedsegs),video_compressed_bytes=sum(next(m['compressed_bytes'] for m in meta[s]['members'] if m['name'].endswith('/video.hevc')) for s in selectedsegs),speed_range_m_s=[min(w['speed_m_s'] for w in selected),max(w['speed_m_s'] for w in selected)])
 results.append(rec);print(route,rec['status'],len(selectedsegs),rec['video_compressed_bytes'],flush=True)
feasible=[r for r in results if r['status']=='feasible_sensor_only']
pairs=[(a['video_compressed_bytes']+b['video_compressed_bytes'],a,b) for a,b in itertools.combinations(feasible,2) if a['vehicle']!=b['vehicle']]
pairs.sort(key=lambda p:(p[0],p[1]['route'],p[2]['route']))
chosen=[]
if pairs and pairs[0][0]<=200_000_000:chosen=[pairs[0][1],pairs[0][2]]
plan={'rule':'9windows/route, 3each A/D/C, common2m/s speed band, >=3sec center spacing, minimumclips MILP; prioritize two different known vehicles within200MB video payload cap','all_route_results':results,'chosen_routes':chosen,'status':'two_routes_ready_for_video' if chosen else 'budget_or_coverage_insufficient','video_compressed_bytes':sum(r['video_compressed_bytes'] for r in chosen),'sensor_only_diagnosis':True,'no_predictions_inspected':True}
(A/'selection_plan.json').write_text(json.dumps(plan,indent=2)+'\n');print('FINAL',plan['status'],plan['video_compressed_bytes'],flush=True)
