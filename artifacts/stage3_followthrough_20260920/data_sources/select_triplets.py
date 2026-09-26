"""Three same-speed A/D/C triplets per route; not one shared speed for all9."""
from pathlib import Path
import json,itertools
import numpy as np
from scipy.optimize import milp,Bounds,LinearConstraint
from scipy.sparse import lil_matrix
O=Path(__file__).resolve().parent;A=O/'acquired'
read=lambda p:json.loads(p.read_text())
records=[read(p) for p in sorted((A/'records').glob('*.json'))];metas=read(O/'remote_segment_sizes.json');meta={s['segment']:s for r in metas for s in r['segments']}
center=lambda w:(w['start_boot_s']+w['end_boot_s'])/2
results=[]
for route in sorted({r['route'] for r in records}):
 rr=[r for r in records if r['route']==route];wins=[w for r in rr for w in r.get('windows',[])];segs=sorted({w['segment'] for w in wins});classes=[[i for i,w in enumerate(wins) if w['label']==k] for k in range(3)]
 triples=[]
 for ix in itertools.product(*classes):
  ww=[wins[i] for i in ix]
  if max(w['speed_m_s'] for w in ww)-min(w['speed_m_s'] for w in ww)>2:continue
  if any(abs(center(a)-center(b))<3-1e-7 for a,b in itertools.combinations(ww,2)):continue
  triples.append(ix)
 T=len(triples);S=len(segs);rec={'route':route,'vehicle':rr[0]['vehicle'],'candidate_windows':{str(k):len(classes[k]) for k in range(3)},'candidate_triplets':T,'source_segments_inspected':len(rr)}
 if T<3:rec.update(status='insufficient_matching_triplets');results.append(rec);continue
 covering=[set() for w in wins]
 for j,t in enumerate(triples):
  for i in t:covering[i].add(j)
 constraints=[({j:1 for j in range(T)},3,3)]
 for ix in covering:
  if ix:constraints.append(({j:1 for j in ix},-np.inf,1))
 for i,a in enumerate(wins):
  for j in range(i+1,len(wins)):
   if abs(center(a)-center(wins[j]))<3-1e-7:
    union=covering[i]|covering[j]
    if union:constraints.append(({q:1 for q in union},-np.inf,1))
 for j,t in enumerate(triples):
  for s in {wins[i]['segment'] for i in t}:constraints.append(({j:1,T+segs.index(s):-1},-np.inf,0))
 mat=lil_matrix((len(constraints),T+S));lo=[];hi=[]
 for i,(d,l,h) in enumerate(constraints):
  for j,v in d.items():mat[i,j]=v
  lo.append(l);hi.append(h)
 c=np.r_[np.zeros(T),1+np.arange(S)*1e-5]
 ans=milp(c,integrality=np.ones(T+S),bounds=Bounds(np.zeros(T+S),np.ones(T+S)),constraints=LinearConstraint(mat.tocsr(),lo,hi),options={'time_limit':30,'mip_rel_gap':0})
 rec.update(solver_status=int(ans.status),solver_message=ans.message)
 if ans.x is None:rec.update(status='no_solution_returned');results.append(rec);continue
 chosen=[t for j,t in enumerate(triples) if ans.x[j]>.5];selected=[dict(wins[i],triplet_index=n) for n,t in enumerate(chosen) for i in t];ss=sorted({w['segment'] for w in selected})
 assert len(chosen)==3 and len(selected)==9 and len({(w['segment'],w['center_index']) for w in selected})==9
 assert all(abs(center(a)-center(b))>=3-1e-7 for a,b in itertools.combinations(selected,2))
 assert all(max(wins[i]['speed_m_s'] for i in t)-min(wins[i]['speed_m_s'] for i in t)<=2 for t in chosen)
 rec.update(status='feasible_sensor_only',minimum_segments_certified=ans.status==0,selected_windows=selected,selected_segments=ss,number_of_segments=len(ss),video_compressed_bytes=sum(next(m['compressed_bytes'] for m in meta[s]['members'] if m['name'].endswith('/video.hevc')) for s in ss))
 results.append(rec);print(route,rec['status'],len(ss),rec['video_compressed_bytes'],flush=True)
feasible=[r for r in results if r['status']=='feasible_sensor_only'];pairs=[(a['video_compressed_bytes']+b['video_compressed_bytes'],a,b) for a,b in itertools.combinations(feasible,2) if a['vehicle']!=b['vehicle']];pairs.sort(key=lambda x:(x[0],x[1]['route'],x[2]['route']))
chosen=[pairs[0][1],pairs[0][2]] if pairs and pairs[0][0]<=200_000_000 else []
plan={'design_correction':'The first all9-window common speed band was unnecessarily stronger than original paired comparison. Preserved as a separate diagnostic. Here each of3 A/D/C triplets has range<=2m/s; bands can differ. No state, steering, or nonoverlap threshold changed; sensor-only selection before predictions.','all_route_results':results,'chosen_routes':chosen,'status':'two_routes_ready_for_video' if chosen else 'budget_or_coverage_insufficient','video_compressed_bytes':sum(r['video_compressed_bytes'] for r in chosen),'no_model_predictions_inspected':True}
(A/'triplet_selection_plan.json').write_text(json.dumps(plan,indent=2)+'\n');print('FINAL',plan['status'],plan['video_compressed_bytes'],flush=True)
