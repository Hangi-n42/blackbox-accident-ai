"""Read-only counts of existing source-side selection gates; not a new candidate."""
from masks import *
import sys
sys.path.insert(0,str(B))
import geometry as g
rows=[]
for key in POLYGONS:
 rgb=np.load(B/'depth'/(key+'_input.npz'))['rgb'];base=np.load(B/'depth'/(key+'_first.npz'));m=np.load(O/'masks'/(key+'.npz'))['mask'];flow=np.load(B/'flow'/(key+'.npz'));h,w=m.shape[1:];grid=g.roi_grid(h,w);counts=[]
 for j in range(20):
  bg=inside(m[j],grid);gray=cv2.cvtColor(rgb[j],cv2.COLOR_RGB2GRAY).astype('float32');std=np.sqrt(np.maximum(cv2.blur(gray*gray,(7,7))-cv2.blur(gray,(7,7))**2,0));texture=g.sample(std,grid)>=4;c=g.sample(base['depth_conf'][j],grid);conf=c>=np.median(c)
  counts.append({'pair':j,'grid_in_mask':int(bg.sum()),'texture_pass_only':int((bg&texture).sum()),'depth_confidence_pass_only':int((bg&conf).sum()),'both_source_gates_pass':int((bg&texture&conf).sum()),'existing_accepted_source_in_mask':int(inside(m[j],flow[f'p{j}']).sum()),'both_endpoints_in_mask':int((inside(m[j],flow[f'p{j}'])&inside(m[j+1],flow[f'q{j}'])).sum())})
 sums={name:sum(c[name] for c in counts) for name in counts[0] if name!='pair'};rows.append({'key':key,'counts':counts,'sums':sums});print(key,sums)
(O/'filter_audit.json').write_text(json.dumps({'scope':'Source-side texture and confidence counts on the original8px grid. No optical flow recomputation, no changed thresholds. These gates overlap; counts are not causal percentages. Endpoint/forward-backward effects remain combined in existing accepted matches.','rows':rows},ensure_ascii=False,indent=2))
