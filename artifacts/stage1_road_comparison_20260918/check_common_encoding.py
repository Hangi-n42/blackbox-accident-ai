import sys,json,collections
from pathlib import Path
sys.path.insert(0,str(Path.cwd()/'scripts/data'))
from diagnose_uhdm_branches import ROOT,RELEASE,TPODetector,cv2,np,torch,extract_features,feature_probability,write,sha
DATA=ROOT/'artifacts/stage1_road_pairs_20260918';OUT=ROOT/'artifacts/stage1_road_comparison_20260918'
protocol=json.loads((OUT/'protocol.json').read_text());model=RELEASE/'model/stage1';artifact=json.loads((model/protocol['config']['forensic_artifact']).read_text());selected=json.loads((OUT/'selected.json').read_text())['selected']
frames=json.loads((DATA/'prepared_frames.json').read_text());qa={(r['source'],r['frame']):r for r in json.loads((DATA/'crop_qa.json').read_text())};baseline={(r['source'],r['label']):r for s in ['development','holdout'] for r in json.loads((OUT/f'{s}_scores.json').read_text()) if r['mode']=='full_frame'}
groups=collections.defaultdict(list)
for r in frames:groups[r['source'],r['label']].append(r)
rows=[];torch.set_num_threads(2);cv2.setNumThreads(2);detector=TPODetector(model/'tpo',device='cpu')
try:
 for (source,label),items in sorted(groups.items()):
  arrays=[]
  for r in sorted(items,key=lambda x:x['frame']):
   assert sha(ROOT/r['parent'])==r['parent_sha256'];a=cv2.imread(str(ROOT/r['parent']));x0,y0,x1,y1=qa[source,r['frame']][label+'_rect'];a=cv2.resize(a[y0:y1,x0:x1],(640,360),interpolation=cv2.INTER_AREA);arrays.append(cv2.cvtColor(a,cv2.COLOR_BGR2RGB))
  f,_=extract_features(arrays);fp=feature_probability(f,artifact);tp=float(detector.score(arrays).mean());b=baseline[source,label]
  rows.append({'source':source,'label':label,'split':items[0]['split'],'native_crop_forensic':fp,'native_crop_tpo':tp,'jpeg95_forensic':b['forensic'],'jpeg95_tpo':b['tpo'],'current_native':.5*(fp+tp)>=.5,'current_jpeg95':b['ensemble']>=.5,'selected_native':tp>=selected['threshold'],'selected_jpeg95':b['tpo']>=selected['threshold']})
finally:detector.close()
write(OUT/'common_encoding_check.json',{'scope':'Post-evaluation preprocessing sensitivity only; selected settings unchanged; not a new holdout','rows':rows,'forensic_decision_flips':sum((r['native_crop_forensic']>=.5)!=(r['jpeg95_forensic']>=.5) for r in rows),'current_decision_flips':sum(r['current_native']!=r['current_jpeg95'] for r in rows),'selected_decision_flips':sum(r['selected_native']!=r['selected_jpeg95'] for r in rows)})
print('checks done',len(rows))
