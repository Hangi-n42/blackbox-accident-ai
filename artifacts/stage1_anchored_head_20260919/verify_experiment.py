"""Independent integrity and metric checks; never fits or selects a model."""
import collections,hashlib,json,sys
from pathlib import Path
import numpy as np
from sklearn.metrics import f1_score,confusion_matrix
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'scripts/data'))
import experiment_stage1_anchored_head as e

def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
pro=e.read(OUT/'protocol.json')
for p,h in pro['frozen_sha256'].items(): assert sha(ROOT/p)==h,p
for p,h in e.read(OUT/'helper_source_hashes.json').items(): assert sha(ROOT/p)==h,p
split=e.read(OUT/'split_integrity.json');sources={};shapes={};allhashes=collections.defaultdict(set)
for s in ['train','development','confirmation']:
 p=OUT/f'dlc_{s}_acquired.json'
 if not p.exists():continue
 rows=e.read(p);expected=sum(12 for r in e.read(OUT/'dlc_plan.json')['videos'] if r['split']==s)
 assert len(rows)==expected,(s,len(rows),expected)
 for r in rows:
  assert sha(ROOT/r['path'])==r['sha256']
  allhashes[r['sha256']].add(s)
 sources[s]={'frames':len(rows),'document_ids':len({r['document_id'] for r in rows}),'video_ids':len({r['video_id'] for r in rows}),'types':sorted({r['document_type'] for r in rows})}
assert all(len(s)==1 for s in allhashes.values()),'cross-split exact image duplicate'
for d,s,n in [('comma','train',11),('comma','development',12),('public','guard',10),('dlc','train',40),('dlc','development',24)]:
 x,rs=e.dataset(d,s);assert len({r['video'] for r in rs})==n
 assert len(x)==n*24*(1 if d=='public' else 3)
 assert all(r['count']==24 for r in rs) and np.isfinite(x).all()
 assert np.max(abs(np.linalg.norm(x,axis=1)-1))<1e-5
 shapes[d+'_'+s]=list(x.shape)
training=e.read(OUT/'training.json')
for name,r in training['models'].items():assert sha(OUT/f'{name}.npz')==r['sha256']
checked=[];intervals={};rng=np.random.default_rng(20260919)
clusters=e.read(ROOT/'artifacts/stage1_road_pairs_20260918/selection.json')['location_clusters']
for path in sorted(OUT.glob('*_predictions.json')):
 rows=e.read(path);summary=e.read(path.with_name(path.name.replace('_predictions','_summary')))
 y=np.array([r['label']=='recapture' for r in rows])
 for name,s in summary.items():
  pp=np.array([r['scores'][name] for r in rows])>=.5;tn,fp,fn,tp=confusion_matrix(y,pp,labels=[False,True]).ravel()
  m=s['overall'];assert (fp,fn)==(m['fp'],m['fn'])
  if y.any() and (~y).any():assert abs(f1_score(y,pp,average='macro')-m['macro_f1'])<1e-12
 checked.append(path.name)
 if y.any() and (~y).any():
  if path.name.startswith('vd_development'):groups=[clusters[str(r['source'])] for r in rows]
  elif path.name.startswith('dlc_'):groups=[r['cluster'] for r in rows]
  else:groups=[r['group'] for r in rows]
  unique=sorted(set(groups));indexes={g:np.where(np.array(groups)==g)[0] for g in unique}
  for name in summary:
   if name=='v7':continue
   b=np.array([r['scores']['v7'] for r in rows])>=.5;c=np.array([r['scores'][name] for r in rows])>=.5;diff=[]
   # ponytail: 1000 grouped bootstrap draws; too few clusters remain a data limitation.
   for _ in range(1000):
    ix=np.concatenate([indexes[g] for g in rng.choice(unique,len(unique),replace=True)])
    diff.append(f1_score(y[ix],c[ix],average='macro')-f1_score(y[ix],b[ix],average='macro'))
   intervals[path.stem+'_'+name]={'clusters':unique,'n_clusters':len(unique),'delta_f1_95_percentile':np.quantile(diff,[.025,.975]).tolist(),'scope':'exploratory grouped bootstrap; development model selection and few clusters prevent independent significance claim'}
result={'status':'PASS','data_counts':sources,'feature_shapes':shapes,'all_frozen_hashes_unchanged':True,'exact_cross_split_image_duplicates':0,'independent_metric_checks':checked,'bootstrap':intervals,'selection':e.read(OUT/'selection.json')}
e.write(OUT/'verification.json',result);print(json.dumps(result,indent=2))
