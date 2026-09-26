"""Integrity/leakage checks for the public proxy; not proof of target-domain validity."""
import collections,json,hashlib
from pathlib import Path
import cv2,numpy as np
from build_stage1_road_pairs import ROOT,OUT,write

def main():
 rows=json.loads((OUT/'prepared_frames.json').read_text());qa=json.loads((OUT/'crop_qa.json').read_text());issues=json.loads((OUT/'preparation_issues.json').read_text());assert not issues
 assert len(rows)==384 and len(qa)==192 and all(r['status']=='pass_geometric_crop_gate' for r in qa)
 groups=collections.defaultdict(list);hashes=collections.defaultdict(set);source_splits=collections.defaultdict(set);cluster_splits=collections.defaultdict(set);features=[]
 for r in rows:
  groups[r['source'],r['label']].append(r);source_splits[r['source']].add(r['split']);cluster_splits[r['cluster']].add(r['split'])
  for field,digest in [('path','sha256'),('parent','parent_sha256')]:
   p=ROOT/r[field];assert hashlib.sha256(p.read_bytes()).hexdigest()==r[digest]
  im=cv2.imread(str(ROOT/r['path']));assert im.shape==(360,640,3);hashes[r['sha256']].add((r['split'],r['label']))
  small=cv2.resize(cv2.cvtColor(im,cv2.COLOR_BGR2GRAY),(9,8));features.append((r,small[:,1:]>small[:,:-1]))
 assert len(groups)==32 and all(len(v)==12 and len({r['frame'] for r in v})==12 for v in groups.values())
 assert all(len(x)==1 for x in source_splits.values()) and all(len(x)==1 for x in cluster_splits.values())
 assert all(len(x)==1 for x in hashes.values())
 a=[(r,x) for r,x in features if r['split']=='development' and r['label']=='original'];b=[(r,x) for r,x in features if r['split']=='holdout' and r['label']=='original']
 nearest=min((int(np.count_nonzero(x!=y)),r['path'],s['path']) for r,x in a for s,y in b)
 report={'status':'PASS_PROXY_INTEGRITY_NOT_TARGET_VALIDATION','sources':len(source_splits),'prepared_images':len(rows),'frames_per_source_class':12,'sources_per_split':dict(collections.Counter(next(iter(v)) for v in source_splits.values())),
 'visual_clusters_per_split':dict(collections.Counter(next(iter(v)) for v in cluster_splits.values())), 'source_overlap':0,'cluster_overlap':0,'cross_label_or_split_exact_duplicates':0,
 'nearest_cross_split_original_dhash_distance':nearest[0],'nearest_files':nearest[1:],'dhash_caveat':'heuristic screening, not proof of independent recording sessions',
 'minimum_geometric_inliers':min(r['inliers'] for r in qa),'minimum_screen_crop_margin_px':min(r['screen_margin_px'] for r in qa), 'maximum_median_reprojection_error_px':max(r['median_error'] for r in qa),
 'files_checked':len(rows)*2,'device_holdout':False,'dashcam_verified':False,'human_review':False}
 write(OUT/'integrity.json',report);print(json.dumps(report,indent=2))
if __name__=='__main__':main()
