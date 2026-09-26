"""Small reproducible checks for the frozen acquisition only; never load predictions."""
import hashlib,json
from pathlib import Path
OUT=Path(__file__).resolve().parent;ROOT=OUT.parents[2]
def sha(p):
 with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
reports=[json.loads((OUT/x).read_text()) for x in ['acquisition.json','acquisition_round2.json']]
plans=[json.loads((OUT/x).read_text()) for x in ['selection.json','selection_round2.json']]
records=[x for r in reports for x in r['records']];assert len(records)==6
assert sha(OUT/'selection.json')==reports[0]['selection_sha256']
assert sha(OUT/'selection_round2.json')==reports[1]['selection_sha256']
prior=[]
for raw in plans[0]['local_mp4_inventory']:
 p=ROOT/raw;assert p.is_file();prior.append({'path':raw,'bytes':p.stat().st_size,'sha256':sha(p)})
checks=[]
for r in records:
 p=ROOT/r['source_video'];mapping=OUT/f'{r["id"]}.pts.json';m=json.loads(mapping.read_text())['mapping']
 assert sha(p)==r['provider_sha256']==r['local_sha256'];assert sha(mapping)==r['pts_sha256'];assert len(m)==r['frames']
 assert all(x['frame_id']==i for i,x in enumerate(m));assert all(b['time_s']>a['time_s'] for a,b in zip(m,m[1:]))
 dup=[x['path'] for x in prior if x['sha256']==r['local_sha256']]
 assert not dup
 checks.append({'id':r['id'],'frames':len(m),'native_intervals_s':sorted(set(round(b['time_s']-a['time_s'],12) for a,b in zip(m,m[1:]))),'matching_prior_bytes':dup,'hash_pts_pass':True})
assert len(set(r['local_sha256'] for r in records))==6
summary={'status':'pass','videos':6,'video_bytes':sum(r['bytes'] for r in records),'frames':sum(r['frames'] for r in records),'prior_mp4_sha_comparisons':len(prior),'new_vs_prior_exact_byte_duplicates':0,'within_new_exact_byte_duplicates':0,'same_incident_perceptual_duplicate_audit_complete':False,'independent_eval_certified':False,'checks':checks}
assert summary['video_bytes']<100000000
(OUT/'integrity.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
(OUT/'prior_byte_inventory.json').write_text(json.dumps(prior,indent=2))
print(json.dumps(summary,ensure_ascii=False))
