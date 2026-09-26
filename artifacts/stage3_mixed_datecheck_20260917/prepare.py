import json,hashlib
from pathlib import Path
O=Path(__file__).resolve().parent;R=O.parents[1];B=R/'artifacts/stage3_training_basis_20260917';P=R/'artifacts/stage3_mixed_routes_20260917'
assert not (O/'freeze.json').exists()
cases=json.loads((B/'cases.json').read_text());old=json.loads((P/'freeze.json').read_text())['held_dates'];eligible={};excluded={}
for v,role in [('rav4','train'),('civic','comparison')]:
 vc=[c for c in cases if c['role']==role];eligible[v]=[];excluded[v]={}
 for date in sorted({c['route'].split('|')[1][:10] for c in vc}):
  selected=[c for c in vc if c['route'].split('|')[1][:10]==date];rest=[c for c in vc if c not in selected]
  a=[sum(c['training_class_counts'][str(k)] for c in selected) for k in range(4)];b=[sum(c['training_class_counts'][str(k)] for c in rest) for k in range(4)]
  if date==old[v]:excluded[v][date]='original validation date, not new'
  elif not all(b):excluded[v][date]='remaining single-vehicle training loses a class'
  elif not all(a[:3]):excluded[v][date]='heldout date lacks one of three moving classes'
  else:eligible[v].append(date)
folds=[{'rav4':a,'civic':b} for a in eligible['rav4'] for b in eligible['civic']];assert folds
source=(P/'run.py').read_text();freeze={'eligible_new_dates':eligible,'excluded_dates':excluded,'folds':folds,'comparison':'mixed_budget_rav4 vs rav4, unchanged prior sampling including RNG order; all5models retained for exact method reuse','primary_checks':'F1 improvement, overall and per-vehicle opposite-count nonincrease, publicF1 nonregression. Opposite rates denominator true accelerating+decelerating; report directions and paired transitions. No promotion based on averages hiding failures.','not_independent':'same two vehicles, exposed proxy labels; overlapping heldout dates across folds, no pooled repeated rows score','prior_script_sha256':hashlib.sha256(source.encode()).hexdigest()}
(O/'freeze.json').write_text(json.dumps(freeze,indent=2))
for i,hold in enumerate(folds,1):
 p=O/f'fold_{i}';p.mkdir();s=source.replace("R=O.parents[1]","R=O.parents[2]")
 a=s.index(' hold={}');b=s.index(' for c in cases:c[\'experiment_role\']',a)
 s=s[:a]+' hold='+repr(hold)+'\n'+s[b:]
 s=s.replace("'selection':'earliest date per vehicle with all3 moving classes and all4 classes left in training; no outcome-dependent selection'","'selection':'predeclared alternative dates from parent freeze; same label-support rule'")
 (p/'run.py').write_text(s)
print(json.dumps(freeze,indent=2))
