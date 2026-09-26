"""JSON serialization-only retry; preserve the frozen training script."""
import run,json
import numpy as np
paths=[run.O/n for n in ['metrics.json','metrics.csv','changes.json','changed_predictions.csv']]
before={str(p):run.sha(p) for p in paths}
def write(path,value):
 path.write_text(json.dumps(value,indent=2,ensure_ascii=False,default=lambda v:bool(v) if isinstance(v,np.bool_) else (_ for _ in ()).throw(TypeError(type(v).__name__))))
run.write=write
run.evaluate()
assert all(run.sha(p)==before[str(p)] for p in paths)
write(run.O/'serialization_retry.json',{'reason':'numpy.bool_ success-gate serialization failed after metrics saved; only conversion to builtin bool changed','metrics_and_changes_byte_identical':True,'training_rerun':False})
