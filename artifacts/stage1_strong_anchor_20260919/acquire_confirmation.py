"""Same fixed DLC acquisition with canonical provider URLs and bounded Range validation."""
import sys,json,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'scripts/data'))
import acquire_stage1_anchored_dlc as acquire
acquire.OUT=OUT
session=acquire.requests.Session()

def canonical_get(url,*args,**kwargs):
    # Diagnostic confirmed exact206 Content-Range in1.92s without per-request cache-busting query.
    # The same public URL, headers, size limit, CRC and rate-limit handling remain in use.
    kwargs['timeout']=(10,15)
    return session.get(url.split('&anchor_range=')[0],*args,**kwargs)

acquire.requests.get=canonical_get
if __name__=='__main__':
    assert json.loads((OUT/'decision.json').read_text())['gate_pass']
    acquire.acquire('confirmation')
