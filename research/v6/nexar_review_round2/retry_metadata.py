"""Retry only the frozen metadata request, preserving its initial failure/plan."""
import sys,importlib.util,json,datetime
from pathlib import Path
sys.dont_write_bytecode=True
here=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('round2_metadata',here/'prepare_metadata.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
original_write=m.write
def write(name,value):
    if name=='selection_policy_frozen.json':
        old=json.loads((here/name).read_text(encoding='utf8'))
        assert {k:v for k,v in old.items() if k!='created_utc'}=={k:v for k,v in value.items() if k!='created_utc'},'Frozen selection changed'
        original_write('metadata_retry.json',dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),frozen_policy_sha256=m.sha(here/name),retry_script_sha256=m.sha(__file__),selection_unchanged=True))
        return
    if name=='metadata_access_failure.json':name='metadata_retry_failure.json'
    original_write(name,value)
m.write=write
m.main()
