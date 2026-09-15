"""Acquire only the four pre-prediction reviewer-agreed replacements."""
import json
from concurrent.futures import ThreadPoolExecutor
from extract_v5_dada import OUT, extract

def main():
    plan=json.loads((OUT/'replacements_frozen.json').read_text(encoding='utf-8'))
    wanted=set(plan['mapping'].values())
    assert wanted=={'10/155','49/026','49/036','50/133'}
    selection=json.loads((OUT/'selection_frozen.json').read_text(encoding='utf-8'))
    index=json.loads((OUT/'selected_members.json').read_text(encoding='utf-8'))['members_by_source']
    sources=[s for s in selection['sources'] if s['source_key'] in wanted]
    assert len(sources)==4 and all(s['role']=='reserve' for s in sources)
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures=[pool.submit(extract,s,index[s['source_key']]) for s in sources]
        for f in futures: print(json.dumps(f.result()),flush=True)

if __name__=='__main__': main()
