"""Paired interval evaluation for the one automatic first-frame gate."""
import json,hashlib,sys
from fractions import Fraction as F
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT/'artifacts/submissions/verify_v6/model/stage2/code'))
from solution import stage2_v2 as v2
from trace_selection import grade
read=lambda p:json.loads(p.read_text())
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()

def strict_state(raw):
    def pairs(items):
        obj={}
        for k,v in items:
            if k in obj:raise ValueError('duplicate key')
            obj[k]=v
        return obj
    def reject(value):raise ValueError('invalid JSON constant')
    try:o=json.loads(raw,object_pairs_hook=pairs,parse_constant=reject)
    except (ValueError,TypeError):return None
    s=o.get('lane_state') if isinstance(o,dict) and set(o)=={'lane_state'} else None
    return s if s in ['INSIDE','OUTSIDE','UNCERTAIN'] else None

def paired(a,b,lo,hi):
    a,b,lo,hi=[F(str(x)) for x in [a,b,lo,hi]];t=F(3,10)
    cuts=sorted({lo,hi,*[x for p in [a,b] for x in [p-t,p+t] if lo<=x<=hi]});samples=cuts+[(x+y)/2 for x,y in zip(cuts,cuts[1:])]
    delta=[int(abs(b-x)<=t)-int(abs(a-x)<=t) for x in samples]
    points=[lo,hi]+[p for p in [a,b] if lo<=p<=hi];mae=[abs(b-x)-abs(a-x) for x in points]
    return dict(accuracy_delta=[min(delta),max(delta)],mae_delta_seconds=[float(min(mae)),float(max(mae))])

def aggregate(rows,arm):
    n=len(rows);counts={k:sum(r[arm]['result']==k for r in rows) for k in ['correct','wrong','indeterminate']}
    return dict(n=n,**counts,accuracy_bounds=[counts['correct']/n,(counts['correct']+counts['indeterminate'])/n] if n else None,
      mae_bounds_seconds=[sum(r[arm][key] for r in rows)/n for key in ['minimum_error_seconds','maximum_error_seconds']] if n else None)

assert strict_state('{"lane_state":"INSIDE"}')=='INSIDE'
assert strict_state('{"lane_state":"INSIDE","lane_state":"OUTSIDE"}') is None
assert paired(1,0,0,0)['accuracy_delta']==[1,1]
assert paired(1,1,0,2)['accuracy_delta']==[0,0]

def main():
    run=read(HERE/'run/report.json');assert run['status']=='complete' and len(run['workers'])==16 and all(w['exit_status']==0 for w in run['workers'])
    frozen=read(HERE/'freeze.json')
    for name,digest in frozen['files'].items():assert sha(ROOT/name)==digest,name
    inventory={r['ID']:r for r in read(HERE/'inventory.json')['cases']};traces={r['ID']:r for r in read(HERE/'path_replay.json')['rows']};rows=[]
    for ref in read(HERE/'references.json')['cases']:
        sid=ref['ID'];src=inventory[sid];trace=traces[sid];history=read(ROOT/src['baseline_folder']/'calls.json')[2];records={}
        for arm in ['control','gate']:
            folder=HERE/'run'/f'{sid}_{arm}';r=read(folder/'result.json');w=read(folder/'worker_report.json');calls=read(folder/'calls.json');job=read(HERE/'inputs'/f'{sid}_{arm}.job.json')
            assert r['ID']==w['ID']==job['ID']==sid and w['status']=='complete' and w['model_calls']==len(calls)==1 and w['network_attempts']==0
            assert calls[0]==r['call'] and calls[0]['text'].strip()==r['raw'].strip()
            assert calls[0]['prompt']==job['prompt'] and calls[0]['max_new_tokens']==40 and calls[0]['image_sizes']==[[1536,768]]
            assert calls[0]['deepstack_fix'] and calls[0]['compute_dtype']=='native' and calls[0]['decode_mode']=='sync'
            records[arm]=r
        c,g=records['control'],records['gate']
        for key in ['prompt','max_new_tokens','image_sha256','image_sizes','processor_input_sha256']:assert c['call'][key]==history[key]
        paths=[ROOT/r['path'] for r in src['source']['images']];nums=[r['frame'] for r in src['source']['images']]
        value=v2._json_object(c['raw']);idx=v2._choice(value,'entry_frame',paths,[nums.index(f) for f in trace['entry_candidates']],0)
        base={**trace['baseline'],'entry_frame':nums[idx]};state=strict_state(g['raw']);cand={**base,'entry_frame':nums[0] if state=='INSIDE' else base['entry_frame']}
        row=dict(ID=sid,reference=ref['entry'],eligible=ref['eligible'],baseline=base,candidate=cand,state=state,state_raw=g['raw'],
            control_raw_matches_history=c['raw'].strip()==history['text'].strip(),control_entry_matches_history=base['entry_frame']==trace['final_entry'],
            control_processor_exact=True,other_three_unchanged=all(base[k]==cand[k] for k in ['collision_frame','entry_side','evasion_space']),entry_changed=base['entry_frame']!=cand['entry_frame'])
        if ref['eligible']:
            times=trace['original_times'];lo,hi=[times[str(ref['entry'][k])] for k in ['lower_frame','upper_frame']];a,b=times[str(base['entry_frame'])],times[str(cand['entry_frame'])]
            row['timing']=dict(baseline=grade(a,lo,hi),candidate=grade(b,lo,hi),**paired(a,b,lo,hi),reference_seconds=[lo,hi])
            row['new_false_first']=ref['entry']['status']=='during_clip' and cand['entry_frame']==nums[0] and base['entry_frame']!=nums[0]
        else:row['timing']=None;row['new_false_first']=None
        rows.append(row)
    good=[r for r in rows if r['eligible']];n=len(good);timings=[r['timing'] for r in good]
    metrics={a:aggregate(timings,a) for a in ['baseline','candidate']}
    deltas={k:[sum(t[k][i] for t in timings)/n for i in [0,1]] for k in ['accuracy_delta','mae_delta_seconds']}
    gate=dict(gain_ids=[r['ID'] for r in good if r['timing']['baseline']['result']=='wrong' and r['timing']['candidate']['result']=='correct'],
        definite_loss_ids=[r['ID'] for r in good if r['timing']['baseline']['result']=='correct' and r['timing']['candidate']['result']=='wrong'],
        possible_loss_ids=[r['ID'] for r in good if r['timing']['accuracy_delta'][0]<0],new_false_first_ids=[r['ID'] for r in good if r['new_false_first']],
        unsupported_identity_commit_ids=[r['ID'] for r in rows if r['ID']=='CCD_000052' and r['state']=='INSIDE'],
        controls_reproduced=all(r['control_raw_matches_history'] and r['control_entry_matches_history'] for r in rows),other_three_preserved=all(r['other_three_unchanged'] for r in rows),
        both_strata={r['reference']['status'] for r in good}=={'before_start','during_clip'})
    gate['pass']=bool(gate['gain_ids']) and not any(gate[k] for k in ['definite_loss_ids','possible_loss_ids','new_false_first_ids','unsupported_identity_commit_ids']) and gate['controls_reproduced'] and gate['other_three_preserved'] and gate['both_strata'] and deltas['mae_delta_seconds'][1]<=0
    output=dict(status='complete',actual_model_calls=16,rows=rows,metrics=metrics,paired_delta_bounds=deltas,gate=gate,
        strata={s:{a:aggregate([r['timing'] for r in good if r['reference']['status']==s],a) for a in ['baseline','candidate']} for s in ['before_start','during_clip']},
        official_S2=None,scope='One automatic candidate on eight exposed AI-reviewed cases. Fresh Q3 controls; three other outputs cached and preserved. Not full fresh four-query inference, CUDA equivalence or independent generalization.')
    (HERE/'evaluation.json').write_text(json.dumps(output,ensure_ascii=False,indent=2)+'\n');print(json.dumps({k:output[k] for k in ['metrics','paired_delta_bounds','gate']},ensure_ascii=False))

if __name__=='__main__':main()
