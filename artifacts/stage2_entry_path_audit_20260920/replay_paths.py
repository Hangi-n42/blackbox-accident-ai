"""Replay cached answers through unchanged V6 policy, without model inference."""
import hashlib,json,math,sys
from pathlib import Path
import numpy as np
from PIL import Image
ROOT=Path(__file__).resolve().parents[2];HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'artifacts/submissions/verify_v6/model/stage2/code'))
from solution import stage2_v2 as v2
from solution import stage2_uncapped_jerk_v6c as policy
read=lambda p:json.loads(p.read_text())
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()

def bounded(im):
    scale=min(1.,math.sqrt(1200000/(im.width*im.height)))
    return im.convert('RGB').resize(tuple(max(32,int(v*scale)//32*32) for v in im.size),Image.Resampling.BICUBIC)

rows=[]
for row in read(HERE/'inventory.json')['cases']:
    source=row['source'];paths=[ROOT/r['path'] for r in source['images']];numbers=[v2._frame_number(p) for p in paths]
    for p,r in zip(paths,source['images']):assert sha(p)==r['sha256']
    folder=ROOT/row['baseline_folder'];old=read(folder/'result.json');calls=read(folder/'calls.json')
    for name,digest in row['baseline_sha256'].items():assert sha(folder/name)==digest
    class Replay:
        count=0
        def ask(self,images,prompt,max_new_tokens):
            c=calls[self.count];self.count+=1
            assert c['prompt']==prompt and c['max_new_tokens']==max_new_tokens
            assert [hashlib.sha256(bounded(im).tobytes()).hexdigest() for im in images]==c['image_sha256']
            if self.count==3:
                out=HERE/'q3_inputs';out.mkdir(exist_ok=True);images[0].save(out/f"{row['ID']}.png")
            return c['text']
    replay=Replay()
    with np.load(folder/'motion.npz',allow_pickle=False) as motion:pred,diag=policy._predict_file(paths,motion['base_scores'],motion['new_scores'],replay)
    assert replay.count==len(calls)==4 and pred==old['baseline_prediction']
    assert diag['entry_candidates']==old['diagnostics']['entry_candidates']
    rawcontact=diag['collision'];contact_index=v2._choice(rawcontact,'collision_frame',paths,[numbers.index(f) for f in diag['collision_candidates']],numbers.index(diag['motion_proposal']))
    expected=v2._uniform_indices(0,contact_index,12)
    assert [numbers[i] for i in expected]==diag['entry_candidates']
    rawentry=diag['entry'].get('entry_frame');valid=type(rawentry) is int and rawentry in diag['entry_candidates']
    rows.append(dict(ID=row['ID'],review_selected=row['review_selected'],source_group=source['source_group'],all_frames=numbers,
        original_times={r['frame']:r['pts_seconds'] for r in source['images']},vlm_contact_frame=numbers[contact_index],
        final_contact_frame=pred['collision_frame'],precontact_frames=numbers[:contact_index+1],entry_candidates=diag['entry_candidates'],
        final_entry=pred['entry_frame'],raw_entry=rawentry,raw_entry_is_shown_integer=valid,choice_fallback_or_mapping=not valid,
        all_four_cached_calls_exact=True,baseline=pred,q3_prompt=calls[2]['prompt'],q3_image_sizes=calls[2]['image_sizes'],q3_prompt_tokens=calls[2]['prompt_tokens'],
        replay_source_sha256=sha(Path(policy.__file__)),sheet_source_sha256=sha(Path(v2.__file__))))
out=dict(status='PASS',model_calls=0,cached_calls_replayed=96,rows=rows,scope='Same policy source and image/prompt replay, different MLX model from CUDA NF4; not a fresh submitted-server evaluation.')
(HERE/'path_replay.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n')
print(json.dumps([{k:r[k] for k in ['ID','vlm_contact_frame','final_contact_frame','entry_candidates','final_entry','raw_entry_is_shown_integer']} for r in rows if r['review_selected']],indent=2))
