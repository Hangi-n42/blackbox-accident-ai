"""Read-only tokenizer serialization/encoding audit; no model execution."""
from pathlib import Path
import json,hashlib
from tokenizers import Tokenizer
ROOT=Path(__file__).resolve().parents[2]
paths=[ROOT/'artifacts/candidates/qwen3_vl_4b/tokenizer.json',ROOT/'artifacts/submissions/verify_v6/model/stage2/vlm/tokenizer.json']
docs=[json.loads(p.read_text('utf8')) for p in paths]
tokenizers=[Tokenizer.from_file(str(p)) for p in paths]
samples=['Return JSON with collision_frame only. Allowed event frames: [0, 1, 584, 1256].','<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|> Choose the first wheel touching the lane.\n<|im_end|>','{"entry_side":"RIGHT","evasion_space":1}', '접촉 시점·진입 방향: 왼쪽과 오른쪽', "camera car's lane; before/after.\n0123456789"]
def merges(d):return [v if isinstance(v,list) else v.split(' ') for v in d['model']['merges']]
result=dict(status='PASS' if all(tokenizers[0].encode(s).ids==tokenizers[1].encode(s).ids for s in samples) else 'FAIL',
    paths=[str(p) for p in paths],sha256=[hashlib.sha256(p.read_bytes()).hexdigest() for p in paths],
    pretokenizer_equal=docs[0]['pre_tokenizer']==docs[1]['pre_tokenizer'],vocab_equal=docs[0]['model']['vocab']==docs[1]['model']['vocab'],
    normalized_merges_equal=merges(docs[0])==merges(docs[1]),encoding_pairs=[dict(text=s,ids_equal=tokenizers[0].encode(s).ids==tokenizers[1].encode(s).ids) for s in samples],
    warning_origin='Local Transformers4.57.6 checks model_type only when saved transformers_version<=4.57.2. Saved Qwen3_VL config version4.57.6 falls through to Mistral warning without establishing Mistral identity.',
    interpretation='Warning alone is not evidence that V6 tokenization differs from original Qwen assets. Do not apply Mistral regex to Qwen merely to suppress this warning.',model_or_tokenizer_changed=False)
with (Path(__file__).parent/'tokenizer_audit.json').open('x',encoding='utf8') as f:json.dump(result,f,ensure_ascii=False,indent=2)
print(json.dumps({k:v for k,v in result.items() if k in ('status','pretokenizer_equal','vocab_equal','normalized_merges_equal')}))
