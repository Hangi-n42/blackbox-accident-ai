"""Rebuild frozen VLM RGB inputs using recorded answers; never load a model."""
from pathlib import Path
import hashlib,json,math,sys
import numpy as np
from PIL import Image,ImageDraw
ROOT=Path(__file__).resolve().parents[3];OUT=Path(__file__).resolve().parent
RELEASE=ROOT/'artifacts/submissions/verify_v6/model/stage2/code'
sys.path.insert(0,str(RELEASE))
from solution import stage2_v2
BASE=ROOT/'artifacts/stage2_goal_20260919/current_baseline'
inputs={r['ID']:r for r in json.loads((BASE/'inputs.json').read_text())}
labels={r['ID']:r for r in json.loads((ROOT/'artifacts/mac_experiments/baseline_20260916/human_labels.json').read_text())}
def sha(p):
 with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
allrows=[]
for sid in ['00007','00010']:
 d=OUT/sid;d.mkdir(exist_ok=True);run=BASE/'run/stage2/human_dev'/sid
 calls=json.loads((run/'calls.json').read_text());result=json.loads((run/'result.json').read_text());motion=np.load(run/'motion.npz')
 paths=[ROOT/x['path'] for x in inputs[sid]['images']]
 assert all(sha(p)==x['sha256'] for p,x in zip(paths,inputs[sid]['images']))
 assert hashlib.sha256(motion['base_scores'].tobytes()).hexdigest()==result['diagnostics']['uncapped_jerk']['base_score_sha256']
 class Replay:
  def __init__(self):self.i=0;self.rows=[]
  def ask(self,images,prompt,max_new_tokens=128):
   rec=calls[self.i];self.i+=1
   assert rec['prompt']==prompt and rec['max_new_tokens']==max_new_tokens
   actual=[]
   for j,im in enumerate(images):
    im.save(d/f'call{self.i}_raw.png');im=im.convert('RGB');budget=max(1024,1200000//len(images));scale=min(1.0,math.sqrt(budget/(im.width*im.height)));size=(max(32,int(im.width*scale)//32*32),max(32,int(im.height*scale)//32*32));bound=im.resize(size,Image.Resampling.BICUBIC)
    digest=hashlib.sha256(bound.tobytes()).hexdigest();assert digest==rec['image_sha256'][j] and list(size)==rec['image_sizes'][j]
    bound.save(d/f'call{self.i}_bounded.png');actual.append({'size':list(size),'rgb_sha256':digest,'cached_sha256':rec['image_sha256'][j],'match':True})
   self.rows.append({'call':self.i,'response':rec['text'],'images':actual,'prompt_match':True})
   return rec['text']
 replay=Replay();internal,diag=stage2_v2._predict_file(paths,motion['base_scores'],replay);assert replay.i==4
 review=labels[sid]['draft']['review'];gt=review['contact']['frame'];final=result['prediction']['collision_frame'];vlm=internal['collision_frame']
 wanted=sorted(set([0,1,2,max(0,final-2),final,min(len(paths)-1,final+2),gt-3,gt,gt+3,vlm]))
 canvas=Image.new('RGB',(1920,390*math.ceil(len(wanted)/3)));draw=ImageDraw.Draw(canvas)
 for k,n in enumerate(wanted):
  im=Image.open(paths[n]).convert('RGB');im.save(d/f'native_{n:06d}.png');im.thumbnail((640,360));x=k%3*640;y=k//3*390;canvas.paste(im,(x,y));draw.text((x+4,y+363),f'{sid} f{n} native PTS {inputs[sid]["images"][n]["pts_seconds"]:.6f}',fill='white')
 canvas.save(d/'native_context.png')
 row={'ID':sid,'cache':'fresh current_baseline/run','calls_json_sha256':sha(run/'calls.json'),'result_json_sha256':sha(run/'result.json'),'input_hashes_all_match':True,'four_call_rgb_hashes_match':True,'call_reconstruction':replay.rows,'internal_prediction':internal,'final_prediction':result['prediction'],'diagnostics':result['diagnostics'],'human_draft_contact':review['contact'],'human_draft_entry':review['entry'],'human_draft_target':review['target'],'human_draft_evaluation_eligible':False,'context_native_indices':wanted,'new_model_calls':0,'production_changes':0,'base_score_hash_match':True}
 (d/'reconstruction.json').write_text(json.dumps(row,ensure_ascii=False,indent=2));allrows.append(row)
(OUT/'reconstruction_summary.json').write_text(json.dumps(allrows,ensure_ascii=False,indent=2));print('two cases, eight calls: exact cached RGB hashes and prompts match')
