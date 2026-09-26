from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
R=Path(__file__).resolve().parents[2];B=R/'artifacts/stage3_depth_pnp_20260920';O=Path(__file__).resolve().parent
(O/'review').mkdir(exist_ok=True)
for p in sorted((B/'depth').glob('*_input.npz')):
 key=p.name.removesuffix('_input.npz');a=np.load(p)['rgb'];sheet=Image.new('RGB',(1008,7*276),'white');d=ImageDraw.Draw(sheet)
 for j,im in enumerate(a):
  x=j%3*336;y=j//3*276;sheet.paste(Image.fromarray(im).resize((336,252)),(x,y+24));d.text((x+4,y+4),f'{key} frame {j}',fill='black')
 sheet.save(O/'review'/(key+'.jpg'))
print('8 sheets, all 168 frames')
