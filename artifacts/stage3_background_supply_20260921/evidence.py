from run import *
from PIL import Image,ImageDraw
(O/'evidence').mkdir(exist_ok=True)
for row in read(O/'source_pair/results.json'):
 key=row['key'];rgb=np.load(B/'depth'/(key+'_input.npz'))['rgb'];pts=np.load(O/'source_pair'/(key+'.npz'));sheet=Image.new('RGB',(1008,3*408),'white');dd=ImageDraw.Draw(sheet)
 for n,j in enumerate([0,10,19]):
  p,q=pts[f'DIS_p{j}'],pts[f'DIS_q{j}'];pair=row['methods']['DIS']['pnp'][j];ii=set(pair.get('inlier_indices',[]))
  for col,(frame,xy) in enumerate([(j,p),(j+1,q)]):
   im=Image.fromarray(rgb[frame]);d=ImageDraw.Draw(im)
   for i,z in enumerate(xy):
    color='lime' if i in ii else 'orange';x,y=map(float,z);d.ellipse((x-1.2,y-1.2,x+1.2,y+1.2),fill=color)
   sheet.paste(im.resize((504,378)),(col*504,n*408+24))
  dd.text((4,n*408+4),f'{key} pair{j} n={len(p)} PnP={pair["valid"]} reason={pair["reason"]}',fill='black')
 sheet.save(O/'evidence'/(key+'.jpg'))
