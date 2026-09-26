from run import *
from PIL import Image,ImageDraw
(O/'anchors').mkdir(exist_ok=True)
for old in read(B/'results.json'):
 key=old['key'];rgb=np.load(B/'depth'/(key+'_input.npz'))['rgb'];sheet=Image.new('RGB',(1008,3*408),'white');d=ImageDraw.Draw(sheet)
 for n,j in enumerate([0,5,10,15,20]):
  im=Image.fromarray(rgb[j]).resize((504,378));dd=ImageDraw.Draw(im)
  for u in [.1,.2,.3,.4,.5,.6,.7,.8,.9]:dd.line((u*504,0,u*504,378),fill=(100,80,70));dd.text((u*504+1,2),str(u),fill='yellow')
  for v in [.3,.4,.5,.6,.7,.8]:dd.line((0,v*378,504,v*378),fill=(100,80,70));dd.text((1,v*378+1),str(v),fill='yellow')
  x=n%2*504;y=n//2*408;sheet.paste(im,(x,y+24));d.text((x+3,y+2),f'{key} f{j}',fill='black')
 sheet.save(O/'anchors'/(key+'.jpg'))
