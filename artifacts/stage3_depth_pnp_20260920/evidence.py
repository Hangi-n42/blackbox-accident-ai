"""Compact source/depth/PnP evidence; plotting only, no image alteration of data."""
from prepare import *
import cv2
from PIL import ImageDraw,ImageFont
font=ImageFont.truetype('/System/Library/Fonts/Supplemental/Arial.ttf',15)
out=O/'evidence';out.mkdir(exist_ok=True)
for r in read(O/'results.json'):
 key=r['key'];inp=np.load(O/'depth'/(key+'_input.npz'));dep=np.load(O/'depth'/(key+'_first.npz'))['depth'];flow=np.load(O/'flow'/(key+'.npz'));lo,hi=np.quantile(np.log(dep),[.05,.95])
 sheet=Image.new('RGB',(1200,780),'white');draw=ImageDraw.Draw(sheet)
 draw.text((8,4),f"{key} label={r['label']} sensor q={r['sensor_q']:+.4f}/s valid={r['valid']}",font=font,fill='black')
 for col,j in enumerate([0,10,19]):
  im=Image.fromarray(inp['rgb'][j]).resize((400,225));sheet.paste(im,(400*col,30))
  color=cv2.applyColorMap(((np.log(dep[j])-lo)/(hi-lo)*255).clip(0,255).astype('uint8'),cv2.COLORMAP_TURBO)[:,:,::-1]
  sheet.paste(Image.fromarray(color).resize((400,225)),(400*col,270))
  im=Image.fromarray(inp['rgb'][j]);dd=ImageDraw.Draw(im);p=flow[f'p{j}'];q=flow[f'q{j}'];pair=r['features']['first']['pairs'][j];inliers=set(pair.get('inlier_indices',[]))
  for ii,(a,b) in enumerate(zip(p,q)):
   color='lime' if ii in inliers else 'red';dd.line((float(a[0]),float(a[1]),float(b[0]),float(b[1])),fill=color,width=1);dd.ellipse((float(a[0])-1,float(a[1])-1,float(a[0])+1,float(a[1])+1),fill=color)
  sheet.paste(im.resize((400,225)),(400*col,510));draw.text((400*col+4,739),f"pair{j}: n={pair['points']} inliers={pair.get('inliers',0)} valid={pair['valid']}",font=font,fill='black')
 draw.text((8,761),'Rows: RGB / log-depth (one color scale per clip) / fixed DIS matches, RANSAC inliers green',font=font,fill='black')
 sheet.save(out/(key+'.jpg'))
print('8 source/depth/match sheets saved')
