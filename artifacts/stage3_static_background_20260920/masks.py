"""AI RGB-reviewed conservative static-surface regions, frozen before PnP."""
from pathlib import Path
import json,hashlib
import numpy as np
import cv2
from PIL import Image,ImageDraw
O=Path(__file__).resolve().parent;R=O.parents[1];B=R/'artifacts/stage3_depth_pnp_20260920'
POLYGONS={
'expanded_11_225':[[[.06,.75],[.38,.58],[.55,.58],[.61,.70],[.67,.75]],[[.08,.57],[.22,.51],[.25,.53],[.08,.65]]],
'expanded_14_215':[[[.10,.75],[.42,.58],[.63,.58],[.81,.67],[.93,.75]]],
'expanded_19_275':[[[.08,.75],[.32,.62],[.61,.62],[.92,.70],[.92,.76]]],
'expanded_14_555':[[[.08,.75],[.42,.56],[.54,.54],[.60,.59],[.75,.69],[.87,.75]]],
'expanded_11_525':[[[.46,.60],[.59,.56],[.65,.59],[.90,.73],[.90,.74],[.46,.74]],[[.80,.425],[.89,.425],[.89,.455],[.80,.455]]],
'expanded_09_335':[[[.08,.73],[.43,.53],[.53,.53],[.66,.58],[.90,.74]],[[.08,.56],[.39,.48],[.40,.50],[.08,.64]]],
'zod_000002_140':[[[.14,.73],[.46,.55],[.50,.55],[.50,.67],[.66,.75],[.35,.75]]],
'zod_000026_161':[[[.08,.70],[.45,.54],[.55,.54],[.91,.70],[.85,.75],[.15,.75]]]
}

def mask_for(key,j,h,w):
 m=np.zeros((h,w),np.uint8)
 polys=POLYGONS[key].copy()
 if key=='zod_000002_140' and j>=8:polys=polys+[[[.69,.61],[.91,.66],[.91,.71],[.70,.64]]]
 for poly in polys:cv2.fillPoly(m,[np.rint(np.array(poly)*[w-1,h-1]).astype(np.int32)],1)
 if key=='zod_000026_161':
  cv2.rectangle(m,(int(.45*w),int(.50*h)),(int(.56*w),int(.62*h)),0,-1)
  if j in [5,11]:m[:]=0
  if j>=19:cv2.rectangle(m,(int(.75*w),int(.60*h)),(w-1,h-1),0,-1)
 return cv2.erode(m,np.ones((7,7),np.uint8))

def inside(mask,p):
 x=np.rint(p[:,0]).astype(int);y=np.rint(p[:,1]).astype(int);h,w=mask.shape;valid=(x>=0)&(x<w)&(y>=0)&(y<h);out=np.zeros(len(p),bool);out[valid]=mask[y[valid],x[valid]]>0;return out

def main():
 (O/'masks').mkdir(exist_ok=True);(O/'overlays').mkdir(exist_ok=True)
 for key in POLYGONS:
  rgb=np.load(B/'depth'/(key+'_input.npz'))['rgb'];flow=np.load(B/'flow'/(key+'.npz'));h,w=rgb.shape[1:3];m=np.array([mask_for(key,j,h,w) for j in range(21)]);np.savez_compressed(O/'masks'/(key+'.npz'),mask=m)
  sheet=Image.new('RGB',(1008,1932),'white');draw=ImageDraw.Draw(sheet)
  for j,im in enumerate(rgb):
   pic=im.copy();pic[m[j]>0]=(pic[m[j]>0]*.75+np.array([0,255,0])*.25).astype('uint8')
   count=0
   if j<20:
    p,q=flow[f'p{j}'],flow[f'q{j}'];keep=inside(m[j],p)&inside(m[j+1],q);count=int(keep.sum())
    for xy in p[keep]:cv2.circle(pic,tuple(np.rint(xy).astype(int)),2,(255,255,0),-1)
   x=j%3*336;y=j//3*276;sheet.paste(Image.fromarray(pic).resize((336,252)),(x,y+24));draw.text((x+3,y+4),f'{key} f{j} kept={count}',fill='black')
  sheet.save(O/'overlays'/(key+'.jpg'))
 print('Masks and pre-PnP overlays created; no PnP evaluation run.')
if __name__=='__main__':main()
