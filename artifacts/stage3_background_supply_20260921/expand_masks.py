"""Expand only RGB-reviewed road and rigid structure interiors. No prediction input."""
from run import *
from PIL import Image,ImageDraw
(O/'masks').mkdir(exist_ok=True);(O/'mask_review').mkdir(exist_ok=True)
P={
'expanded_11_225':[[[.08,.75],[.08,.59],[.43,.52],[.56,.53],[.78,.73],[.78,.75]]],
'expanded_14_215':[[[.08,.75],[.45,.52],[.54,.52],[.86,.64],[.92,.75]]],
'expanded_19_275':[[[.08,.75],[.08,.62],[.38,.55],[.58,.55],[.91,.58],[.92,.75]],[[.855,.345],[.92,.325],[.92,.405],[.855,.410]]],
'expanded_14_555':[[[.08,.75],[.08,.65],[.42,.52],[.54,.49],[.58,.53],[.88,.67],[.90,.75]]],
'expanded_11_525':[[[.46,.74],[.46,.54],[.59,.50],[.67,.55],[.92,.65],[.92,.75]],[[.81,.392],[.885,.392],[.885,.445],[.81,.445]]],
'expanded_09_335':[[[.08,.74],[.44,.47],[.58,.46],[.73,.54],[.92,.59],[.92,.74]],[[.08,.48],[.45,.412],[.46,.443],[.08,.62]]],
'zod_000002_140':[[[.30,.74],[.46,.545],[.51,.545],[.52,.675],[.67,.74]]],
'zod_000026_161':[[[.08,.52],[.43,.495],[.43,.535],[.08,.655]],[[.58,.505],[.92,.545],[.92,.655],[.58,.55]]]
}
rows=[]
for key in P:
 rgb=np.load(B/'depth'/(key+'_input.npz'))['rgb'];old=np.load(S/'masks'/(key+'.npz'))['mask'];h,w=old.shape[1:];m=[]
 for j in range(21):
  added=np.zeros((h,w),np.uint8)
  for poly in P[key]:cv2.fillPoly(added,[np.rint(np.array(poly)*[w-1,h-1]).astype('int32')],1)
  if key=='expanded_09_335':
   # Roadside sign face; linear location interpolation from five reviewed anchor frames.
   left=np.interp(j,[0,5,10,15,20],[.682,.690,.704,.720,.743]);right=left+.022
   cv2.rectangle(added,(round(left*w),round(.302*h)),(round(right*w),round(.375*h)),1,-1)
  added=cv2.erode(added,np.ones((7,7),np.uint8));combined=np.maximum(old[j],added)
  if key=='zod_000026_161':
   if j in [5,11]:combined[:]=0
   if j>=19:cv2.rectangle(combined,(int(.75*w),int(.60*h)),(w-1,h-1),0,-1)
  m.append(combined)
 m=np.array(m);np.savez_compressed(O/'masks'/(key+'.npz'),mask=m)
 sheet=Image.new('RGB',(1008,1932),'white');d=ImageDraw.Draw(sheet)
 for j,im in enumerate(rgb):
  im=im.copy();im[m[j]>0]=(im[m[j]>0]*.7+np.array([0,255,0])*.3).astype('uint8');x=j%3*336;y=j//3*276;sheet.paste(Image.fromarray(im).resize((336,252)),(x,y+24));d.text((x+3,y+3),f'{key} frame{j}',fill='black')
 sheet.save(O/'mask_review'/(key+'.jpg'));rows.append({'key':key,'polygons_added':P[key],'original_mask_preserved':True})
write(O/'mask_annotations.json',{'reviewer':'AI, not human; original and anchor RGB inspected, all-frame overlay review required before measurements','frames':168,'excludes':'Vehicles, hood, sky, foliage, known wipers. Shadows/reflections may remain. Not exhaustive static segmentation.','rows':rows})
